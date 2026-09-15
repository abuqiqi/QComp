"""验证流式局部 NMSE 的数学一致性、单次前向和异常清理。"""
import torch
import pytest
from torch import nn
from qcomp import CompressionPlan, CompressionTarget, MPOSpec, get_backend
from qcomp.evaluation.local_output_nmse import local_output_error_sums
from qcomp.workflows.streaming_nmse import prepare_candidates, measure_batch
from qcomp.evaluation.nmse_inputs import encode_input
from scripts.run_qwen3_local_output_nmse import validate_resume, parse_args


class Tiny(nn.Module):
    """包含两个 block 的确定性 baseline。"""
    def __init__(self):
        """初始化投影与计数。"""
        super().__init__()
        torch.manual_seed(17)
        self.embedding = nn.Embedding(8,4)
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(4,4,bias=False)) for _ in range(2)])
        self.calls = 0
    def forward(self,input_ids,attention_mask,use_cache):
        """依次执行两个 block。"""
        self.calls += 1
        value = self.embedding(input_ids)
        for block in self.blocks:
            value = block(value)
        return value


def test_streaming_matches_direct_and_cleans_hooks(monkeypatch):
    """一次前向返回准确统计，失败清理全部 hooks。"""
    model = Tiny()
    spec = MPOSpec((2,2),(2,2),(1,1,1))
    plans = {str(i):CompressionPlan((CompressionTarget(f'blocks.{i}.0','mpo',spec),)) for i in range(2)}
    backend = get_backend('native','mpo')
    decompositions = []
    original_decompose = backend.decompose
    def counted(*args):
        """记录真实分解调用，确认跨批候选被复用。"""
        decompositions.append(1)
        return original_decompose(*args)
    monkeypatch.setattr(backend, 'decompose', counted)
    candidates,_ = prepare_candidates(model,plans,{'mpo':backend},{'mpo':backend},torch.float32,42)
    batch = {'input_ids':torch.tensor([[1,2],[3,0]]),'attention_mask':torch.tensor([[1,1],[1,0]])}
    expected = {}
    with torch.no_grad():
        value = model.embedding(batch['input_ids'])
        for i,block in enumerate(model.blocks):
            ref = block(value)
            e,p,_ = local_output_error_sums(ref,candidates[str(i)](value),batch['attention_mask'])
            expected[str(i)] = [float(e),float(p)]
            value = ref
        reference = value.clone()
    outputs=[]
    handle=model.register_forward_hook(lambda m,a,o:outputs.append(o.clone()))
    groups={f'blocks.{i}.0':f'blocks.{i}' for i in range(2)}
    result=measure_batch(model,model,plans,candidates,groups,batch)
    handle.remove()
    assert model.calls==1
    assert len(decompositions)==2
    assert torch.equal(outputs[0],reference)
    assert result==expected
    assert all(not m._forward_hooks for m in model.modules())
    class Broken(nn.Module):
        """模拟失败候选。"""
        def forward(self,value):
            """抛出执行异常。"""
            raise RuntimeError('failed')
    candidates['1']=Broken()
    with pytest.raises(RuntimeError):
        measure_batch(model,model,plans,candidates,groups,batch)
    assert all(not m._forward_hooks for m in model.modules())


def test_encoding_shift_and_resume_identity():
    """验证因果移位、截断及恢复身份。"""
    class Encoder:
        """提供固定 token。"""
        def _encode_pair(self,context,continuation):
            """返回固定上下文与续写。"""
            return [1,2,3],[4,5]
    assert encode_input(Encoder(),'q','a',3)==[2,3,4]
    state={'format_version':1,'identity':{'x':1}}
    validate_resume(state,{'x':1})
    with pytest.raises(ValueError):
        validate_resume(state,{'x':2})
    with pytest.raises(SystemExit):
        parse_args(['--baseline-cache-mode','memory'])


def test_task_batches_deduplicate_and_reference_answers():
    """验证同题去重、题目计数、标准答案选择和可重放输入摘要。"""
    from types import SimpleNamespace
    from qcomp.evaluation.nmse_inputs import TaskInputs
    from qcomp.evaluation.lm_eval import LMEvalConfig
    class Encoder:
        """用字符编码保持边界可观察。"""
        def _encode_pair(self,context,continuation):
            """将上下文和续写映射为字符序号。"""
            return list(map(ord,context)),list(map(ord,continuation))
    class Task:
        """提供两题、两个选项及生成类字段。"""
        def __init__(self,answer):
            """保存确定性题目。"""
            self.eval_docs=[{'answer':answer},{'answer':answer}]
            self.config=SimpleNamespace(repeats=1)
        def set_fewshot_seed(self,seed):
            """固定模板不需要随机数。"""
        def get_config(self,name):
            """返回测试所需模板参数。"""
            return {'num_fewshot':0,'target_delimiter':' '}[name]
        def fewshot_context(self,doc,**kwargs):
            """固定上下文。"""
            return 'Q:'
        def doc_to_prefix(self,doc):
            """测试不使用生成前缀。"""
            return None
        def construct_requests(self,**kwargs):
            """两个单 token 选项对应相同的真实模型输入。"""
            return [SimpleNamespace(args=('Q:','A')),SimpleNamespace(args=('Q:','B'))]
    stream=TaskInputs.__new__(TaskInputs)
    stream.config=LMEvalConfig(task='mmlu',batch_size=1)
    stream.strategy='likelihood'
    stream.encoder=Encoder()
    stream.tokenizer=SimpleNamespace(pad_token_id=0)
    stream.tasks={'test':Task('solution')}
    batches=list(stream.batches())
    assert len(batches)==2
    assert sum(b['examples'] for b in batches)==2
    assert sum(b['logical_requests'] for b in batches)==4
    assert batches[0]['input_ids'].tolist()==[[81,58]]
    assert batches[-1]['digest']==list(stream.batches())[-1]['digest']
    stream.strategy='reference_answer'
    for name,answer,expected in [('gsm8k','work #### 4','Q: work #### '),('triviaqa',{'value':'Paris','aliases':['PARIS','paris']},'Q: Pari')]:
        stream.tasks={name:Task(answer)}
        b=next(stream.batches())
        assert b['input_ids'].tolist()==[list(map(ord,expected))]
        assert b['logical_requests']==1


def test_main_resume_commits_only_complete_batches(tmp_path, monkeypatch):
    """中途失败后恢复跳过已提交输入，最终统计与连续运行一致。"""
    import json
    from types import SimpleNamespace
    from scripts import run_qwen3_local_output_nmse as script
    model=Tiny()
    weights=tmp_path/'weights';weights.mkdir();(weights/'x.safetensors').write_bytes(b'identity')
    config={'retention_ratios':[.4], 'tasks':[{'name':'tiny','evaluation':{'task':'tiny'},
        'input_strategy':'likelihood','historical_results':str(tmp_path/'history.json')}]}
    (tmp_path/'history.json').write_text('{}')
    path=tmp_path/'config.json';path.write_text(json.dumps(config))
    plan=CompressionPlan((CompressionTarget('blocks.0.0','mpo',MPOSpec((2,2),(2,2),(1,1,1))),))
    record={'name':'c','module_path':'blocks.0.0','block':0,'projection':'q_proj','rank':1,
        'labels':['rho:0.4'],'requested_retention_ratio':.4,'actual_retention_ratio':.5,
        'dense_parameters':16,'compressed_parameters':8}
    monkeypatch.setattr(script,'load_runtime_config',lambda p:SimpleNamespace(model_name_or_path=str(weights)))
    tokenizer=SimpleNamespace(backend_tokenizer=SimpleNamespace(to_str=lambda:'tokenizer'))
    monkeypatch.setattr(script,'load_causal_lm',lambda *a,**k:SimpleNamespace(model=model,tokenizer=tokenizer))
    monkeypatch.setattr(script,'select_qwen3_targets',lambda *a,**k:('blocks.0.0',))
    monkeypatch.setattr(script,'build_candidates',lambda *a:({'c':plan},{'c':dict(record)},{}))
    monkeypatch.setattr(script.CompressionExecutionConfig,'build_backends',lambda *a:({},{}))
    monkeypatch.setattr(script,'prepare_candidates',lambda *a:({'c':nn.Linear(1,1)},{'c':.1}))
    monkeypatch.setattr(script,'plot_results',lambda *a:[])
    monkeypatch.setattr(script,'compare_history',lambda *a:None)
    model.base_model=nn.Identity()
    class Inputs:
        """提供三批可重放输入。"""
        def __init__(self,*a):
            """保存固定任务身份和题数。"""
            self.identity={'v':1};self.config=SimpleNamespace(limit=None,sample_start_index=0)
            self.tasks={'tiny':SimpleNamespace(eval_docs=[0,1,2])}
        def batches(self):
            """逐批输出摘要和计数。"""
            for i in range(3):
                yield dict(input_ids=torch.tensor([[i]]),attention_mask=torch.ones(1,1,dtype=torch.long),
                           examples=1,logical_requests=1,sequences=1,digest=str(i))
    monkeypatch.setattr(script,'TaskInputs',Inputs)
    calls=[]
    def measure(*args):
        """第二批首次失败，之后成功。"""
        i=int(args[-1]['input_ids'][0,0]);calls.append(i)
        if calls==[0,1]:
            raise RuntimeError('interrupt')
        return {'c':[float(i+1),10.]}
    monkeypatch.setattr(script,'measure_batch',measure)
    output=tmp_path/'run'
    with pytest.raises(RuntimeError,match='interrupt'):
        script.main(['--config',str(path),'--output',str(output),'--device','cpu'])
    checkpoint=json.loads((output/'checkpoint.json').read_text())
    assert checkpoint['tasks']['tiny']['batches']==1
    script.main(['--resume',str(output)])
    assert calls==[0,1,1,2]
    checkpoint=json.loads((output/'checkpoint.json').read_text())
    assert checkpoint['tasks']['tiny']['sums']['c']==[6.,30.]
    assert checkpoint['tasks']['tiny']['complete']


def test_historical_matching_requires_exact_plan(tmp_path, monkeypatch):
    """历史指标只关联相同完整配置，不借给其他 rank。"""
    import json
    from matplotlib.figure import Figure
    from scripts.nmse_reporting import compare_history
    from qcomp.workflows.compression_plan_io import compression_plan_to_dict
    plans=[compression_plan_to_dict(CompressionPlan((CompressionTarget('model.layers.0.self_attn.q_proj','mpo',MPOSpec((2,2),(2,2),(1,r,1))),))) for r in (1,2)]
    cases=[{'compression_plan':plan,'historical_metrics':None,'module_path':'model.layers.0.self_attn.q_proj',
            'block':0,'projection':'q_proj','sources':{'mmlu':{'output_nmse':.2}},
            'dense_parameters':16,'compressed_parameters':8} for plan in plans]
    document={'experiment':{'historical_identity_note':'unknown historical tokens'},'cases':cases}
    (tmp_path/'local_output_nmse.json').write_text(json.dumps(document))
    (tmp_path/'report.md').write_text('# report\n')
    history={'evaluation_config':{'metric_directions':{'acc':'higher'}},'execution':{},
             'baseline':{'metrics':{'acc':.8}},'cases':[{'compression_plan':plans[0],'metrics':{'acc':.7}}]}
    monkeypatch.setattr(Figure,'savefig',lambda *a,**k:None)
    compare_history(tmp_path,{'mmlu':history},['q_proj'])
    result=json.loads((tmp_path/'local_output_nmse.json').read_text())
    assert result['cases'][0]['historical_metrics']['mmlu']['acc']['drop_pp']==pytest.approx(10)
    assert result['cases'][1]['historical_metrics'] is None


def test_report_aggregates_source_sums(tmp_path):
    """多个来源按总分子和总分母聚合，禁止直接平均 NMSE。"""
    import json
    from scripts.nmse_reporting import write_results
    record={'name':'c','module_path':'layer','block':0,'projection':'q_proj',
            'dense_parameters':16,'compressed_parameters':8,'rank':1}
    tasks={}
    for name,error,power in [('small',1.,1.),('large',1.,9.)]:
        tasks[name]={'sums':{'c':[error,power]},'valid_tokens':2,'examples':1,
                     'logical_requests':1,'sequences':1,'complete':True,
                     'preparation_seconds':0.,'evaluation_seconds':0.,'write_seconds':0.}
    state={'tasks':tasks,'identity':{'config':{'tasks':[{},{}]}}}
    experiment={'candidate_tensor_bytes':16,'decomposition_seconds':0.,'historical_identity_note':'test'}
    write_results(tmp_path,experiment,{'c':record},state,['layer'])
    result=json.loads((tmp_path/'local_output_nmse.json').read_text())
    assert result['cases'][0]['output_nmse']==pytest.approx(.2)
    assert result['cases'][0]['valid_tokens']==4
