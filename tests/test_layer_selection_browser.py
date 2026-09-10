"""在真实离线浏览器中验证选层计算与页面交互。

主要内容：
- ``dashboard``、``page``：生成独立测试页面并启动隔离浏览器。
- 数学用例：权重网格、并列分摊、分位数和掉点过滤。
- 交互用例：热力图联动、稳定性失效、取消、导入导出与 A/B 对比。
"""

import json
from pathlib import Path

import pytest

from scripts.build_layer_selection_dashboard import build_dashboard
from test_layer_selection_dashboard import make_sources

playwright = pytest.importorskip(
    "playwright.sync_api", reason="浏览器测试需要 playwright 和 Chromium"
)


@pytest.fixture(scope="module")
def dashboard(tmp_path_factory):
    """生成包含多指标 GSM8K 的确定性页面，避免依赖真实实验文件。"""
    root = tmp_path_factory.mktemp("dashboard")
    config = make_sources(root)
    log = root / "gsm8k/events.jsonl"
    events = [json.loads(s) for s in log.read_text().splitlines()]
    metric = "exact_match_flexible_extract"
    events[0]["fields"]["metrics"].append(metric)
    events[0]["fields"]["metric_directions"][metric] = "higher"
    for event in events[1:-1]:
        event["fields"]["metrics"][metric] = 0.625
        event["fields"]["degradations"][metric] = 0.125
    log.write_text("\n".join(json.dumps(e) for e in events))
    report = root / "gsm8k/report.md"
    report.write_text(report.read_text() + f"| {metric} | 0.75 |\n")
    source = root / "sources.json"
    source.write_text(json.dumps(config))
    return build_dashboard(source, root / "output")


@pytest.fixture()
def page(dashboard):
    """打开 file 页面并禁止联网；测试结束确认没有脚本异常和网络请求。"""
    with playwright.sync_playwright() as service:
        browser = service.chromium.launch(headless=True, args=["--no-sandbox"])
        context = browser.new_context(
            offline=True, viewport={"width": 1440, "height": 1000}
        )
        tab = context.new_page()
        errors, requests = [], []
        tab.on("pageerror", lambda error: errors.append(str(error)))
        tab.on(
            "request",
            lambda request: (
                requests.append(request.url) if request.url.startswith("http") else None
            ),
        )
        tab.goto(dashboard.as_uri())
        yield tab
        browser.close()
        assert not errors
        assert not requests


def test_math_grid_ties_and_quantiles(page):
    """默认网格 381 组，边界并列不偏向名称，排名分位数使用平均名次。"""
    result = page.evaluate(
        """async () => {
      const C = LayerSelection;
      const p = JSON.parse(document.getElementById('dashboard-data').textContent);
      const s = C.defaults(p), grid = C.weightGrid(s.datasets);
      const tied = C.tiedRanks([{index:0,score:1},{index:1,score:1},{index:2,score:2}],1);
      const rows = [{index:0,eligible:true,losses:[0,2]}, {index:1,eligible:true,losses:[2,0]}];
      const stable = await C.stability(rows, [[1,0],[0,1],[.5,.5]], 1);
      return {count:grid.length, sums:grid.every(g=>Math.abs(g.reduce((a,b)=>a+b,0)-1)<1e-12),tied,stable};
    }"""
    )
    assert result["count"] == 381 and result["sums"]
    assert [x["share"] for x in result["tied"]] == [0.5, 0.5, 0]
    assert [x["rank"] for x in result["tied"]] == [1.5, 1.5, 3]
    for stat in result["stable"]["stats"].values():
        assert stat["frequency"] == 0.5
        assert stat["medianRank"] == 1.5 and stat["p95Rank"] == 2


def test_math_negative_drops_filter_and_order(page):
    """负掉点不抵消损失，零权重任务仍参与最大掉点筛选。"""
    result = page.evaluate(
        """() => {
      const p={modules:[{name:'a',saving:.1},{name:'b',saving:.2}], datasets:[
        {id:'x',defaultMetric:'acc',baseline:{acc:.5},values:{acc:[.6,.4]}},
        {id:'y',defaultMetric:'acc',baseline:{acc:.8},values:{acc:[.6,.8]}}]};
      const s=LayerSelection.defaults(p);s.k=1;
      const rows=LayerSelection.evaluate(p,s);
      const ordered=LayerSelection.fixedOrder(rows).map(r=>r.name);
      s.capEnabled=true;s.cap=15;
      const filtered=LayerSelection.evaluate(p,s).map(r=>r.eligible);
      s.datasets[0].enabled=false;LayerSelection.resetRanges(s);
      const single=LayerSelection.weightGrid(s.datasets);
      return {scores:rows.map(r=>r.score),drops:rows[0].drops,ordered,filtered,single};
    }"""
    )
    assert result["scores"] == pytest.approx([10, 5])
    assert result["drops"] == pytest.approx([-10, 20])
    assert result["ordered"] == ["b", "a"]
    assert result["filtered"] == [False, True]
    assert result["single"] == [[1]]


def test_controls_and_module_linkage(page):
    """勾选、指标、权重、K、掉点上限和热力图联动即时更新。"""
    assert page.locator("#selected-count").inner_text() == "32"
    assert page.locator("rect.cell").count() == 252
    assert page.locator("#ranking-body tr").count() == 252
    gsm = page.locator('[data-index="3"]')
    gsm.locator("select").select_option("exact_match_flexible_extract")
    assert (
        page.locator("#ranking-body tr").first.locator("td").nth(3).inner_text()
        == "22.5000"
    )
    gsm.locator('[data-field="enabled"]').uncheck()
    assert page.locator(".normalized-weight").first.inner_text() == "实际 25.00%"
    assert page.locator('[data-index="0"] [data-field="lower"]').input_value() == "10"
    page.locator("#k").fill("1")
    assert page.locator("#selected-count").inner_text() == "1"
    page.locator("#k").fill("252")
    assert page.locator("#selected-count").inner_text() == "252"
    page.locator("#heatmap [data-module='0']").click()
    assert "model.layers.0.self_attn.q_proj" in page.locator("#detail").inner_text()
    assert "focused" in page.locator("#row-0").get_attribute("class")
    page.locator("#row-1").click()
    assert "focused" in page.locator("#heatmap [data-module='1']").get_attribute(
        "class"
    )
    page.locator("#cap-enabled").check()
    page.locator("#cap").fill("0")
    assert page.locator("#selected-count").inner_text() == "0"
    assert page.locator("#export-json").is_disabled()
    page.locator("#cap-enabled").uncheck()
    page.evaluate(
        """() => document.querySelectorAll('[data-field=weight]').forEach(e=>{e.value=0;e.dispatchEvent(new Event('input',{bubbles:true}));})"""
    )
    assert "权重总和为零" in page.locator("#status").inner_text()
    page.locator("#equal").click()
    for index in range(5):
        checkbox = page.locator(f'[data-index="{index}"] [data-field="enabled"]')
        if checkbox.is_checked():
            checkbox.uncheck()
    assert "至少勾选" in page.locator("#status").inner_text()


def test_stability_cancel_invalidate_and_export(page):
    """计算结果可以取消，设置改变后不能导出旧候选。"""
    page.locator("#stable-mode").click()
    assert page.locator("#export-json").is_disabled()
    page.evaluate(
        "() => {document.getElementById('calculate').click();document.getElementById('cancel').click();}"
    )
    assert "取消" in page.locator("#status").inner_text()
    assert page.locator("#export-json").is_disabled()
    page.locator("#calculate").click()
    page.wait_for_function(
        "document.getElementById('status').textContent.includes('已完成 381')"
    )
    assert page.locator("#selected-count").inner_text() == "32"
    assert not page.locator("#export-json").is_disabled()
    with page.expect_download() as download:
        page.locator("#export-json").click()
    plan = json.loads(Path(download.value.path()).read_text())
    assert plan["stability"]["count"] == 381
    assert sum(
        s["frequency"] for s in plan["stability"]["stats"].values()
    ) == pytest.approx(32)
    page.locator("#k").fill("16")
    assert "待更新" in page.locator("#status").inner_text()
    assert page.locator("#export-json").is_disabled()
    page.locator("#import-file").set_input_files(
        {
            "name": "plan.json",
            "mimeType": "application/json",
            "buffer": json.dumps(plan).encode(),
        }
    )
    page.wait_for_function(
        "document.getElementById('status').textContent.includes('方案已导入')"
    )
    assert page.locator("#k").input_value() == "32"
    # 五个任务的下限都设为 100%，使权重网格不可行。
    page.evaluate(
        """() => document.querySelectorAll('[data-field=upper], [data-field=lower]').forEach(e=>{e.value=100;e.dispatchEvent(new Event('input',{bubbles:true}));})"""
    )
    page.locator("#calculate").click()
    assert "没有总和" in page.locator("#status").inner_text()


def test_snapshots_json_csv_and_hash_rejection(page):
    """A/B 保存设置快照，文件往返核对来源且不信任导入的候选名单。"""
    page.locator("#save-a").click()
    page.locator('[data-k="16"]').click()
    page.locator("#save-b").click()
    assert "共同模块 16" in page.locator("#comparison").inner_text()
    with page.expect_download() as download:
        page.locator("#export-json").click()
    plan = json.loads(Path(download.value.path()).read_text())
    assert len(plan["selectedModules"]) == 16 and len(plan["sources"]) == 5
    with page.expect_download() as csv:
        page.locator("#export-csv").click()
    lines = Path(csv.value.path()).read_text(encoding="utf-8-sig").splitlines()
    assert len(lines) == 253 and "baseline" in lines[0]
    page.locator('[data-k="64"]').click()
    page.locator("#import-file").set_input_files(
        {
            "name": "plan.json",
            "mimeType": "application/json",
            "buffer": json.dumps(plan).encode(),
        }
    )
    page.wait_for_function(
        "document.getElementById('status').textContent.includes('方案已导入')"
    )
    assert page.locator("#selected-count").inner_text() == "16"
    original_hash = plan["sources"][0]["sha256"]
    plan["sources"][0]["sha256"] = "invalid"
    page.locator("#import-file").set_input_files(
        {
            "name": "bad.json",
            "mimeType": "application/json",
            "buffer": json.dumps(plan).encode(),
        }
    )
    page.wait_for_function(
        "document.getElementById('status').textContent.includes('哈希不匹配')"
    )
    assert page.locator("#selected-count").inner_text() == "16"

    plan["sources"][0]["sha256"] = original_hash
    plan["selectedModules"].reverse()
    page.locator("#import-file").set_input_files(
        {
            "name": "tampered.json",
            "mimeType": "application/json",
            "buffer": json.dumps(plan).encode(),
        }
    )
    page.wait_for_function(
        "document.getElementById('status').textContent.includes('名单与设置')"
    )
    assert page.locator("#selected-count").inner_text() == "16"
