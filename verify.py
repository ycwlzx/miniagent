"""验证脚本: 5 个场景, 前 4 个离线跑 (不调用模型, 免费且可重复), 第 5 个真实端到端。

**为什么把验证做成一等公民**:

上下文治理这类机制有个共同特点: 它的触发条件是"快超限了",
而日常短对话根本不会触发。结果是**写完不验证, 永远不知道它有没有用** ——
或者更糟, 以为自己写对了, 直到某次长任务在生产环境里崩掉。

所以这里的每个场景都对应一种真实失败:

    场景 1  预算与水位      —— "窗口还剩多少? 我算的对不对?"
    场景 2  截断与落盘      —— "一次大输出会不会把预算吃掉一半?"
    场景 3  结构修复        —— "截断之后接口会不会整批拒绝?"
    场景 4  历史裁剪        —— "真装不下了, 降级路径能不能跑通?"
    场景 5  端到端          —— "真实任务下的数字是多少?"

跑法:
    python verify.py            # 只跑离线场景
    python verify.py --e2e      # 加上真实端到端 (需要 LLM_API_KEY)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from miniagent.cli import build_agent, load_env  # noqa: E402
from miniagent.tools import build_default_registry  # noqa: E402
from miniagent.context import (  # noqa: E402
    TokenBudget,
    TruncateConfig,
    apply_to_messages,
    collect_tool_names,
    estimate_request,
    estimate_text,
    repair_messages,
    truncate_tool_result,
)
from miniagent.llm import LLMResponse  # noqa: E402

# --------------------------------------------------------------- 测试工具 --

PASS, FAIL = "PASS", "FAIL"
_results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((name, PASS if ok else FAIL, detail))
    mark = "\033[32m✓\033[0m" if ok else "\033[31m✗\033[0m"
    print(f"    {mark} {name}" + (f"   {detail}" if detail else ""))
    return ok


class FakeLLM:
    """脚本化的假模型: 用于离线验证治理逻辑。

    为什么要它: 治理逻辑的正确性和模型说什么话无关。
    用假模型可以把 loop 跑完整, 又不花 token、不依赖网络、结果完全可复现 ——
    这是"能不能测"和"只能靠人工观察"的分界线。
    """

    def __init__(self, script: list[LLMResponse], prompt_tokens: int = 0) -> None:
        self.script = script
        self.i = 0
        self.prompt_tokens = prompt_tokens
        self.model = "fake-model"

    def chat(self, messages, tools=None) -> LLMResponse:  # noqa: ANN001
        if self.i < len(self.script):
            resp = self.script[self.i]
            self.i += 1
        else:
            resp = LLMResponse(content="(脚本已用完)", finish_reason="stop")
        # 模拟服务商返回的真实 token (用于校准逻辑)
        resp.prompt_tokens = self.prompt_tokens or estimate_request(messages, tools)
        return resp

    def stats(self) -> dict[str, int]:
        return {"calls": self.i, "prompt_tokens": 0, "completion_tokens": 0}


# ------------------------------------------------------------------ 场景 --


def case_1_budget() -> dict[str, Any]:
    print("\n场景 1 · 预算与水位")
    budget = TokenBudget(window=8_000, max_output=1_000, safety_buffer=512)

    check("预算公式正确", budget.budget == 8_000 - 1_000 - 512,
          f"预算 {budget.budget}")

    small = [{"role": "user", "content": "你好"}]
    p_small = budget.pressure(small)
    check("小请求水位低", 0 < p_small < 0.1, f"{p_small:.2%}")

    big = [{"role": "user", "content": "中" * 20_000}]
    p_big = budget.pressure(big)
    check("大请求水位超 1.0", p_big > 1.0, f"{p_big:.2%}")

    # 校准: 假造一次"我们估 1000, 实际 800"的样本
    budget.pressure([{"role": "user", "content": "x" * 3_500}])
    stats = budget.calibrate(int(budget.last_estimated * 0.8))
    check("校准能算出偏差", stats["samples"] == 1 and stats["mean_bias"] > 0,
          f"偏差 {stats['mean_bias']:+.1%}")

    # 中文/英文估算差异 (用来解释"为什么缓冲不能太小")
    cjk = estimate_text("这是一段中文文本用来测试估算")
    eng = estimate_text("this is an english sentence for testing")
    check("中文按字估算更贵", cjk / 15 > eng / 38, f"中文 {cjk} / 英文 {eng}")

    return {"budget": budget.budget, "p_small": p_small, "p_big": p_big}


def case_2_truncate() -> dict[str, Any]:
    print("\n场景 2 · 截断与落盘")
    tmp = Path(tempfile.mkdtemp(prefix="miniagent_verify_"))

    long_text = "\n".join(f"第 {i} 行内容 " + "x" * 40 for i in range(1, 601))
    original_chars = len(long_text)

    # --- shell 结果: 应保留尾部 ---
    cfg = TruncateConfig(max_chars=2_000, workspace=tmp)
    r_shell = truncate_tool_result("run_shell", long_text, cfg)
    check("shell 结果被截断", r_shell.truncated and len(r_shell.content) < original_chars,
          f"{original_chars:,} → {len(r_shell.content):,} 字符")
    check("shell 结果保留尾部", "第 600 行" in r_shell.content,
          "尾部内容存在")
    check("截断提示含原始大小", f"{original_chars:,}" in r_shell.content)
    check("已落盘且文件存在",
          bool(r_shell.offload_path) and (tmp / r_shell.offload_path).exists(),
          r_shell.offload_path or "(无)")

    # --- read_file 结果: 应保留头部, 且豁免落盘 (防循环放大) ---
    r_read = truncate_tool_result("read_file", long_text, cfg)
    check("read_file 保留头部", "第 1 行" in r_read.content)
    check("read_file 豁免落盘", r_read.offload_path is None,
          "避免 落盘→读→再落盘 循环")

    # --- 短结果不该被动 ---
    r_short = truncate_tool_result("run_shell", "ok", cfg)
    check("短结果不截断", not r_short.truncated)

    # --- 历史遍历 ---
    history = [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "run_shell", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": long_text},
    ]
    names = collect_tool_names(history)
    check("能从历史里还原工具名", names.get("c1") == "run_shell")
    out, results = apply_to_messages(history, names, cfg)
    check("遍历历史时完成截断", len(results) == 1 and len(out[1]["content"]) < original_chars)

    shutil.rmtree(tmp, ignore_errors=True)
    return {
        "original_chars": original_chars,
        "after_shell": len(r_shell.content),
        "after_read": len(r_read.content),
    }


def case_3_repair() -> dict[str, Any]:
    print("\n场景 3 · 消息结构自修复")

    def call(cid: str, name: str = "read_file") -> dict[str, Any]:
        return {"id": cid, "type": "function",
                "function": {"name": name, "arguments": "{}"}}

    # --- 孤儿结果: 调用被裁掉了, 只留下结果 ---
    orphan = [
        {"role": "user", "content": "任务"},
        {"role": "tool", "tool_call_id": "gone", "content": "孤立结果"},
    ]
    r = repair_messages(orphan)
    check("丢弃孤儿结果", r.dropped_orphans == 1 and len(r.messages) == 1,
          r.summary())

    # --- 缺失结果: 有调用没结果 (中断场景) ---
    missing = [
        {"role": "user", "content": "任务"},
        {"role": "assistant", "content": "", "tool_calls": [call("c1")]},
    ]
    r = repair_messages(missing)
    check("回填缺失结果", r.backfilled == 1 and len(r.messages) == 3,
          r.summary())
    check("回填内容诚实(不伪造数据)",
          "不可用" in r.messages[2]["content"],
          r.messages[2]["content"][:40])
    check("回填位置紧跟调用之后", r.messages[2]["role"] == "tool"
          and r.messages[2]["tool_call_id"] == "c1")

    # --- 畸形调用: name 缺失 ---
    malformed = [
        {"role": "user", "content": "任务"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "bad", "type": "function", "function": {"name": "", "arguments": "{}"}},
            call("good"),
        ]},
        {"role": "tool", "tool_call_id": "bad", "content": "这个结果是孤儿了"},
        {"role": "tool", "tool_call_id": "good", "content": "正常结果"},
    ]
    r = repair_messages(malformed)
    check("剥离畸形调用", r.stripped_calls == 1, r.summary())
    check("畸形调用的结果被正确判为孤儿", r.dropped_orphans == 1)
    check("合法配对保持不变",
          any(m.get("tool_call_id") == "good" for m in r.messages))

    # --- 正常结构不应被改动 ---
    healthy = [
        {"role": "user", "content": "任务"},
        {"role": "assistant", "content": "", "tool_calls": [call("c1")]},
        {"role": "tool", "tool_call_id": "c1", "content": "结果"},
        {"role": "assistant", "content": "完成"},
    ]
    r = repair_messages(healthy)
    check("完好结构零改动", not r.changed and len(r.messages) == 4, r.summary())

    return {"dropped": r.dropped_orphans}


def case_4_snip() -> dict[str, Any]:
    print("\n场景 4 · 历史裁剪 (降级路径)")

    # 窗口设很小, 逼出裁剪
    llm = FakeLLM([LLMResponse(content="完成", finish_reason="stop")])
    tmp = Path(tempfile.mkdtemp(prefix="miniagent_snip_"))
    agent = build_agent(tmp, window=2_000, max_output=200, safety_buffer=100, llm=llm)

    # 塞一大段历史, 使水位超过 100%
    agent_run_messages = [
        {"role": "system", "content": "系统提示"},
        {"role": "user", "content": "任务"},
    ]
    for i in range(30):
        agent_run_messages.append({"role": "user", "content": f"第 {i} 轮 " + "长" * 200})
        agent_run_messages.append({"role": "assistant", "content": "回应 " + "答" * 200})

    tools = agent.registry.schemas()
    before = agent.budget.pressure(agent_run_messages, tools)
    check("构造出的历史已超预算", before > 1.0, f"水位 {before:.2%}")

    sniped, dropped = agent._snip_history(agent_run_messages, tools)
    after = agent.budget.pressure(sniped, tools)
    check("裁剪确实降低了水位", after < before, f"{before:.2%} → {after:.2%}")
    check("裁剪保留了 system 消息", sniped[0]["role"] == "system")
    check("裁剪丢弃了消息", dropped > 0, f"丢弃 {dropped} 条")
    check("保留最近消息", len(sniped) >= 1 + agent.options.keep_recent)

    # --- 关键: 裁剪救不回来的情况必须**明确失败**, 而不是静默继续 ---
    # 这个场景很真实: 单条用户消息本身就超预算 (比如粘进来一个超长文件),
    # 这时候裁剪历史没有任何用 —— 因为超的不是历史。
    # 正确行为是识别出来并报错, 而不是把消息悄悄丢掉一部分发出去。
    llm2 = FakeLLM([])
    agent2 = build_agent(tmp, window=300, max_output=100, safety_buffer=50, llm=llm2)
    res = agent2.run("请分析这段内容: " + "巨" * 3_000)
    check("裁剪无效时明确失败", not res.success and "预算不足" in res.answer,
          res.answer[:46] + "…")
    check("失败发生在请求之前 (未浪费 token)", llm2.i == 0,
          f"模型调用次数 {llm2.i}")

    shutil.rmtree(tmp, ignore_errors=True)
    return {"pressure_before": before, "pressure_after": after, "dropped": dropped}


def case_5_e2e(workspace: Path) -> dict[str, Any] | None:
    print("\n场景 5 · 真实端到端 (调用模型)")

    load_env(ROOT)
    import os

    if not os.getenv("LLM_API_KEY"):
        print("    ⚠ 未设置 LLM_API_KEY, 跳过 (设好环境变量或 .env 后重跑)")
        return None

    from miniagent.llm import LLMClient

    client = LLMClient()
    agent = build_agent(workspace, window=128_000, max_output=4_096,
                        safety_buffer=1_024, on_event=None, llm=client)

    task = (
        "请先读 miniagent/context/budget.py 的前 60 行, "
        "然后用一句话说明它的核心公式是什么。不要修改任何文件。"
    )
    print(f"    任务: {task}")
    result = agent.run(task)

    check("任务成功结束", result.success, f"{result.turns} 轮")
    check("产生了工具调用", result.stats["tool_calls"] > 0,
          f"{result.stats['tool_calls']} 次")

    err = agent.budget.error_stats()
    check("拿到了真实 token 用于校准", err.get("samples", 0) > 0,
          f"{int(err.get('samples', 0))} 样本, 偏差 {err.get('mean_bias', 0):+.1%}")

    print(f"\n    答案: {result.answer.strip()[:160]}")
    return {
        "turns": result.stats["turns"],
        "tool_calls": result.stats["tool_calls"],
        "max_pressure": result.stats["max_pressure"],
        "truncate_events": result.stats["truncate_events"],
        "snip_events": result.stats["snip_events"],
        "repair_events": result.stats["repair_events"],
        "estimate_bias": err.get("mean_bias", 0.0),
        "worst_under": err.get("max_under", 0.0),
        "prompt_tokens": client.total_prompt_tokens,
    }


def case_6_governance_effect() -> dict[str, Any]:
    """治理效果对照: 同一个长任务, 治理前后差多少。

    这是整个验证里**最能说明问题**的场景 —— 它把"治理有没有用"从一句
    "我实现了上下文治理"变成了两组可以直接比较的数字。

    模拟的场景很常见: 一个需要反复读文件的任务, 每轮工具结果 30K 字符左右
    (相当于读一个中等大小的源码文件)。跑 15 轮之后:

      - 不治理: 15 × 30K 原样堆积, 必然超预算, 任务在真实环境里会直接失败
      - 治理后: 截断 + 结构修复之后回到安全水位
    """
    print("\n场景 6 · 治理效果对照 (长任务模拟)")

    tmp = Path(tempfile.mkdtemp(prefix="miniagent_gov_"))
    workspace_registry = build_default_registry(tmp)
    tools = workspace_registry.schemas()
    budget = TokenBudget(window=128_000, max_output=4_096, safety_buffer=1_024)

    big_result = "\n".join(f"第 {i:>4} 行 " + "x" * 60 for i in range(500))
    print(f"    单条工具结果 {len(big_result):,} 字符 × 15 轮")

    raw: list[dict[str, Any]] = [
        {"role": "system", "content": "系统提示"},
        {"role": "user", "content": "完成一个需要多轮探索的任务"},
    ]
    for i in range(15):
        cid = f"c{i}"
        raw.append({
            "role": "assistant", "content": "",
            "tool_calls": [{"id": cid, "type": "function",
                            "function": {"name": "read_file", "arguments": "{}"}}],
        })
        raw.append({"role": "tool", "tool_call_id": cid, "content": big_result})

    p_raw = budget.pressure(raw, tools)

    # --- 实验组: 过一遍治理 (截断 + 结构修复) ---
    cfg = TruncateConfig(max_chars=2_000, workspace=tmp)
    names = collect_tool_names(raw)
    governed, trunc_results = apply_to_messages(raw, names, cfg)
    governed = repair_messages(governed).messages
    p_gov = budget.pressure(governed, tools)
    second_pass = repair_messages(governed)

    check("对照组(不治理)会超预算", p_raw > 1.0, f"水位 {p_raw:.1%}")
    check("实验组(过治理)回到预算内", p_gov <= 1.0, f"水位 {p_gov:.1%}")
    check("截断在历史遍历中全部生效", len(trunc_results) == 15,
          f"{len(trunc_results)} 条")
    check("治理后结构合法 (无需二次修复)", not second_pass.changed,
          second_pass.summary())

    reduction = 1 - p_gov / p_raw
    check("水位下降幅度超过 80%", reduction > 0.80, f"下降 {reduction:.1%}")

    raw_chars = sum(len(m.get("content") or "") for m in raw)
    gov_chars = sum(len(m.get("content") or "") for m in governed)

    shutil.rmtree(tmp, ignore_errors=True)
    return {
        "raw_pressure": p_raw,
        "governed_pressure": p_gov,
        "reduction": reduction,
        "raw_chars": raw_chars,
        "governed_chars": gov_chars,
        "truncated": len(trunc_results),
    }


# ------------------------------------------------------------------ 主流程 --


def main() -> int:
    parser = argparse.ArgumentParser(description="miniagent 验证脚本")
    parser.add_argument("--e2e", action="store_true", help="包含真实端到端场景")
    args = parser.parse_args()

    print("=" * 66)
    print("miniagent 验证")
    print("=" * 66)

    m1 = case_1_budget()
    m2 = case_2_truncate()
    m3 = case_3_repair()
    m4 = case_4_snip()
    m5 = case_5_e2e(ROOT) if args.e2e else None
    m6 = case_6_governance_effect()

    # ---- 汇总 ----
    print("\n" + "=" * 66)
    print("结果汇总")
    print("=" * 66)
    passed = sum(1 for _, s, _ in _results if s == PASS)
    failed = [(n, d) for n, s, d in _results if s == FAIL]
    print(f"  {passed} 通过 / {len(failed)} 失败  (共 {len(_results)} 项)")
    for n, d in failed:
        print(f"    ✗ {n}  {d}")

    print("\n  简历可用的数字:")
    print(f"    【治理效果】长任务模拟: 水位 {m6['raw_pressure']:.1%} → "
          f"{m6['governed_pressure']:.1%} (下降 {m6['reduction']:.1%})")
    print(f"                消息体积 {m6['raw_chars']:,} → {m6['governed_chars']:,} 字符")
    print(f"    【截断策略】{m2['original_chars']:,} → {m2['after_shell']:,} 字符 "
          f"(shell 留尾) / {m2['after_read']:,} 字符 (文件留头)")
    print(f"    【历史裁剪】水位 {m4['pressure_before']:.1%} → {m4['pressure_after']:.1%}, "
          f"丢弃 {m4['dropped']} 条消息")
    if m5:
        print(f"    【端到端】{m5['turns']} 轮 / {m5['tool_calls']} 次工具调用 / "
              f"峰值水位 {m5['max_pressure']:.1%}")
        print(f"    【估算精度】偏差 {m5['estimate_bias']:+.1%} "
              f"(最差低估 {m5['worst_under']:+.1%}, 用于解释安全缓冲取值)")
        print(f"    【上下文消耗】{m5['prompt_tokens']:,} tokens")

    print("\n  可导出的原始数据:")
    print(json.dumps({"case1": m1, "case2": m2, "case3": m3, "case4": m4,
                      "case5": m5, "case6": m6},
                     ensure_ascii=False, indent=2)[:400] + " ...")

    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
