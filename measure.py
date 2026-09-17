# -*- coding: utf-8 -*-
"""真实任务测量：跑一批真任务，采集治理机制的真实数据。

与 verify.py 的 case_6 的区别（这正是它被质疑的原因）：
  case_6 是【构造】场景 —— 15 轮固定大小的假工具结果，测出来的
  "下降 94.4%" 只是截断阈值的镜像。
  本脚本跑的是【真实】任务：真模型、真工具调用、真多轮。

两个窗口分别测：
  A 组 window=128K —— 常规配置，看真实占用与机制触发情况
  B 组 window=16K  —— 受限配置（模拟小模型/边缘部署），逼出治理机制

采集指标（全部来自真实运行）：
  - 任务完成率
  - 真实水位峰值
  - 各类治理动作触发次数
  - 估算偏差（样本量）

用法：
    python measure.py            # 跑 A 组
    python measure.py --both     # 跑 A + B
    python measure.py --window 16000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from miniagent.cli import build_agent, load_env  # noqa: E402
from miniagent.llm import LLMClient  # noqa: E402

# ---- 真实任务集：都可完成、都只读（不改文件）、都需要多轮工具调用 ----
TASKS = [
    ("读 miniagent/context/budget.py，用一句话说出预算公式是什么", 1),
    ("读 miniagent/context/repair.py，说出三步修复的顺序和作用", 1),
    ("用 grep 找出项目里所有 `def build_` 开头的函数定义在哪个文件", 1),
    ("统计项目里有多少个 .py 文件，并说出最大的那个文件名", 1),
    ("读 README.md，列出四个核心机制的标题", 1),
    ("读 miniagent/tools/base.py，说明为什么工具执行失败不抛异常", 1),
    ("读 miniagent/loop.py 和 miniagent/context/__init__.py，说明 context 层为什么不允许调用 LLM", 2),
    ("读 miniagent/safety.py，说出它的模式匹配有什么局限", 1),
]


def run_one(task: str, window: int, max_chars: int, quiet: bool = True):
    """跑一个真实任务，返回统计。"""
    client = LLMClient()
    agent = build_agent(
        ROOT,
        window=window,
        max_output=2_048 if window < 32_000 else 4_096,
        safety_buffer=1_024,
        max_chars=max_chars,
        on_event=None,
        llm=client,
    )
    t0 = time.time()
    try:
        result = agent.run(task)
        ok = result.success
        answer = (result.answer or "").strip().replace("\n", " ")[:80]
        err = ""
    except Exception as exc:  # noqa: BLE001
        ok, answer, err = False, "", f"{type(exc).__name__}: {exc}"[:120]

    stats = agent.stats
    bias = agent.budget.error_stats()
    return {
        "success": ok,
        "seconds": round(time.time() - t0, 1),
        "turns": stats["turns"],
        "tool_calls": stats["tool_calls"],
        "max_pressure": round(stats["max_pressure"], 4),
        "truncate_events": stats["truncate_events"],
        "snip_events": stats["snip_events"],
        "snipped_messages": stats["snipped_messages"],
        "repair_events": stats["repair_events"],
        "llm_calls": client.call_count,
        "prompt_tokens": client.total_prompt_tokens,
        "bias_samples": int(bias.get("samples", 0)),
        "bias_mean": round(bias.get("mean_bias", 0.0), 4),
        "bias_worst": round(bias.get("max_under", 0.0), 4),
        "answer": answer,
        "error": err,
    }


def run_group(label: str, window: int, max_chars: int):
    print(f"\n{'='*70}")
    print(f"{label}   window={window:,}  max_chars={max_chars:,}")
    print("=" * 70)
    rows = []
    for i, (task, _) in enumerate(TASKS, 1):
        print(f"\n[{i}/{len(TASKS)}] {task[:56]}…")
        r = run_one(task, window, max_chars)
        rows.append(r)
        mark = "OK  " if r["success"] else "FAIL"
        print(f"   {mark} {r['seconds']}s | 轮数 {r['turns']} | 工具 {r['tool_calls']} | "
              f"峰值水位 {r['max_pressure']:.1%}")
        print(f"        截断 {r['truncate_events']} | 裁剪 {r['snip_events']} | "
              f"结构修复 {r['repair_events']} | LLM {r['llm_calls']} 次 | "
              f"{r['prompt_tokens']:,} tokens")
        if r["answer"]:
            print(f"        答: {r['answer'][:70]}")
        if r["error"]:
            print(f"        错: {r['error']}")

    # ---- 汇总 ----
    n = len(rows)
    ok_n = sum(1 for r in rows if r["success"])
    print(f"\n{'-'*70}")
    print(f"{label} 汇总")
    print(f"{'-'*70}")
    print(f"  任务完成率      : {ok_n}/{n} = {ok_n/n*100:.0f}%")
    print(f"  真实水位峰值     : 平均 {sum(r['max_pressure'] for r in rows)/n:.1%}"
          f"  最高 {max(r['max_pressure'] for r in rows):.1%}")
    print(f"  截断触发         : 共 {sum(r['truncate_events'] for r in rows)} 次"
          f"（{sum(1 for r in rows if r['truncate_events'])} 个任务触发）")
    print(f"  裁剪触发         : 共 {sum(r['snip_events'] for r in rows)} 次"
          f"（{sum(1 for r in rows if r['snip_events'])} 个任务触发）")
    print(f"  结构修复触发     : 共 {sum(r['repair_events'] for r in rows)} 次")
    print(f"  工具调用总数     : {sum(r['tool_calls'] for r in rows)}")
    print(f"  LLM 调用总数     : {sum(r['llm_calls'] for r in rows)}")
    print(f"  总 prompt tokens : {sum(r['prompt_tokens'] for r in rows):,}")
    print(f"  总耗时           : {sum(r['seconds'] for r in rows):.0f}s")

    samples = sum(r["bias_samples"] for r in rows)
    if samples:
        # 按 token 加权平均偏差
        wsum = sum(r["prompt_tokens"] for r in rows) or 1
        wbias = sum(r["bias_mean"] * r["prompt_tokens"] for r in rows) / wsum
        print(f"  估算偏差         : {wbias:+.2%}（{samples} 个样本）")
        print(f"  最差单次低估      : {min(r['bias_worst'] for r in rows):+.2%}")

    return {"label": label, "window": window, "rows": rows,
            "success_rate": ok_n / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=128_000)
    ap.add_argument("--max-chars", type=int, default=8_000)
    ap.add_argument("--both", action="store_true")
    ap.add_argument("--out", default="measure_result.json")
    args = ap.parse_args()

    load_env(ROOT)

    groups = []
    if args.both:
        groups.append(run_group("A 组 常规窗口", 128_000, 8_000))
        groups.append(run_group("B 组 受限窗口", 16_000, 2_000))
    else:
        label = "A 组 常规窗口" if args.window >= 100_000 else f"窗口 {args.window:,}"
        groups.append(run_group(label, args.window, args.max_chars))

    out = ROOT / args.out
    out.write_text(json.dumps(groups, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n原始数据已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
