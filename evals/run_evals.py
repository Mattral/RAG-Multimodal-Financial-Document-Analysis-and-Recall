"""Evaluation runner — RAGAS + LLM-as-judge + CI regression gate.

Usage:
    python -m evals.run_evals --dataset evals/golden_datasets/financial_qa.jsonl
    python -m evals.run_evals --fail-on-regression

Exit codes:
    0  — all evals passed, no regression
    1  — quality regression detected (fails CI gate)
    2  — eval run error
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


async def _run(args):
    from src.rag_system.components.evaluator import (
        GoldenDatasetRunner,
        RagasEvaluator,
    )
    from src.rag_system.pipeline import create_pipeline

    pipeline = await create_pipeline()
    evaluator = RagasEvaluator()
    runner = GoldenDatasetRunner(
        pipeline=pipeline,
        evaluator=evaluator,
        golden_dataset_path=args.dataset,
        regression_threshold=args.regression_threshold,
        history_path="evals/history.json",
    )

    report = await runner.run(tenant_id="eval")

    # Console output
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"Run ID:              {report.run_id}")
    print(f"Timestamp:           {report.timestamp}")
    print(f"Samples:             {report.num_samples}")
    print(f"Pass Rate:           {report.pass_rate:.1%}")
    print(f"Avg Faithfulness:    {report.avg_faithfulness:.3f}")
    print(f"Avg Answer Relevancy:{report.avg_answer_relevancy:.3f}")
    print(f"Avg Numeric Accuracy:{report.avg_numeric_accuracy:.3f}")
    print(f"Avg Latency:         {report.avg_latency_ms:.0f}ms")
    print(f"Total Cost:          ${report.total_cost_usd:.4f}")
    regression_status = "YES ⚠" if report.regression_detected else "No ✅"
    print(f"Regression Detected: {regression_status}")
    print("=" * 60)

    # Per-sample failures
    failures = [r for r in report.results if not r.passed]
    if failures:
        print(f"\nFailed samples ({len(failures)}):")
        for r in failures[:5]:
            print(f"  Q: {r.question[:80]}")
            print(
                f"     faithfulness={r.faithfulness:.2f}, "
                f"numeric_accuracy={r.numeric_accuracy:.2f}"
            )

    # Save JSON report
    if args.output:
        Path(args.output).write_text(
            json.dumps(
                {
                    "run_id": report.run_id,
                    "pass_rate": report.pass_rate,
                    "avg_faithfulness": report.avg_faithfulness,
                    "avg_numeric_accuracy": report.avg_numeric_accuracy,
                    "regression_detected": report.regression_detected,
                    "num_samples": report.num_samples,
                },
                indent=2,
            )
        )
        print(f"\nReport written to {args.output}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Run RAG quality evaluations")
    parser.add_argument(
        "--dataset", default="evals/golden_datasets/financial_qa.jsonl"
    )
    parser.add_argument("--output", default=None, help="Path to write JSON report")
    parser.add_argument("--regression-threshold", type=float, default=0.05)
    parser.add_argument("--fail-on-regression", action="store_true")
    args = parser.parse_args()

    try:
        report = asyncio.run(_run(args))
    except Exception as exc:
        logger.error("eval_run_failed", error=str(exc))
        sys.exit(2)

    if args.fail_on_regression and report.regression_detected:
        print("\n❌ CI GATE FAILED: Quality regression detected")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
