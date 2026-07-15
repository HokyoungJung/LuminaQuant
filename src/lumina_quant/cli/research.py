"""Read-only research evidence commands."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lq research")
    commands = parser.add_subparsers(dest="command", required=True)
    proof = commands.add_parser("cost-proof", help="validate frozen cost-proof-v1 evidence")
    proof.add_argument("--input", required=True, help="frozen cost-proof JSON evidence")
    proof.add_argument("--config", required=True, help="frozen replacement profile")
    proof.add_argument(
        "--source-data-manifest", required=True, help="authenticated source-data manifest"
    )
    proof.add_argument(
        "--router-replay-manifest", required=True, help="authenticated R1/R2 replay manifest"
    )
    proof.add_argument(
        "--router-source-artifact", required=True, help="router replay source artifact"
    )
    proof.add_argument("--lifecycle", required=True, help="symbol lifecycle registry")
    proof.add_argument("--membership", required=True, help="fold membership manifest")
    proof.add_argument("--trial-ledger", required=True, help="complete whole-search trial ledger")
    proof.add_argument("--producer-source", required=True, help="cost-proof producer source")
    proof.add_argument("--commit-receipt", required=True, help="cost-proof commit receipt")
    proof.add_argument(
        "--router-producer-source", required=True, help="router replay producer source"
    )
    proof.add_argument(
        "--router-commit-receipt", required=True, help="router replay commit receipt"
    )
    args = parser.parse_args(argv)
    if args.command != "cost-proof":  # pragma: no cover - argparse owns this boundary
        return 2
    from lumina_quant.research.cost_proof import evaluate_cost_proof_file

    report = evaluate_cost_proof_file(
        args.input,
        args.config,
        source_data_manifest_path=args.source_data_manifest,
        router_replay_manifest_path=args.router_replay_manifest,
        router_source_artifact_path=args.router_source_artifact,
        lifecycle_path=args.lifecycle,
        membership_path=args.membership,
        trial_ledger_path=args.trial_ledger,
        producer_source_path=args.producer_source,
        commit_receipt_path=args.commit_receipt,
        router_producer_source_path=args.router_producer_source,
        router_commit_receipt_path=args.router_commit_receipt,
    )
    print(report.to_json())
    return {"PASS": 0, "REJECT": 1, "STOP": 2}[report.status]


if __name__ == "__main__":
    raise SystemExit(main())
