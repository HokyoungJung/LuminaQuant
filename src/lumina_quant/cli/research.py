"""Read-only research evidence commands."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lq research")
    commands = parser.add_subparsers(dest="command", required=True)
    proof = commands.add_parser("cost-proof", help="validate frozen cost-proof-v2 evidence")
    proof.add_argument("--input", required=True, help="frozen cost-proof JSON evidence")
    proof.add_argument("--config", required=True, help="frozen replacement profile")
    proof.add_argument(
        "--source-data-manifest", required=True, help="authenticated source-data manifest"
    )
    proof.add_argument(
        "--source-run-receipt", required=True, help="authenticated source-run receipt"
    )
    proof.add_argument(
        "--search-run-receipt", required=True, help="authenticated search-run receipt"
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
    for option, help_text in (
        ("market-artifact", "market row artifact SHA256=PATH"),
        ("funding-artifact", "funding row artifact SHA256=PATH"),
        ("router-artifact", "Router cost artifact SHA256=PATH"),
        ("trial-result-artifact", "trial result artifact SHA256=PATH"),
    ):
        proof.add_argument(
            f"--{option}", action="append", default=[], metavar="SHA256=PATH", help=help_text
        )
    proof.add_argument("--source-data-commit-sha256", required=True, metavar="SHA256")
    proof.add_argument("--search-run-receipt-sha256", required=True, metavar="SHA256")
    proof.add_argument("--cost-proof-commit-sha256", required=True, metavar="SHA256")
    proof.add_argument("--router-source-artifact-sha256", required=True, metavar="SHA256")
    proof.add_argument("--router-commit-receipt-sha256", required=True, metavar="SHA256")
    args = parser.parse_args(argv)
    if args.command != "cost-proof":  # pragma: no cover - argparse owns this boundary
        return 2

    from lumina_quant.research.cost_proof import CostProofReport, evaluate_cost_proof_file

    def stop(reason: str) -> int:
        print(CostProofReport("STOP", "cost_proof_v2", (reason,), (), None).to_json())
        return 2

    def bindings(values: list[str]) -> dict[str, str] | None:
        parsed: dict[str, str] = {}
        for value in values:
            digest, separator, path = value.partition("=")
            if (
                not separator
                or not path
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
                or digest in parsed
            ):
                return None
            parsed[digest] = path
        return parsed

    market_artifacts = bindings(args.market_artifact)
    funding_artifacts = bindings(args.funding_artifact)
    router_artifacts = bindings(args.router_artifact)
    trial_result_artifacts = bindings(args.trial_result_artifact)
    roots = {
        "source_data_commit_sha256": args.source_data_commit_sha256,
        "search_run_receipt_sha256": args.search_run_receipt_sha256,
        "cost_proof_commit_sha256": args.cost_proof_commit_sha256,
        "router_source_artifact_sha256": args.router_source_artifact_sha256,
        "router_commit_receipt_sha256": args.router_commit_receipt_sha256,
    }
    if (
        market_artifacts is None
        or funding_artifacts is None
        or router_artifacts is None
        or trial_result_artifacts is None
        or any(
            len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
            for value in roots.values()
        )
    ):
        return stop("invalid artifact binding or trusted root")

    report = evaluate_cost_proof_file(
        args.input,
        args.config,
        source_data_manifest_path=args.source_data_manifest,
        source_run_receipt_path=args.source_run_receipt,
        search_run_receipt_path=args.search_run_receipt,
        router_replay_manifest_path=args.router_replay_manifest,
        router_source_artifact_path=args.router_source_artifact,
        lifecycle_path=args.lifecycle,
        membership_path=args.membership,
        trial_ledger_path=args.trial_ledger,
        producer_source_path=args.producer_source,
        commit_receipt_path=args.commit_receipt,
        router_producer_source_path=args.router_producer_source,
        router_commit_receipt_path=args.router_commit_receipt,
        market_artifact_paths=market_artifacts,
        funding_artifact_paths=funding_artifacts,
        router_artifact_paths=router_artifacts,
        trial_result_artifact_paths=trial_result_artifacts,
        trusted_roots=roots,
    )
    print(report.to_json())
    return {"PASS": 0, "REJECT": 1, "STOP": 2}[report.status]


if __name__ == "__main__":
    raise SystemExit(main())
