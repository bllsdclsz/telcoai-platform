"""Command line entry point for the churn pipeline, API and monitoring."""

import argparse
import json

from churn.config import Settings


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="churn")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download", help="download the raw Telco churn dataset")
    sub.add_parser("train", help="train, evaluate and register a model")
    promote_cmd = sub.add_parser("promote", help="move a model version between aliases")
    promote_cmd.add_argument("--from", dest="source", required=True, help="e.g. dev, staging")
    promote_cmd.add_argument("--to", dest="target", required=True, help="e.g. staging, prod")
    pipeline_cmd = sub.add_parser("pipeline", help="run the Prefect training flow")
    pipeline_cmd.add_argument("--promote-to", default="staging", help="'' to skip promotion")
    sub.add_parser("features", help="publish features to Feast (offline + online store)")
    monitor_cmd = sub.add_parser("monitor", help="run the Prefect drift monitoring flow")
    monitor_cmd.add_argument("--no-retrain", action="store_true", help="report drift only")
    simulate_cmd = sub.add_parser("simulate", help="send customer traffic to the API")
    simulate_cmd.add_argument("--n", type=int, default=500, help="customers to send")
    simulate_cmd.add_argument("--drift", action="store_true", help="apply the shift scenario")
    simulate_cmd.add_argument("--api-url", default="http://127.0.0.1:8000")
    simulate_cmd.add_argument("--seed", type=int, default=0)
    serve_cmd = sub.add_parser("serve", help="run the scoring API")
    serve_cmd.add_argument("--host", default="127.0.0.1")
    serve_cmd.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    settings = Settings()
    match args.command:
        case "download":
            from churn.data import download

            print(download(settings.raw_data_url, settings.raw_data_path))
        case "train":
            from churn.train import train

            result = train(settings)
            summary = {
                "run_id": result.run_id,
                "version": result.model_version,
                "metrics": result.metrics,
            }
            print(json.dumps(summary, indent=2))
        case "promote":
            from churn.registry import promote

            version = promote(settings, args.source, args.target)
            print(f"{settings.registered_model_name} v{version}: {args.source} -> {args.target}")
        case "pipeline":
            from churn.flows import training_flow

            training_flow(promote_to=args.promote_to or None, settings=settings)
        case "features":
            import os

            from feast import FeatureStore

            from churn.data import load
            from churn.feature_store import publish

            os.environ.setdefault("FEAST_REDIS_CONNECTION", "localhost:6379")
            store = FeatureStore(repo_path=str(settings.feature_repo))
            customers = load(settings.raw_data_path)
            publish(store, customers, settings.offline_features_path)
            print(f"published {len(customers)} customers to Feast")
        case "monitor":
            from churn.flows import drift_monitoring_flow

            drift_monitoring_flow(retrain_on_drift=not args.no_retrain, settings=settings)
        case "simulate":
            from churn.data import load
            from churn.simulate import replay, shift

            customers = load(settings.raw_data_path).sample(
                args.n, replace=True, random_state=args.seed
            )
            if args.drift:
                customers = shift(customers, seed=args.seed)
            print(f"scored {replay(customers, args.api_url)} customers")
        case "serve":
            import uvicorn

            uvicorn.run("churn.serve.app:app", host=args.host, port=args.port)
