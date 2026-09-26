"""Command line entry point: ``churn download | train | serve``."""

import argparse
import json

from churn.config import Settings


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="churn")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download", help="download the raw Telco churn dataset")
    sub.add_parser("train", help="train, evaluate and register a model")
    serve = sub.add_parser("serve", help="run the scoring API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
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
        case "serve":
            import uvicorn

            uvicorn.run("churn.serve.app:app", host=args.host, port=args.port)
