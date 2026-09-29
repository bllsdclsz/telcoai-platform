"""Command line entry point: ``rag ingest | search | ask | serve | eval-retrieval``."""

import argparse
import json
import sys
from dataclasses import asdict

import yaml

from support_rag.config import LANGUAGES, Settings


def _print_report_table(reports: list) -> None:
    head = f"{'dense model':55} {'sparse':12} {'hit@1':>6} {'hit@3':>6} {'mrr':>6} "
    head += " ".join(f"{lang}@3" for lang in LANGUAGES) + f" {'xling@3':>7} {'ms/q':>6}"
    print(head)
    for r in reports:
        c, o = r.config, r.overall
        row = f"{c['dense_model']:55} {str(c['sparse_model'] or '-')[:12]:12} "
        row += f"{o['hit@1']:6.3f} {o['hit@3']:6.3f} {o['mrr@10']:6.3f} "
        row += " ".join(f"{r.by_lang[lang]['hit@3']:5.3f}" for lang in LANGUAGES)
        row += f" {r.unfiltered['hit@3']:7.3f} {r.ms_per_query:6.1f}"
        print(row)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rag")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", help="chunk, embed and index the help-center corpus")
    search = sub.add_parser("search", help="search the index")
    search.add_argument("query")
    search.add_argument("--lang", choices=LANGUAGES)
    search.add_argument("-k", type=int, default=5)
    ask = sub.add_parser("ask", help="answer a question with the configured LLM")
    ask.add_argument("question")
    ask.add_argument("--lang", choices=LANGUAGES, required=True)
    serve = sub.add_parser("serve", help="run the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8100)
    ev = sub.add_parser("eval-retrieval", help="score retrieval on the golden questions")
    ev.add_argument(
        "--dense", action="append", help="dense model(s) to compare (default: configured)"
    )
    ev.add_argument("--no-sparse", action="store_true", help="dense-only retrieval")
    ev.add_argument("--gate", action="store_true", help="exit 1 if below eval/thresholds.yaml")
    ev.add_argument("--report", help="write the JSON report(s) to this file")
    args = parser.parse_args(argv)

    settings = Settings()
    match args.command:
        case "ingest":
            from support_rag.chunking import chunk_article
            from support_rag.corpus import load_corpus
            from support_rag.embeddings import FastEmbedDense, FastEmbedSparse
            from support_rag.index import build_index, connect

            articles = load_corpus(settings.corpus_dir)
            chunks = [c for a in articles for c in chunk_article(a, settings.chunk_max_words)]
            sparse = FastEmbedSparse(settings.sparse_model) if settings.sparse_model else None
            n = build_index(
                connect(settings),
                settings.collection,
                chunks,
                FastEmbedDense(settings.dense_model),
                sparse,
            )
            print(f"indexed {len(articles)} articles as {n} chunks into '{settings.collection}'")
        case "search":
            from support_rag.embeddings import FastEmbedDense, FastEmbedSparse
            from support_rag.index import connect
            from support_rag.retrieve import Retriever

            sparse = FastEmbedSparse(settings.sparse_model) if settings.sparse_model else None
            retriever = Retriever(
                connect(settings), settings.collection, FastEmbedDense(settings.dense_model), sparse
            )
            for h in retriever.search(args.query, k=args.k, lang=args.lang):
                print(f"{h.score:.3f}  [{h.lang}] {h.article_id:28} {h.title}")
        case "ask":
            from support_rag.api import build_assistant

            answer = build_assistant(settings).ask(args.question, args.lang)
            print(answer.text)
            print(
                f"\n[{answer.reason}] {answer.prompt} | {answer.model} | {answer.latency_ms:.0f} ms"
            )
            for src in answer.cited:
                print(f"  [{src.n}] {src.title} - {src.url}")
        case "serve":
            import uvicorn

            from support_rag.api import create_app

            uvicorn.run(create_app(), host=args.host, port=args.port)
        case "eval-retrieval":
            from support_rag.evaluate import check_thresholds, run_retrieval_eval

            sparse_model = None if args.no_sparse else settings.sparse_model
            reports = [
                run_retrieval_eval(settings, model, sparse_model)
                for model in (args.dense or [settings.dense_model])
            ]
            _print_report_table(reports)
            if args.report:
                with open(args.report, "w", encoding="utf-8") as f:
                    json.dump([asdict(r) for r in reports], f, indent=2)
            if args.gate:
                thresholds = yaml.safe_load(
                    (settings.eval_dir / "thresholds.yaml").read_text(encoding="utf-8")
                )["retrieval"]
                violations = check_thresholds(reports[0], thresholds)
                for r in reports[0].failures:
                    print(
                        f"  miss [{r['lang']}] {r['q']!r}: expected {r['expected']}, "
                        f"got {r['top1']} (rank {r['rank']})"
                    )
                if violations:
                    print("RETRIEVAL GATE FAILED:\n  " + "\n  ".join(violations))
                    sys.exit(1)
                print("retrieval gate passed")
