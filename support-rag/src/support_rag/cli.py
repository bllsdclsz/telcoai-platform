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
    sev = sub.add_parser("eval-safety", help="score the input guardrails on eval/safety.yaml")
    sev.add_argument("--gate", action="store_true", help="exit 1 if below eval/thresholds.yaml")
    sev.add_argument("--rules-only", action="store_true", help="skip the learned classifier")
    sub.add_parser("train-injection", help="train the prompt-injection classifier artifact")
    aud = sub.add_parser("audit", help="query the request traces (audit log)")
    aud.add_argument("--reason", help="e.g. blocked_input, ungrounded, answered")
    aud.add_argument("--lang", choices=LANGUAGES)
    aud.add_argument("--request-id", help="the request_id returned by /ask")
    aud.add_argument("--limit", type=int, default=20)
    cal = sub.add_parser("calibrate-judge", help="score the LLM judge on known answers")
    cal.add_argument("--judge", help="judge model (default: configured)")
    gen = sub.add_parser("eval-generation", help="answer quality: facts, language, LLM judge")
    gen.add_argument("--model", help="generator model (default: configured)")
    gen.add_argument("--prompt-version", type=int, help="answer prompt version (default: latest)")
    gen.add_argument("--judge", help="judge model (default: configured)")
    gen.add_argument("--no-judge", action="store_true", help="deterministic checks only")
    gen.add_argument("--gate", action="store_true", help="exit 1 if below eval/thresholds.yaml")
    gen.add_argument("--report", help="write the JSON report to this file")
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
        case "calibrate-judge":
            from support_rag.generation_eval import Judge, calibrate_judge
            from support_rag.llm import LiteLLMChat
            from support_rag.prompts import load_prompt

            judge_llm = LiteLLMChat(
                args.judge or settings.judge_model,
                api_base=settings.llm_api_base,
                reasoning_effort=settings.reasoning_effort,
            )
            cal_report = calibrate_judge(
                settings, Judge(judge_llm, load_prompt(settings.prompts_dir, "judge"))
            )
            acc = "  ".join(f"{c} {v:.0%}" for c, v in cal_report.accuracy.items())
            print(
                f"judge {cal_report.judge_model} ({cal_report.judge_prompt}) + style check "
                f"on {cal_report.n} known answers: {acc}"
                f"  (parse errors: {cal_report.parse_errors})"
            )
            for d in cal_report.disagreements:
                print(
                    f"  {d['id']:22} {d['criterion']:9} expected {d['expected']!s:5} "
                    f"got {d['predicted']!s:5} | {d['reason'][:90]}"
                )
        case "eval-generation":
            from support_rag.api import build_assistant
            from support_rag.generation_eval import (
                Judge,
                check_generation_thresholds,
                log_to_mlflow,
                run_generation_eval,
            )
            from support_rag.llm import LiteLLMChat
            from support_rag.prompts import load_prompt

            if args.model:
                settings = settings.model_copy(update={"llm_model": args.model})
            if args.prompt_version:
                settings = settings.model_copy(update={"prompt_version": args.prompt_version})
            judge = None
            if not args.no_judge:
                judge_llm = LiteLLMChat(
                    args.judge or settings.judge_model,
                    api_base=settings.llm_api_base,
                    reasoning_effort=settings.reasoning_effort,
                )
                judge = Judge(judge_llm, load_prompt(settings.prompts_dir, "judge"))
            gen_report = run_generation_eval(settings, build_assistant(settings), judge)
            print(f"{gen_report.model} | {gen_report.prompt} | judge {gen_report.judge_model}")
            for k, v in gen_report.metrics.items():
                print(f"  {k:20} {v:.3f}")
            for c in gen_report.cases:
                problems = []
                if c.expect == "answer" and c.reason != "answered":
                    problems.append(f"not answered ({c.reason})")
                if c.expect == "decline" and c.reason == "answered":
                    problems.append("answered but should decline")
                if c.facts_ok is False:
                    problems.append("fact missing")
                problems += [f"style: {issue}" for issue in c.style_issues]
                if c.verdict:
                    problems += [
                        f"judge: not {k}" for k in ("faithful", "relevant") if c.verdict[k] is False
                    ]
                if problems:
                    print(f"  ! [{c.lang}] {c.q[:50]:50} {', '.join(problems)} | {c.answer[:70]!r}")
            if args.report:
                with open(args.report, "w", encoding="utf-8") as f:
                    json.dump(asdict(gen_report), f, ensure_ascii=False, indent=1)
            if settings.eval_mlflow_uri:
                run_id = log_to_mlflow(
                    gen_report, settings.eval_mlflow_uri, settings.eval_experiment
                )
                print(f"logged to MLflow run {run_id}")
            if args.gate:
                thresholds = yaml.safe_load(
                    (settings.eval_dir / "thresholds.yaml").read_text(encoding="utf-8")
                )["generation"]
                violations = check_generation_thresholds(gen_report, thresholds)
                if violations:
                    print("GENERATION GATE FAILED:\n  " + "\n  ".join(violations))
                    sys.exit(1)
                print("generation gate passed")
        case "audit":
            from datetime import UTC, datetime

            from support_rag.tracing import audit_log

            if not settings.trace_mlflow_uri:
                sys.exit("set RAG_TRACE_MLFLOW_URI to the MLflow server that stores the traces")
            records = audit_log(
                settings.trace_mlflow_uri,
                settings.trace_experiment,
                reason=args.reason,
                lang=args.lang,
                request_id=args.request_id,
                limit=args.limit,
            )
            for r in records:
                when = datetime.fromtimestamp(r["time_ms"] / 1000, UTC).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                print(
                    f"{when} {r['request_id'][:8]} [{r['lang']}] {r['reason']:19} "
                    f"{r['duration_ms'] or 0:>6} ms pii={r['pii_types'] or '-':10} "
                    f"cited={r['cited'] or '-'} | {(r['question'] or '')[:60]!r}"
                )
                if r["guard"]:
                    print(f"{'':28}guard: {r['guard'][:100]}")
            print(f"{len(records)} record(s)")
        case "train-injection":
            from support_rag.embeddings import FastEmbedDense
            from support_rag.injection_model import save, train_injection_classifier

            if settings.injection_classifier is None:
                sys.exit("RAG_INJECTION_CLASSIFIER is disabled")
            clf = train_injection_classifier(settings, FastEmbedDense(settings.dense_model))
            save(clf, settings.injection_classifier)
            print(f"saved {settings.injection_classifier}: {clf.metadata}")
        case "eval-safety":
            from support_rag.safety_eval import check_safety_thresholds, evaluate_safety

            classifier = encoder = None
            if not args.rules_only and settings.injection_classifier:
                from support_rag.embeddings import FastEmbedDense
                from support_rag.injection_model import InjectionClassifier

                classifier = InjectionClassifier.load(settings.injection_classifier)
                encoder = FastEmbedDense(classifier.embedding_model)
            rep = evaluate_safety(settings, classifier, encoder)
            print(
                f"{'detector':22} tuning({rep.n_tuning}) held-out({rep.n_holdout}) "
                f"FP held-out({rep.n_benign}) FP golden({rep.n_golden})"
            )
            for name, sc in [("rules", rep.rules), ("rules + classifier", rep.combined)]:
                if sc is not None:
                    print(
                        f"{name:22} {sc.tuning_recall:10.1%} {sc.holdout_recall:12.1%} "
                        f"{sc.holdout_fpr:15.1%} {sc.golden_fpr:13.1%}"
                    )
            print(f"PII recall/precision {rep.pii_recall:.1%} / {rep.pii_precision:.1%}")
            for m in rep.shipped.missed_holdout:
                print(f"  missed held-out attack: {m}")
            for b in rep.shipped.false_positives:
                print(f"  false positive: {b}")
            for e in rep.pii_errors:
                print(f"  pii: {e['text']!r} expected {e['expected']} got {e['found']}")
            if args.gate:
                thresholds = yaml.safe_load(
                    (settings.eval_dir / "thresholds.yaml").read_text(encoding="utf-8")
                )["safety"]
                violations = check_safety_thresholds(rep, thresholds)
                if violations:
                    print("SAFETY GATE FAILED:\n  " + "\n  ".join(violations))
                    sys.exit(1)
                print("safety gate passed")
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
