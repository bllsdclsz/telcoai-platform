"""Answer-quality evaluation: deterministic checks + an LLM judge that is itself calibrated.

`rag eval-generation` runs the assistant on eval/generation_golden.yaml and scores:
- deterministic: expected facts present, a source cited, correct declines, and style (answer
  language, no duplicated sentences, no echoed instructions or meta commentary);
- LLM judge (a different model family than the generator, so no model grades itself):
  faithful / relevant, pass or fail per answer.
`rag calibrate-judge` scores the judge and the style check on eval/judge_calibration.yaml
(known-good and known-bad answers). Calibration moved "clean" out of the judge: qwen2.5 7B was
right on only 71% of style verdicts (see support-rag/README.md). Generations run first and
judgments after, so a small GPU never has to swap two models per question.
"""

import json
import re
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

import yaml

from support_rag.assistant import LANGUAGE_NAMES, Answer, Assistant
from support_rag.config import Settings
from support_rag.corpus import load_corpus
from support_rag.llm import ChatModel
from support_rag.prompts import PromptTemplate

JUDGE_CRITERIA = ("faithful", "relevant")

# Frequent function words: enough to tell DE/FR/IT/EN apart in a few sentences.
_STOPWORDS = {
    "de": "der die das und ist nicht sie ich mit von den ein eine zu im auf für wird sind bei "
    "ihre ihr oder auch werden kann nach über wenn dem des einer",
    "fr": "le la les et est pas vous je avec de des un une pour dans sur par au aux votre vos "
    "qui que ne plus sont être peut ce cette",
    "it": "il lo la le gli e è non tu io con di del della un una per in su da al ai tuo tua "
    "che sono essere puoi questo questa gli nel",
    "en": "the and is not you i with of a an to in on for your are be can this that it by "
    "from or will at",
}
_STOP_SETS = {lang: set(words.split()) for lang, words in _STOPWORDS.items()}

# Text a customer-facing answer never contains: echoed instructions and talk about the
# assistant itself, its rules or its sources (DE/FR/IT/EN).
_META_PATTERNS = [
    r"\banswer in (english|german|french|italian)\b",
    r"\b(concise|at most \d+ sentences|with citations)\b",
    r"\bas an ai\b",
    r"\b(based on|according to|from) the (provided |given |numbered )?sources?\b",
    r"\bthe (sources|rules|instructions) (do not|don't|say|state)\b",
    r"^\s*(answer|final answer|antwort|reponse|risposta)\s*:",
    r"\b(als ki|gemass den (quellen|anweisungen)|laut den quellen)\b",
    r"\b(en tant qu'ia|selon les sources|d'apres les sources)\b",
    r"\b(come ia|secondo le fonti|in base alle fonti)\b",
]
_META = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in _META_PATTERNS]


def detect_language(text: str) -> str:
    tokens = re.findall(r"[a-zàâäçéèêëîïôöùûüœß']+", text.lower())
    scores = {lang: sum(t in words for t in tokens) for lang, words in _STOP_SETS.items()}
    return max(scores, key=lambda lang: scores[lang])


def _strip_accents(text: str) -> str:
    import unicodedata

    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def style_issues(answer: str, lang: str) -> list[str]:
    """Deterministic style problems: wrong language, repeated sentences, echo/meta text."""
    issues = []
    if detect_language(answer) != lang:
        issues.append("wrong language")
    sentences = [
        re.sub(r"\W+", " ", s).strip().lower()
        for s in re.split(r"(?<=[.!?])\s+|\n+", re.sub(r"\[\d+\]", "", answer))
    ]
    long_sentences = [s for s in sentences if len(s.split()) >= 4]
    if len(long_sentences) != len(set(long_sentences)):
        issues.append("repeated sentence")
    normalized = _strip_accents(answer)
    if any(p.search(normalized) for p in _META):
        issues.append("echo or meta text")
    return issues


def facts_present(text: str, facts: list[str]) -> bool:
    """Every fact must appear; a fact may list accepted spellings separated by '|'."""
    lowered = text.lower()
    return all(any(alt.lower() in lowered for alt in fact.split("|")) for fact in facts)


# --- Judge ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Verdict:
    faithful: bool | None
    relevant: bool | None
    reason: str = ""
    error: str = ""

    def get(self, criterion: str) -> bool | None:
        value: bool | None = getattr(self, criterion)
        return value


def parse_verdict(text: str) -> Verdict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return Verdict(None, None, error="no JSON object in judge output")
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError as exc:
        return Verdict(None, None, error=f"invalid JSON: {exc}")
    values = {c: data.get(c) if isinstance(data.get(c), bool) else None for c in JUDGE_CRITERIA}
    missing = [c for c, v in values.items() if v is None]
    return Verdict(
        faithful=values["faithful"],
        relevant=values["relevant"],
        reason=str(data.get("reason", ""))[:300],
        error=f"missing {missing}" if missing else "",
    )


class Judge:
    def __init__(self, llm: ChatModel, prompt: PromptTemplate) -> None:
        self.llm, self.prompt = llm, prompt

    def __call__(self, question: str, lang: str, sources: str, answer: str) -> Verdict:
        completion = self.llm.complete(
            self.prompt.render(
                language_name=LANGUAGE_NAMES[lang].split(" (")[0],
                question=question,
                sources=sources,
                answer=answer,
            ),
            temperature=0.0,
            max_tokens=200,
        )
        return parse_verdict(completion.text)


@dataclass
class CalibrationReport:
    judge_model: str
    judge_prompt: str
    n: int
    accuracy: dict[str, float]  # judge criteria + "clean" from the deterministic style check
    parse_errors: int
    disagreements: list[dict[str, Any]] = field(default_factory=list)


def calibrate_judge(settings: Settings, judge: Judge) -> CalibrationReport:
    articles = {a.id: a for a in load_corpus(settings.corpus_dir)}
    items = yaml.safe_load(
        (settings.eval_dir / "judge_calibration.yaml").read_text(encoding="utf-8")
    )
    correct = Counter[str]()
    errors, disagreements = 0, []
    for item in items:
        article = articles[item["source"]]
        verdict = judge(
            item["q"], article.lang, f"[1] {article.title}\n\n{article.body}", item["answer"]
        )
        errors += bool(verdict.error)
        issues = style_issues(item["answer"], article.lang)
        predictions = {c: verdict.get(c) for c in JUDGE_CRITERIA} | {"clean": not issues}
        for criterion, predicted in predictions.items():
            expected = item["labels"][criterion]
            if predicted == expected:
                correct[criterion] += 1
            else:
                disagreements.append(
                    {
                        "id": item["id"],
                        "criterion": criterion,
                        "expected": expected,
                        "predicted": predicted,
                        "reason": verdict.reason if criterion in JUDGE_CRITERIA else str(issues),
                    }
                )
    return CalibrationReport(
        judge_model=judge.llm.model,
        judge_prompt=judge.prompt.ref,
        n=len(items),
        accuracy={c: correct[c] / len(items) for c in (*JUDGE_CRITERIA, "clean")},
        parse_errors=errors,
        disagreements=disagreements,
    )


# --- Generation evaluation -------------------------------------------------------------------


@dataclass
class CaseResult:
    q: str
    lang: str
    expect: str
    reason: str
    answer: str
    cited: list[str]
    latency_s: float
    facts_ok: bool | None = None
    style_issues: list[str] = field(default_factory=list)
    verdict: dict[str, Any] | None = None


@dataclass
class GenerationReport:
    model: str
    prompt: str
    prompt_sha256: str
    judge_model: str | None
    metrics: dict[str, float]
    cases: list[CaseResult]


def _verdict(r: CaseResult, criterion: str) -> bool | None:
    return None if r.verdict is None else r.verdict[criterion]


def _quality_pass(r: CaseResult) -> bool | None:
    """Answered, right facts, clean style, and (when judged) faithful and relevant."""
    judged = [_verdict(r, c) for c in JUDGE_CRITERIA]
    if r.verdict is not None and None in judged:
        return None
    return bool(r.facts_ok) and not r.style_issues and all(v is not False for v in judged)


def _rate(values: list[bool | None]) -> float:
    known = [v for v in values if v is not None]
    return sum(known) / len(known) if known else float("nan")


def run_generation_eval(
    settings: Settings, assistant: Assistant, judge: Judge | None = None
) -> GenerationReport:
    cases = yaml.safe_load(
        (settings.eval_dir / "generation_golden.yaml").read_text(encoding="utf-8")
    )
    results: list[tuple[dict[str, Any], Answer, CaseResult]] = []
    for case in cases:  # all generations first ...
        start = time.perf_counter()
        ans = assistant.ask(case["q"], case["lang"])
        r = CaseResult(
            case["q"],
            case["lang"],
            case["expect"],
            ans.reason,
            ans.text,
            [s.article_id for s in ans.cited],
            round(time.perf_counter() - start, 2),
        )
        if case["expect"] == "answer" and ans.answered:
            r.facts_ok = facts_present(ans.text, case.get("facts", []))
            r.style_issues = style_issues(ans.text, case["lang"])
        results.append((case, ans, r))

    if judge is not None:  # ... then all judgments (one model loaded at a time)
        for case, ans, r in results:
            if case["expect"] == "answer" and ans.answered:
                sources = "\n\n".join(f"[{s.n}] {s.text}" for s in ans.cited)
                r.verdict = asdict(judge(case["q"], case["lang"], sources, ans.text))

    answerable = [r for _, _, r in results if r.expect == "answer"]
    declines = [r for _, _, r in results if r.expect == "decline"]
    answered = [r for r in answerable if r.reason == "answered"]
    metrics = {
        "answer_rate": len(answered) / len(answerable),
        "fact_accuracy": sum(bool(r.facts_ok) for r in answerable) / len(answerable),
        "style_clean": _rate([not r.style_issues for r in answered]),
        "decline_accuracy": sum(r.reason != "answered" for r in declines) / len(declines),
        "median_latency_s": statistics.median(r.latency_s for r in answerable),
    }
    if judge is not None:
        for c in JUDGE_CRITERIA:
            metrics[f"judge_{c}"] = _rate([_verdict(r, c) for r in answered])
    # Share of answerable questions that got a fully good answer (unanswered ones count as fail).
    metrics["quality_pass"] = sum(bool(_quality_pass(r)) for r in answered) / len(answerable)
    any_answer = next((a for _, a, _ in results if a.model), None)
    return GenerationReport(
        model=any_answer.model if any_answer else "",
        prompt=assistant.prompt.ref,
        prompt_sha256=assistant.prompt.sha256,
        judge_model=judge.llm.model if judge else None,
        metrics={k: round(v, 4) for k, v in metrics.items()},
        cases=[r for _, _, r in results],
    )


def log_to_mlflow(report: GenerationReport, tracking_uri: str, experiment: str) -> str:
    """One MLflow run per evaluation: compare models and prompt versions side by side."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=f"{report.model} | {report.prompt}") as run:
        mlflow.log_params(
            {
                "model": report.model,
                "prompt": report.prompt,
                "prompt_sha256": report.prompt_sha256,
                "judge_model": report.judge_model,
            }
        )
        mlflow.log_metrics(report.metrics)
        mlflow.log_dict({"cases": [asdict(c) for c in report.cases]}, "cases.json")
    return run.info.run_id


def check_generation_thresholds(
    report: GenerationReport, thresholds: dict[str, float]
) -> list[str]:
    violations = []
    for name, minimum in thresholds.items():
        value = report.metrics.get(name)
        if value is None:
            violations.append(f"{name}: not measured (run with the judge)")
        elif value < minimum:
            violations.append(f"{name} {value:.3f} < {minimum}")
    return violations
