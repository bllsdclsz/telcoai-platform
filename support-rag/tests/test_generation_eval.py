import pytest
import yaml

from support_rag.assistant import Answer, Source
from support_rag.config import Settings
from support_rag.generation_eval import (
    CaseResult,
    GenerationReport,
    Judge,
    calibrate_judge,
    check_generation_thresholds,
    detect_language,
    facts_present,
    parse_verdict,
    run_generation_eval,
    style_issues,
)
from support_rag.llm import Completion
from support_rag.prompts import load_prompt


@pytest.mark.parametrize(
    ("text", "lang"),
    [
        ("Eine Ersatz-SIM kostet CHF 40 und kommt innert 2 Arbeitstagen per Post [1].", "de"),
        ("Avec Swiss+, vous avez 5 Go de données par mois dans l'UE [1].", "fr"),
        ("Il secondo sollecito costa CHF 30 e arriva dopo 10 giorni [1].", "it"),
        ("The reset link is valid for 1 hour and you can request a new one [1].", "en"),
    ],
)
def test_detect_language(text: str, lang: str) -> None:
    assert detect_language(text) == lang


def test_facts_accept_alternative_spellings() -> None:
    assert facts_present("costs CHF 3,50 per invoice", ["3.50|3,50"])
    assert facts_present("takes 3 to 5 days", ["3", "5"])
    assert not facts_present("takes 3 days", ["3", "5"])


@pytest.mark.parametrize(
    ("answer", "lang", "issue"),
    [
        ("A new SIM costs CHF 40 [1].", "de", "wrong language"),
        (
            "La SIM costa CHF 40 e arriva in 2 giorni [1].\n"
            "La SIM costa CHF 40 e arriva in 2 giorni [1].",
            "it",
            "repeated sentence",
        ),
        (
            "A new SIM costs CHF 40 [1].\n\nAnswer in English, concise, with citations.",
            "en",
            "echo or meta text",
        ),
        ("Based on the provided sources, a new SIM costs CHF 40 [1].", "en", "echo or meta text"),
        ("Selon les sources, la carte SIM coûte CHF 40 [1].", "fr", "echo or meta text"),
    ],
)
def test_style_issues_are_detected(answer: str, lang: str, issue: str) -> None:
    assert issue in style_issues(answer, lang)


def test_clean_answers_have_no_style_issues() -> None:
    assert (
        style_issues("Die Portierung dauert 3 bis 5 Arbeitstage und ist kostenlos [1].", "de") == []
    )
    assert style_issues("Le répéteur coûte CHF 99 ou CHF 4 par mois en location [1].", "fr") == []


def test_parse_verdict_is_robust() -> None:
    v = parse_verdict('Here: {"faithful": true, "relevant": false, "reason": "off topic"} ok')
    assert (v.faithful, v.relevant, v.reason, v.error) == (True, False, "off topic", "")
    assert parse_verdict("no json at all").error
    assert parse_verdict('{"faithful": "yes"}').error.startswith("missing")


class ScriptedJudgeLLM:
    """Judges by keyword: 'CHF 25' is unfaithful, 'free of charge' is off-question."""

    model = "fake/judge"

    def complete(self, messages, *, temperature, max_tokens) -> Completion:  # type: ignore[no-untyped-def]
        answer = messages[1]["content"].split("<<<")[1]
        faithful = "CHF 25" not in answer and "gratuito" not in answer and "5 GB" not in answer
        relevant = "free of charge" not in answer and "upgrade" not in answer
        text = f'{{"faithful": {str(faithful).lower()}, "relevant": {str(relevant).lower()}}}'
        return Completion(text, self.model, 1, 1, 1.0)


def test_calibration_scores_judge_and_style_check(settings: Settings) -> None:
    judge = Judge(ScriptedJudgeLLM(), load_prompt(settings.prompts_dir, "judge"))
    report = calibrate_judge(settings, judge)
    n = len(
        yaml.safe_load((settings.eval_dir / "judge_calibration.yaml").read_text(encoding="utf-8"))
    )
    assert report.n == n and report.parse_errors == 0
    assert set(report.accuracy) == {"faithful", "relevant", "clean"}
    assert report.accuracy["clean"] == 1.0  # the deterministic style check on the labeled set


class FakeAssistant:
    """Answers every answerable golden question with its first fact; declines the rest."""

    def __init__(self, settings: Settings) -> None:
        cases = yaml.safe_load(
            (settings.eval_dir / "generation_golden.yaml").read_text(encoding="utf-8")
        )
        self.facts = {c["q"]: c.get("facts") for c in cases}
        self.prompt = load_prompt(settings.prompts_dir, "answer")

    def ask(self, question: str, lang: str) -> Answer:
        facts = self.facts[question]
        if not facts:
            return Answer("fallback", lang, "model_no_answer", model="fake/gen")
        words = {
            "de": "Das kostet und dauert",
            "fr": "Cela coûte et dure pour vous",
            "it": "Questo costa e dura per il",
            "en": "This is the cost and the time for",
        }
        text = f"{words[lang]} {' '.join(f.split('|')[0] for f in facts)} [1]."
        src = Source(1, "a.en", "t", "u", "text", 0.9)
        return Answer(text, lang, "answered", cited=[src], model="fake/gen")


def test_generation_eval_metrics_without_judge(settings: Settings) -> None:
    report = run_generation_eval(settings, FakeAssistant(settings))  # type: ignore[arg-type]
    m = report.metrics
    assert m["answer_rate"] == 1.0 and m["fact_accuracy"] == 1.0 and m["decline_accuracy"] == 1.0
    assert m["style_clean"] == 1.0 and m["quality_pass"] == 1.0
    assert "judge_faithful" not in m and report.model == "fake/gen"


def test_thresholds_report_missing_and_low_metrics() -> None:
    report = GenerationReport("m", "p@v1", "sha", None, {"fact_accuracy": 0.5}, [])
    violations = check_generation_thresholds(report, {"fact_accuracy": 0.9, "judge_faithful": 0.8})
    assert violations == [
        "fact_accuracy 0.500 < 0.9",
        "judge_faithful: not measured (run with the judge)",
    ]


def test_case_result_defaults() -> None:
    r = CaseResult("q", "en", "answer", "answered", "a", [], 0.1)
    assert r.style_issues == [] and r.verdict is None
