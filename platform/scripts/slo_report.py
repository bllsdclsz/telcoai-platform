"""SLO report: availability, latency and error budget per service and environment, from Prometheus.

    make prometheus-ui &   # port-forward to http://localhost:9090
    make slo-report        # or: python platform/scripts/slo_report.py --window 7d > report.md

Standard library only. Uses the slo:* series recorded by platform/charts/slo (every 30 s), so
request counts are estimates (rate x interval); the ratios are exact.
"""

import argparse
import json
import sys
import urllib.parse
import urllib.request
from datetime import UTC, datetime

AVAILABILITY_OBJECTIVE = 0.995
LATENCY_OBJECTIVE = 0.99
RULE_INTERVAL_S = 30


def query(base: str, promql: str) -> dict[tuple[str, str], float]:
    url = f"{base.rstrip('/')}/api/v1/query?" + urllib.parse.urlencode({"query": promql})
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.load(resp)["data"]["result"]
    return {
        (r["metric"].get("env", "?"), r["metric"].get("service", "?")): float(r["value"][1])
        for r in data
    }


def firing_alerts(base: str) -> list[dict[str, str]]:
    with urllib.request.urlopen(f"{base.rstrip('/')}/api/v1/alerts", timeout=30) as resp:
        alerts = json.load(resp)["data"]["alerts"]
    return [
        a["labels"] | {"summary": a["annotations"].get("summary", "")}
        for a in alerts
        if a["state"] == "firing"
    ]


def pct(x: float | None, digits: int = 2) -> str:
    return "n/a" if x is None else f"{100 * x:.{digits}f}%"


HEADER = [
    "| Env | Service | Data | Requests | Availability | Budget left | Latency SLI | Burn (1 h) |",
    "| --- | ------- | ---- | -------- | ------------ | ----------- | ----------- | ---------- |",
]


def report(base: str, window: str) -> str:
    def over(fn: str, series: str, scale: int = 1) -> dict[tuple[str, str], float]:
        return query(base, f"sum by (env, service) ({fn}({series}[{window}])) * {scale}")

    total = over("sum_over_time", "slo:requests:rate5m", RULE_INTERVAL_S)
    errors = over("sum_over_time", 'slo:requests:rate5m{status=~"5.."}', RULE_INTERVAL_S)
    slow = over("sum_over_time", "slo:requests_slow:rate5m")
    timed = over("sum_over_time", "slo:requests_timed:rate5m")
    # samples of one series x interval = how much of the window has data
    covered = query(
        base,
        f"max by (env, service) (count_over_time(slo:requests:rate5m[{window}]))"
        f" * {RULE_INTERVAL_S}",
    )
    burn = query(base, "slo:error_ratio:rate1h / 0.005")

    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    out = [
        f"# SLO report ({window} window, {now})",
        "",
        f"Objectives: availability {pct(AVAILABILITY_OBJECTIVE, 1)} of /predict requests not"
        f" 5xx; latency {pct(LATENCY_OBJECTIVE, 0)} faster than 300 ms."
        " Budget left: share of the allowed failures not yet used.",
        "",
        *HEADER,
    ]
    order = {"prod": 0, "test": 1, "dev": 2}
    for key in sorted(total, key=lambda k: (order.get(k[0], 9), k[1])):
        n = total[key]
        err_ratio = errors.get(key, 0.0) / n if n else None
        availability = None if err_ratio is None else 1 - err_ratio
        budget_left = None if err_ratio is None else 1 - err_ratio / (1 - AVAILABILITY_OBJECTIVE)
        latency = 1 - slow[key] / timed[key] if timed.get(key) else None
        hours = covered.get(key, 0) / 3600
        avail_miss = availability is not None and availability < AVAILABILITY_OBJECTIVE
        latency_miss = latency is not None and latency < LATENCY_OBJECTIVE
        burn_1h = f"{burn[key]:.1f}x" if key in burn else "n/a"
        out.append(
            f"| {key[0]} | {key[1]} | {hours:.1f} h | ≈{n:,.0f} "
            f"| {pct(availability)}{' ⚠' if avail_miss else ''} | {pct(budget_left, 0)} "
            f"| {pct(latency)}{' ⚠' if latency_miss else ''} | {burn_1h} |"
        )
    alerts = sorted(firing_alerts(base), key=lambda a: (a.get("env", ""), a["alertname"]))
    out += ["", f"## Firing alerts ({len(alerts)})", ""]
    out += [
        f"- **{a['alertname']}** ({a.get('severity', '?')}, {a.get('env', '?')}): {a['summary']}"
        for a in alerts
    ] or ["None."]
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prometheus", default="http://localhost:9090")
    parser.add_argument("--window", default="30d", help="PromQL duration, e.g. 30d, 7d, 6h")
    args = parser.parse_args()
    sys.stdout.buffer.write(report(args.prometheus, args.window).encode("utf-8"))


if __name__ == "__main__":
    main()
