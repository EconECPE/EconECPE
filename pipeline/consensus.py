"""Consensus over N models' labels: span-matched voting + judge adjudication.

Silver-label construction (Phase 2):
  1. For each comment, cluster the models' pairs across models by emotion_span
     token-IoU (greedy 1:1 per model pair, union-find across models).
  2. A cluster supported by ≥ --min-support models (default 2) becomes a silver
     pair; singletons are kept in the output but excluded (included=false).
  3. Fields are decided by majority vote. Genuine ties (1-1, 1-1-1) on
     emotion / cause_category go to the JUDGE — a 4th, higher-capacity
     reasoning model (deepseek/deepseek-v4-pro, the one used for the released
     silver set). The judge sits inside the silver ensemble by design: the
     disjointness the study needs is between the *gold* annotators (Claude
     Sonnet 5, Gemini 3.5 Flash) and every silver model, so silver-vs-gold
     never scores a family against its own output. 2-1 majorities are accepted
     and recorded as agreement=2/3; --judge-splits escalates those to the judge
     too (costs more; use when funded).
  4. Spans: the medoid (highest summed IoU to the rest of the cluster);
     cause_source is re-derived from the chosen span (spec §4). Intensity: mean.
     Sarcasm: majority (auxiliary metadata per §10 outcome, κ=0.42).

Output: data/labels/<run>/consensus.jsonl — one line per comment:
    {comment_id, stratum, neutral, pairs: [{...fields, support, models,
     agreement: {field: "2/3"|"judge"|"3/3"}, included}], excluded_singletons}

    python -m pipeline.consensus --run bakeoff --sample data/samples/pilot_v03_250_seed7.jsonl
        [--judge-mock] [--judge-model X] [--min-support 2] [--judge-splits]
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import config

from .prompt import format_input
from .report import _load_labels, _load_sample, span_iou
from .textnorm import find_span_source
from .validate import _sources_of

JUDGE_MODEL = os.environ.get("ECONECPE_JUDGE_MODEL", "deepseek/deepseek-v4-pro")
VOTED_FIELDS = ("emotion", "cause_category", "target_asset", "sarcasm")
JUDGE_FIELDS = ("emotion", "cause_category")  # fields whose disputes summon the judge


# ── Clustering ───────────────────────────────────────────────────────────────

def _cluster(pairs_by_model: dict[str, list[dict]], thr: float = 0.5) -> list[list[tuple]]:
    """Group pairs across models by emotion_span IoU. Returns clusters of
    (model, pair). Greedy 1:1 matching per model-pair, then union-find."""
    nodes = [(m, i) for m, ps in pairs_by_model.items() for i in range(len(ps))]
    parent = {n: n for n in nodes}

    def find(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    def union(a, b):
        parent[find(a)] = find(b)

    models = list(pairs_by_model)
    for ai in range(len(models)):
        for bi in range(ai + 1, len(models)):
            ma, mb = models[ai], models[bi]
            cands = sorted(
                ((span_iou(x.get("emotion_span"), y.get("emotion_span")), i, j)
                 for i, x in enumerate(pairs_by_model[ma])
                 for j, y in enumerate(pairs_by_model[mb])), reverse=True)
            ua, ub = set(), set()
            for iou, i, j in cands:
                if iou < thr:
                    break
                if i in ua or j in ub:
                    continue
                ua.add(i)
                ub.add(j)
                union((ma, i), (mb, j))

    groups: dict[tuple, list[tuple]] = {}
    for n in nodes:
        groups.setdefault(find(n), []).append(n)

    clusters = []
    for members in groups.values():
        # If transitivity pulled in 2+ pairs from one model, keep that model's
        # best-connected pair; the rest become singletons.
        by_model: dict[str, list[tuple]] = {}
        for m, i in members:
            by_model.setdefault(m, []).append((m, i))
        kept, spilled = [], []
        for m, ns in by_model.items():
            if len(ns) == 1:
                kept.append(ns[0])
                continue
            def conn(n):
                p = pairs_by_model[n[0]][n[1]]
                return sum(span_iou(p.get("emotion_span"),
                                    pairs_by_model[o[0]][o[1]].get("emotion_span"))
                           for o in members if o[0] != n[0])
            best = max(ns, key=conn)
            kept.append(best)
            spilled += [n for n in ns if n != best]
        clusters.append([(m, pairs_by_model[m][i]) for m, i in kept])
        clusters += [[(m, pairs_by_model[m][i])] for m, i in spilled]
    return clusters


# ── Voting ───────────────────────────────────────────────────────────────────

def _medoid_span(spans: list[str | None]) -> str | None:
    real = [s for s in spans if s]
    if not real:
        return None
    return max(real, key=lambda s: sum(span_iou(s, o) for o in real if o is not s))


def _vote_cluster(cluster: list[tuple], item: dict) -> dict:
    """Majority-vote one cluster into a candidate consensus pair + dispute list."""
    ps = [p for _, p in cluster]
    n = len(ps)
    out: dict = {"support": n, "models": [m for m, _ in cluster],
                 "agreement": {}, "disputes": {}}

    for f in VOTED_FIELDS:
        votes = Counter(p.get(f) for p in ps)
        top, top_n = votes.most_common(1)[0]
        if top_n > n / 2:
            out[f] = top
            out["agreement"][f] = f"{top_n}/{n}"
        else:  # genuine tie
            out[f] = top  # provisional; judge (or fallback) decides
            out["agreement"][f] = "tie"
            if f in JUDGE_FIELDS:
                out["disputes"][f] = sorted(votes)

    out["intensity"] = round(sum(p["intensity"] for p in ps) / n, 2)
    out["emotion_span"] = _medoid_span([p.get("emotion_span") for p in ps])

    null_causes = sum(1 for p in ps if p.get("cause_span") is None)
    if null_causes > n / 2:
        out["cause_span"], out["cause_source"] = None, None
        out["cause_category"] = "unclear"
        out["agreement"]["cause_category"] = f"{null_causes}/{n}"
        out["disputes"].pop("cause_category", None)
    else:
        out["cause_span"] = _medoid_span([p.get("cause_span") for p in ps])
        out["cause_source"] = find_span_source(out["cause_span"], _sources_of(item))
    return out


# ── Judge ────────────────────────────────────────────────────────────────────

def _judge_messages(item: dict, pair: dict) -> list[dict]:
    with open(os.path.join(config.PROMPTS_DIR, "judge.md"), encoding="utf-8") as f:
        system = f.read()
    disputed = {f: cands for f, cands in pair["disputes"].items()}
    user = (format_input(item)
            + f"\n\nPAIR UNDER ADJUDICATION:\n"
              f"emotion_span: {pair['emotion_span']!r}\n"
              f"cause_span: {pair['cause_span']!r}\n\n"
              f"DISPUTED FIELDS (candidates): {json.dumps(disputed)}\n"
              f"Reply with a JSON object with exactly these keys: "
              f"{list(disputed)}")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _run_judge(disputes: list[dict], model: str, run_dir: str, mock: bool) -> dict[str, dict]:
    """disputes: [{_jid, item, pair}] → jid -> {field: value} (validated)."""
    from .llm import extract_json, label

    class _MockJudge:
        class chat:  # noqa: N801 — OpenAI client shape
            class completions:  # noqa: N801
                @staticmethod
                def create(*, messages, **kw):
                    from types import SimpleNamespace as NS
                    req = json.loads(messages[-1]["content"]
                                     .rsplit("(candidates): ", 1)[-1].split("\n")[0])
                    pick = {f: sorted(c)[0] for f, c in req.items()}
                    return NS(choices=[NS(message=NS(content=json.dumps(pick)),
                                          finish_reason="stop")], usage=None)

    results = label(
        disputes,
        build_messages=lambda d: _judge_messages(d["item"], d["pair"]),
        parse=extract_json,
        model=model,
        client=_MockJudge() if mock else None,
        id_of=lambda d: d["_jid"],
        out_path=os.path.join(run_dir, "judge.jsonl"),
        max_workers=config.ANNOTATE_MAX_WORKERS,
        temperature=0.0,
    )
    verdicts: dict[str, dict] = {}
    by_id = {d["_jid"]: d for d in disputes}
    for path in (os.path.join(run_dir, "judge.jsonl"),):
        if not os.path.exists(path):
            continue
        for line in open(path, encoding="utf-8"):
            rec = json.loads(line)
            d = by_id.get(rec.get("id"))
            if d is None or rec.get("error") or not isinstance(rec.get("parsed"), dict):
                continue
            ok = {}
            for f, v in rec["parsed"].items():
                if f in d["pair"]["disputes"]:
                    ok[f] = v
            if ok:
                verdicts[rec["id"]] = ok
    del results
    return verdicts


# ── Orchestration ────────────────────────────────────────────────────────────

def build_consensus(run: str, sample_path: str, min_support: int = 2,
                    judge_model: str = JUDGE_MODEL, judge_mock: bool = False,
                    judge_splits: bool = False, no_judge: bool = False,
                    emit_disagreements: str | None = None) -> str:
    run_dir = os.path.join(config.LABELS_DIR, run)
    labels = _load_labels(run_dir)
    sample = _load_sample(sample_path)
    models = list(labels)
    if len(models) < 2:
        raise SystemExit("consensus needs ≥ 2 models' labels in the run dir")

    comments: dict[str, dict] = {}
    disputes: list[dict] = []
    for cid, item in sample.items():
        pairs_by_model = {m: (labels[m].get(cid) or {}).get("pairs") or []
                          for m in models if (labels[m].get(cid) or {}).get("valid")}
        if not pairs_by_model:
            continue
        n_neutral = sum(1 for ps in pairs_by_model.values() if not ps)
        clusters = _cluster(pairs_by_model)
        voted = []
        for cl in clusters:
            v = _vote_cluster(cl, item)
            v["included"] = v["support"] >= min_support
            if judge_splits:  # escalate non-unanimous majorities (e.g. 2/3) too
                for f in JUDGE_FIELDS:
                    ag = v["agreement"].get(f, "")
                    if "/" in ag:
                        won, total = map(int, ag.split("/"))
                        if won < total:
                            v["disputes"].setdefault(f, sorted(
                                {p.get(f) for _, p in cl if p.get(f)}))
            if v["included"] and v["disputes"]:
                v["_jid"] = f"{cid}::{len(disputes)}"
                disputes.append({"_jid": v["_jid"], "item": item, "pair": v})
            voted.append(v)
        comments[cid] = {
            "comment_id": cid, "stratum": item.get("stratum"),
            "neutral": n_neutral > len(pairs_by_model) / 2,
            "pairs": voted,
            "excluded_singletons": sum(1 for v in voted if not v["included"]),
        }

    if emit_disagreements:
        # Targeted-3rd-vote workflow: export the comments where the (usually 2)
        # models tell different stories — a tie in an included cluster, a
        # singleton pair (one model saw it, the other didn't), or a
        # neutral-vs-emotive split (a special case of singletons) — as a
        # sample-format JSONL. Annotate it with the 3rd model into the same run
        # dir, then re-run consensus without --emit-disagreements.
        flagged = [cid for cid, rec in comments.items()
                   if any(v["disputes"] or v["support"] == 1 for v in rec["pairs"])]
        with open(emit_disagreements, "w", encoding="utf-8") as f:
            for cid in flagged:
                f.write(json.dumps(sample[cid], ensure_ascii=False) + "\n")
        print(f"[consensus] {len(flagged)}/{len(comments)} comments disagree "
              f"({100*len(flagged)/len(comments):.0f}%) → {emit_disagreements}", flush=True)
        return emit_disagreements

    if disputes and no_judge:
        print(f"[consensus] skipping judge ({len(disputes)} disputes keep plurality)",
              flush=True)
    elif disputes:
        print(f"[consensus] adjudicating {len(disputes)} disputed pairs "
              f"({'MOCK' if judge_mock else judge_model})", flush=True)
        verdicts = _run_judge(disputes, judge_model, run_dir, judge_mock)
        for d in disputes:
            v = d["pair"]
            for f, val in verdicts.get(d["_jid"], {}).items():
                v[f] = val
                v["agreement"][f] = "judge"
            # unresolved ties keep the provisional plurality; agreement stays "tie"

    out_path = os.path.join(run_dir, "consensus.jsonl")
    n_pairs = n_incl = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for cid, rec in comments.items():
            for v in rec["pairs"]:
                v.pop("_jid", None)
                v.pop("disputes", None)
                n_pairs += 1
                n_incl += v["included"]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    sup = Counter(v["support"] for r in comments.values() for v in r["pairs"])
    ag_e = Counter(v["agreement"].get("emotion") for r in comments.values()
                   for v in r["pairs"] if v["included"])
    ag_c = Counter(v["agreement"].get("cause_category") for r in comments.values()
                   for v in r["pairs"] if v["included"])
    print(f"[consensus] {len(comments)} comments | clusters: {n_pairs} "
          f"(support {dict(sorted(sup.items(), reverse=True))}) | silver pairs: {n_incl} | "
          f"neutral comments: {sum(r['neutral'] for r in comments.values())}", flush=True)
    print(f"[consensus] emotion agreement: {dict(ag_e)} | cause agreement: {dict(ag_c)}",
          flush=True)
    print(f"[consensus] wrote {out_path}", flush=True)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Multi-model consensus + judge")
    ap.add_argument("--run", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--min-support", type=int, default=2)
    ap.add_argument("--judge-model", default=JUDGE_MODEL)
    ap.add_argument("--judge-mock", action="store_true")
    ap.add_argument("--judge-splits", action="store_true",
                    help="also adjudicate 2-1 majorities (more judge calls)")
    ap.add_argument("--no-judge", action="store_true",
                    help="skip adjudication; disputed fields keep plurality")
    ap.add_argument("--emit-disagreements", default=None, metavar="PATH",
                    help="write disagreeing comments as a sample JSONL for a "
                         "targeted 3rd-vote annotation run, then exit")
    args = ap.parse_args()
    build_consensus(args.run, args.sample, args.min_support,
                    args.judge_model, args.judge_mock, args.judge_splits,
                    args.no_judge, args.emit_disagreements)


if __name__ == "__main__":
    main()
