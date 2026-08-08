"""Two-model AI discussion pass over the gold-set adjudication disagreements.

Claude Sonnet 5 and Gemini 3.5 Flash (the same two frontier models that produced
the A and B gold submissions, `data/labels/gold/*.labels.jsonl`) debate each
A-vs-B disagreement from `pipeline.gold_report`: one opens with its analysis of
where and why the two records diverge, the other answers with its own reading,
and they alternate until they converge or exhaust the turn budget. Every turn
ends with `VERDICT: A|B|NEUTRAL|CUSTOM|UNSURE`; a model that thinks the text +
spec can't settle the call flags a line starting `QUESTION FOR A & B:` instead
of guessing.

`--export-gold` turns the verdicts (plus the auto-agreed comments) into
`data/labels/gold/adjudicated.jsonl` — the released gold set. Note what that
makes it: the two annotators are models, and so is the adjudicator, so the
resulting set is LLM-adjudicated end to end and its agreement statistics are
model-model, not human IAA. `pipeline/human_gold_eval.py` exists to bound that.
The gold-tool Adjudication tab can be used to review or override any verdict.

Output (data/labels/gold/debates/, or --out-dir):
    turns.jsonl     append-only, one row per model turn — the resume unit:
                    a killed run replays finished turns for free
    debates.jsonl   one summary row per finished debate
    report.md       digest grouped by outcome + full transcripts (--report)

    python -m pipeline.debate --mock              # offline smoke test, free
    python -m pipeline.debate --limit 2           # live on the first 2 items
    python -m pipeline.debate                     # live, all disagreements
    python -m pipeline.debate --report            # rebuild report.md only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import config

from .llm import call_chat, make_client

from .gold_report import (GOLD_DIR, _fmt_pairs, compare_annotators,
                          load_coded_from_files, load_double_coded, load_gold_sample)
from .prompt import format_input, system_prompt

DEBATE_DIR = os.path.join(GOLD_DIR, "debates")
DEFAULT_MODELS = ("anthropic/claude-sonnet-5", "google/gemini-3.5-flash")

VERDICT_RE = re.compile(r"^\s*VERDICT:\s*(A|B|NEUTRAL|CUSTOM|UNSURE)\b",
                        re.MULTILINE | re.IGNORECASE)
QUESTION_RE = re.compile(r"^\s*QUESTION FOR A (?:&|AND) B:\s*(.+?)\s*$",
                         re.MULTILINE | re.IGNORECASE)
# a CUSTOM verdict's full pair list: first "- emotion:" up to the VERDICT line
CUSTOM_LIST_RE = re.compile(r"^- emotion:.*?(?=\n\s*VERDICT:|\Z)",
                            re.DOTALL | re.MULTILINE)

CUSTOM_BLOCK_FORMAT = """\
- emotion: <one of the frozen emotions>
  emotion_span: <verbatim substring of the comment>
  cause_span: <verbatim substring of comment/parent/post, or null>
  cause_source: comment|parent|post|null
  cause_category: <one of the frozen categories>
  target_asset: equities|bonds|gold|crypto|USD|housing|none
  intensity: <0..1>
  sarcasm: true|false"""

# NOTE — the two prompts below say "two human annotators". That is not true of
# this dataset: the A and B records are claude-sonnet-5's and gemini-3.5-flash's
# output (see pipeline/gold_report.py ANNOTATOR_PROVENANCE). The wording is kept
# VERBATIM because it is the prompt that actually produced the released
# adjudicated.jsonl — editing it here would silently stop this file reproducing
# the shipped labels. Treat it as a documented limitation: the debaters were led
# to weigh A and B as human readings, which plausibly raised the deference they
# gave both. Anyone re-running from scratch should say "two annotators" and
# re-export.
DEBATER_SYSTEM = """\
You are {me}, one of two AI adjudication assistants. The other is {other}.

Two human annotators — A and B — independently annotated Reddit \
comments from r/economics for emotion-cause pairs, following the annotation \
spec below. On the comment you'll be given, their labels disagree. You and \
{other} will discuss the disagreement in writing, taking turns, to help the \
humans resolve it: either converge on the reading the spec best supports, or \
pin down the precise question only the humans can settle.

<annotation-spec>
{spec}
</annotation-spec>

The spec above defines the label space and span rules; you are NOT annotating \
from scratch — you are adjudicating between two existing human readings (or \
proposing a better third one).

Discussion rules:
- Ground every claim in the comment text and the spec: quote spans verbatim, name the rule you're applying.
- Engage with {other}'s latest arguments directly. Concede a point when it is right; hold your position when it isn't and say exactly why. Do not agree merely to be agreeable.
- If the text + spec genuinely cannot settle a point, add a line starting exactly `QUESTION FOR A & B:` with the precise question they must answer.
- Keep each turn under ~250 words, plus the verdict footer.

End EVERY message with exactly one line stating your current position (it may change between turns):

VERDICT: A | B | NEUTRAL | CUSTOM | UNSURE

A / B = that annotator's record is correct as-is; NEUTRAL = the \
comment has no pairs. If (and only if) your verdict is CUSTOM, put the full \
replacement pair list immediately BEFORE the verdict line, one block per pair, \
in exactly this format:

{custom_format}
"""


THIRD_SYSTEM = """\
You are {me}, a third AI adjudication assistant joining an ongoing discussion \
between {a} and {b}. Two human annotators — A and B — independently \
annotated Reddit comments from r/economics for emotion-cause pairs following \
the annotation spec below; on the comment you'll be given their labels \
disagree, and {a} and {b} have debated it without converging. Your job: read \
their discussion, weigh both positions independently against the text and the \
spec, and push the group toward the best spec-supported resolution — or pin \
down the precise question only the humans can settle.

<annotation-spec>
{spec}
</annotation-spec>

Discussion rules:
- Ground every claim in the comment text and the spec: quote spans verbatim, name the rule you're applying.
- Engage with both discussants' arguments directly. Side with whoever is right on each point; do not split the difference for its own sake.
- If the text + spec genuinely cannot settle a point, add a line starting exactly `QUESTION FOR A & B:` with the precise question they must answer.
- Keep each turn under ~250 words, plus the verdict footer.

End EVERY message with exactly one line stating your current position:

VERDICT: A | B | NEUTRAL | CUSTOM | UNSURE

A / B = that annotator's record is correct as-is; NEUTRAL = the \
comment has no pairs. If (and only if) your verdict is CUSTOM, put the full \
replacement pair list immediately BEFORE the verdict line, one block per pair, \
in exactly this format:

{custom_format}
"""


def _short(model: str) -> str:
    return model.split("/")[-1]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Prompt assembly ──────────────────────────────────────────────────────────

def debater_system(me: str, other: str) -> str:
    return DEBATER_SYSTEM.format(me=_short(me), other=_short(other),
                                 spec=system_prompt(), custom_format=CUSTOM_BLOCK_FORMAT)


def load_prior_model_labels(models: tuple[str, str],
                            gold_dir: str = GOLD_DIR) -> dict[str, dict[str, dict]]:
    """model -> comment_id -> {"neutral", "pairs"} from the per-model gold runs
    (data/labels/gold/<slug>.labels.jsonl), where available."""
    out: dict[str, dict[str, dict]] = {}
    for m in models:
        path = os.path.join(gold_dir, m.replace("/", "_") + ".labels.jsonl")
        if not os.path.exists(path):
            continue
        recs = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                pairs = r.get("pairs") or []
                recs[r["comment_id"]] = {"neutral": not pairs, "pairs": pairs}
        out[m] = recs
    return out


def render_positions(cid: str, coded: dict) -> str:
    return "\n".join(_fmt_pairs("A", coded["A"][cid])
                     + _fmt_pairs("B", coded["B"][cid]))


def _own_block(model: str, cid: str, prior: dict[str, dict[str, dict]]) -> str:
    rec = prior.get(model, {}).get(cid)
    if rec is None:
        return ""
    rendered = "\n".join(_fmt_pairs("YOU (earlier)", rec))
    return ("\nFor reference, your own earlier independent annotation of this "
            "comment (it may inform, but does not bind, your position):\n"
            f"{rendered}\n")


def opening_message(item: dict, cid: str, coded: dict, model: str,
                    prior: dict) -> str:
    return (f"{format_input(item)}\n\n"
            f"The two human annotators' independent labels:\n"
            f"{render_positions(cid, coded)}\n"
            f"{_own_block(model, cid, prior)}\n"
            "Open the discussion: identify exactly where and why A and "
            "B diverge, weigh both readings against the spec, and "
            "propose a resolution.")


def response_message(item: dict, cid: str, coded: dict, model: str,
                     prior: dict, opener: str, opener_content: str) -> str:
    return (f"{format_input(item)}\n\n"
            f"The two human annotators' independent labels:\n"
            f"{render_positions(cid, coded)}\n"
            f"{_own_block(model, cid, prior)}\n"
            f"{_short(opener)} has opened the discussion:\n\n{opener_content}\n\n"
            f"Give your own analysis: where do you agree or disagree with "
            f"{_short(opener)}, and what resolution do you propose?")


def joining_message(item: dict, cid: str, coded: dict, model: str, prior: dict,
                    pair: tuple[str, str], transcript: str) -> str:
    return (f"{format_input(item)}\n\n"
            f"The two human annotators' independent labels:\n"
            f"{render_positions(cid, coded)}\n"
            f"{_own_block(model, cid, prior)}\n"
            f"The discussion between {_short(pair[0])} and {_short(pair[1])} "
            f"so far:\n\n{transcript}\n\n"
            "They have not converged. Give your independent analysis of both "
            "positions and state which resolution the spec best supports.")


# ── One debate ───────────────────────────────────────────────────────────────

def parse_verdict(content: str) -> str:
    ms = VERDICT_RE.findall(content or "")
    return ms[-1].upper() if ms else "UNSURE"


def parse_custom_list(content: str) -> str | None:
    """Whitespace-normalized custom pair list from a turn, if any."""
    ms = CUSTOM_LIST_RE.findall(content or "")
    return re.sub(r"\s+", " ", ms[-1]).strip() if ms else None


def _opener_idx(cid: str) -> int:
    """Which model opens is a stable hash of the comment id (≈50/50 overall),
    so partial runs under any --ids/--limit filter resume consistently."""
    return int(hashlib.md5(cid.encode()).hexdigest(), 16) >> 8 & 1


def _agreed(verdicts: dict, customs: dict, participants) -> bool:
    """Unanimous verdict among participants; for CUSTOM the proposed pair
    lists themselves must match, not just the labels."""
    vs = {verdicts[p] for p in participants}
    if len(vs) != 1 or vs == {"UNSURE"}:
        return False
    if vs == {"CUSTOM"}:
        blocks = {customs[p] for p in participants}
        return None not in blocks and len(blocks) == 1
    return True


def _majority(verdicts: dict, customs: dict, participants) -> str | None:
    """Verdict held by >=2 participants (CUSTOMs count together only when
    their pair lists match), or None."""
    keys = [(verdicts[p], customs[p] if verdicts[p] == "CUSTOM" else None)
            for p in participants]
    (top, _), n = max(((k, keys.count(k)) for k in keys), key=lambda x: x[1])
    return top if n >= 2 and top != "UNSURE" else None


def run_debate(cid: str, item: dict, coded: dict, prior: dict,
               models: tuple[str, str], chat_fn, max_turns: int,
               prior_turns: list[dict], record_turn,
               escalate_model: str | None = None, max_ext_turns: int = 6) -> dict:
    """Alternate turns between the two models until their verdicts agree or
    max_turns is hit; if still split and `escalate_model` is set, that model
    joins as a third discussant (reading the full transcript) and the three
    cycle for up to `max_ext_turns` more turns. `prior_turns` (from
    turns.jsonl) are replayed without API calls; `record_turn(row)` persists
    each new turn as it completes."""
    oi = _opener_idx(cid)
    order = (models[oi], models[1 - oi])
    ext_order = (escalate_model, order[0], order[1]) if escalate_model else None

    def expected_speaker(t: int) -> str | None:
        if t < max_turns:
            return order[t % 2]
        return ext_order[(t - max_turns) % 3] if ext_order else None

    keep = 0
    for t, r in enumerate(prior_turns):
        if r["model"] != expected_speaker(t):
            break
        keep += 1
    if keep < len(prior_turns):
        # e.g. the escalation model changed: keep the still-valid prefix (the
        # whole main phase) and redo only the extension
        print(f"[debate] {cid}: speaker order changed — keeping {keep} stored "
              f"turns, redoing {len(prior_turns) - keep}")
        prior_turns = prior_turns[:keep]

    histories = {m: [{"role": "system", "content": debater_system(m, o)}]
                 for m, o in (order, order[::-1])}
    participants = list(models)
    verdicts: dict[str, str] = {m: "UNSURE" for m in models}
    customs: dict[str, str | None] = {m: None for m in models}
    all_turns: list[tuple[str, str]] = []
    questions: list[str] = []
    state = {"cost": 0.0, "n_turns": 0}

    def take_turn(t: int, speaker: str, final_nudge: bool) -> str:
        if final_nudge:
            histories[speaker][-1]["content"] += (
                "\n\n(This is the final exchange — state your final position "
                "and commit to a verdict.)")
        if t < len(prior_turns):  # resumed turn: replay, don't re-call
            content = prior_turns[t]["content"]
        else:
            content, usage = chat_fn(speaker, histories[speaker], cid=cid, turn=t)
            record_turn({"comment_id": cid, "turn": t, "model": speaker,
                         "content": content, "cost_usd": usage.get("cost"),
                         "ts": _now()})
            state["cost"] += usage.get("cost") or 0.0
        histories[speaker].append({"role": "assistant", "content": content})
        all_turns.append((speaker, content))
        verdicts[speaker] = parse_verdict(content)
        block = parse_custom_list(content)
        if block:  # models may re-state CUSTOM without repeating the list
            customs[speaker] = block
        for q in QUESTION_RE.findall(content or ""):
            if q not in questions:
                questions.append(q)
        state["n_turns"] = t + 1
        return content

    outcome = "split"
    for t in range(max_turns):
        speaker, listener = order[t % 2], order[(t + 1) % 2]
        if t == 0:
            histories[speaker].append(
                {"role": "user",
                 "content": opening_message(item, cid, coded, speaker, prior)})
        content = take_turn(t, speaker, final_nudge=t >= max_turns - 2)
        if t == 0:
            histories[listener].append(
                {"role": "user",
                 "content": response_message(item, cid, coded, listener, prior,
                                             speaker, content)})
        else:
            histories[listener].append(
                {"role": "user",
                 "content": f"{_short(speaker)} replied:\n\n{content}"})
        if t >= 1 and _agreed(verdicts, customs, participants):
            outcome = "consensus"
            break

    escalated = False
    if outcome == "split" and escalate_model:
        escalated = True
        participants.append(escalate_model)
        verdicts[escalate_model], customs[escalate_model] = "UNSURE", None
        transcript = "\n\n".join(
            f"--- {_short(m)} (turn {i + 1}):\n{c}"
            for i, (m, c) in enumerate(all_turns))
        histories[escalate_model] = [
            {"role": "system", "content": THIRD_SYSTEM.format(
                me=_short(escalate_model), a=_short(order[0]), b=_short(order[1]),
                spec=system_prompt(), custom_format=CUSTOM_BLOCK_FORMAT)},
            {"role": "user", "content": joining_message(
                item, cid, coded, escalate_model, prior, order, transcript)},
        ]
        base = state["n_turns"]
        for t_ext in range(max_ext_turns):
            t = base + t_ext
            speaker = ext_order[t_ext % 3]
            content = take_turn(t, speaker, final_nudge=t_ext >= max_ext_turns - 3)
            for other in participants:
                if other == speaker:
                    continue
                joined = ("" if t_ext > 0 or other == escalate_model else
                          f"A third adjudicator, {_short(escalate_model)}, has "
                          "joined the discussion to help break the deadlock. ")
                histories[other].append(
                    {"role": "user",
                     "content": f"{joined}{_short(speaker)} replied:\n\n{content}"})
            if t_ext >= 1 and _agreed(verdicts, customs, participants):
                outcome = "consensus"
                break

    return {"comment_id": cid, "models": list(models), "opener": order[0],
            "outcome": outcome,
            "consensus": verdicts[participants[0]] if outcome == "consensus" else None,
            "verdicts": {m: verdicts[m] for m in participants},
            "majority": (None if outcome == "consensus" else
                         _majority(verdicts, customs, participants)),
            "escalated": escalated,
            "escalate_model": escalate_model if escalated else None,
            "n_turns": state["n_turns"], "questions": questions,
            "cost_usd": round(state["cost"], 6), "ts": _now()}


# ── Chat backends ────────────────────────────────────────────────────────────

def live_chat_factory(client, max_tokens: int):
    def chat(model, messages, *, cid, turn):
        # Reasoning models burn output tokens on hidden thinking first, so a
        # tight budget yields finish_reason="length" with empty OR truncated
        # content — the whole budget can go before a single visible token. Any
        # "length" finish gets one retry with double the room; a still-truncated
        # non-empty retry is
        # kept (best effort) rather than paid for a third time.
        tokens, content, usage, finish = max_tokens, None, None, None
        for attempt in range(2):
            content, usage, finish = call_chat(
                client, messages, model, max_tokens=tokens, temperature=0.2,
                timeout=180.0, extra_body={"usage": {"include": True}})
            if finish != "length":
                break
            if attempt == 0:
                print(f"[debate] {cid}#t{turn} {model}: truncated at "
                      f"{tokens} tokens — retrying with {tokens * 2}", flush=True)
                tokens *= 2
        if not content:
            raise RuntimeError(f"empty model output (finish_reason={finish})")
        return content, usage
    return chat


_MOCK_CUSTOM = """\
- emotion: fear_anxiety
  emotion_span: mock span
  cause_span: null
  cause_source: null
  cause_category: other
  target_asset: none
  intensity: 0.5
  sarcasm: false"""


def mock_chat(model, messages, *, cid, turn):
    """Deterministic canned turns, no network. md5(cid)%4 picks a scenario so a
    sample of comments exercises consensus, split, and question+custom paths.
    Turns past the default 6 (the escalation extension) all vote A so the
    split scenario exercises third-model convergence."""
    mode = int(hashlib.md5(cid.encode()).hexdigest(), 16) % 4
    me = _short(model)
    if turn >= 6:
        content = (f"(mock) {me} joins/continues the extension on {cid}, "
                   f"turn {turn + 1}.\n\nVERDICT: A")
        return content, {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}
    if mode in (0, 1):
        v = "A" if mode == 0 else "B"
        content = (f"(mock) {me}, turn {turn + 1} on {cid}: the {v.title()} "
                   f"reading fits the spec better.\n\nVERDICT: {v}")
    elif mode == 2:  # opener and responder never converge → split at max_turns
        v = "A" if turn % 2 == 0 else "B"
        content = f"(mock) {me} holds its position on {cid}, turn {turn + 1}.\n\nVERDICT: {v}"
    elif turn == 0:  # mode 3: question first, then converge on a custom pair
        content = ("(mock) The emotion span boundary is genuinely ambiguous here.\n"
                   "QUESTION FOR A & B: is the second clause part of "
                   "the emotion span?\n\nVERDICT: UNSURE")
    else:
        content = (f"(mock) {me} proposes a merged pair.\n{_MOCK_CUSTOM}\n\n"
                   "VERDICT: CUSTOM")
    return content, {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}


# ── Persistence / resume ─────────────────────────────────────────────────────

class _JsonlWriter:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()

    def append(self, row: dict):
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_state(out_dir: str, models: tuple[str, str]):
    """(done: cid -> latest summary row, partial: cid -> ordered turn rows).
    Partial turns whose recorded models don't match the current pair are
    dropped (the debate restarts) — histories wouldn't line up."""
    done = {r["comment_id"]: r for r in _read_jsonl(os.path.join(out_dir, "debates.jsonl"))}
    by_turn: dict[str, dict[int, dict]] = {}
    for r in _read_jsonl(os.path.join(out_dir, "turns.jsonl")):
        if r["comment_id"] not in done:  # duplicate (cid, turn): latest run wins
            by_turn.setdefault(r["comment_id"], {})[r["turn"]] = r
    partial: dict[str, list[dict]] = {}
    for cid, rows in by_turn.items():
        turns = [rows[t] for t in sorted(rows)]
        # contiguity only — speaker-order validation (and prefix-truncation
        # when e.g. the escalation model changed) happens in run_debate, which
        # knows the expected order; rejecting here would force a full restart
        if [r["turn"] for r in turns] != list(range(len(turns))):
            print(f"[debate] {cid}: stored turns not contiguous — restarting it")
        else:
            partial[cid] = turns
    return done, partial


# ── Report ───────────────────────────────────────────────────────────────────

def build_report(out_dir: str = DEBATE_DIR, coded: dict | None = None,
                 sample: dict | None = None) -> str:
    debates = {r["comment_id"]: r for r in _read_jsonl(os.path.join(out_dir, "debates.jsonl"))}
    by_turn: dict[str, dict[int, dict]] = {}
    for r in _read_jsonl(os.path.join(out_dir, "turns.jsonl")):
        by_turn.setdefault(r["comment_id"], {})[r["turn"]] = r  # latest run wins
    turns = {cid: list(rows.values()) for cid, rows in by_turn.items()}
    if not debates:
        raise SystemExit(f"no finished debates in {out_dir} — run the pass first")

    if coded is None:
        coded = load_double_coded()
    if sample is None:
        sample = load_gold_sample()
    stats = compare_annotators(coded["A"], coded["B"])
    ordered = [c for c in stats["disagreements"] if c in debates]
    models = debates[ordered[0]]["models"]
    total_cost = sum(d["cost_usd"] or 0 for d in debates.values())

    def bucket(d):
        return d["consensus"] if d["outcome"] == "consensus" else "split"

    counts = {}
    for d in debates.values():
        counts[bucket(d)] = counts.get(bucket(d), 0) + 1

    lines = [
        "# AI discussion pass — gold adjudication disagreements", "",
        f"Models: **{models[0]}** vs **{models[1]}** · generated {_now()[:19]}Z",
        f"Debates finished: **{len(ordered)}/{len(stats['disagreements'])}** "
        f"disagreements · total cost ${total_cost:.2f}", "",
        "Outcomes: " + " · ".join(f"{k} ×{v}" for k, v in sorted(counts.items())), "",
        "> Advisory second opinions for the human discussion pass in gold-tool's",
        "> Adjudication tab — the final call on every comment stays with A",
        "> and B. CUSTOM consensus still needs a human read: the two",
        "> models' proposed pair lists may differ.", "",
    ]

    questions = [(c, q) for c in ordered for q in debates[c]["questions"]]
    if questions:
        lines += ["## Questions raised for A & B", ""]
        lines += [f"- `{c}`: {q}" for c, q in questions] + [""]

    lines += ["## Verdict index", ""]
    for key in ("A", "B", "NEUTRAL", "CUSTOM", "split"):
        cids = [c for c in ordered if bucket(debates[c]) == key]
        if not cids:
            continue
        title = f"consensus: {key}" if key != "split" else "split — no consensus"
        lines += [f"### {title} ({len(cids)})", ""]
        for c in cids:
            d = debates[c]
            extra = ("" if key != "split" else " — " + " vs ".join(
                f"{_short(m)}: {v}" for m, v in d["verdicts"].items()))
            if key == "split" and set(d["verdicts"].values()) == {"CUSTOM"}:
                extra += " (all CUSTOM but different pair lists)"
            if key == "split" and d.get("majority"):
                extra += f" — 2/3 majority: {d['majority']}"
            if d.get("escalated"):
                extra += f" · escalated to {_short(d['escalate_model'])}"
            lines.append(f"- `{c}` ({d['n_turns']} turns){extra}")
        lines.append("")

    lines += ["## Transcripts", ""]
    for c in ordered:
        d, item = debates[c], sample[c]
        head = d["consensus"] if d["outcome"] == "consensus" else "split"
        lines += ["---", f"### `{c}` — {head}", "",
                  f"**POST**: {item.get('post_title')}"]
        if item.get("parent_body"):
            lines.append(f"**PARENT**: {item['parent_body']}")
        lines += [f"**COMMENT**: {item['body']}", "",
                  render_positions(c, coded), ""]
        for r in sorted(turns.get(c, []), key=lambda r: r["turn"]):
            if r["turn"] >= d["n_turns"]:  # stale rows from a superseded run
                continue
            lines.append(f"**{_short(r['model'])} (turn {r['turn'] + 1}):**")
            lines += ["> " + l for l in r["content"].splitlines()] + [""]

    out = os.path.join(out_dir, "report.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[debate] wrote {out} ({len(ordered)} debates)", flush=True)
    return out


# ── Export the adjudicated gold set ──────────────────────────────────────────

def _final_blocks(rows: list[dict], n_turns: int) -> dict[str, tuple[int, str, str]]:
    """model -> (turn, raw_block, normalized_block) from each model's last
    block-bearing turn within the debate's final n_turns."""
    out: dict[str, tuple[int, str, str]] = {}
    for r in sorted(rows, key=lambda r: r["turn"]):
        if r["turn"] >= n_turns:
            continue
        ms = CUSTOM_LIST_RE.findall(r["content"] or "")
        if ms:
            out[r["model"]] = (r["turn"], ms[-1].strip(),
                               re.sub(r"\s+", " ", ms[-1]).strip())
    return out


def export_gold(out_dir: str = DEBATE_DIR, coded: dict | None = None,
                sample: dict | None = None) -> str:
    """Debate verdicts + the auto-agreed comments → the adjudicated gold set
    (data/labels/gold/adjudicated.jsonl, the file --score-silver expects).

    Consensus verdicts are final; residual splits fall back to the recorded
    2/3 majority and are marked provisional=true for the human pass. CUSTOM
    pair lists are parsed from the transcripts and re-validated against the
    spec's schema + verbatim-span rules (cause_source re-derived)."""
    from .gold_report import _parse_custom_pairs
    from .validate import validate_annotation

    debates = {r["comment_id"]: r for r in _read_jsonl(os.path.join(out_dir, "debates.jsonl"))}
    by_turn: dict[str, dict[int, dict]] = {}
    for r in _read_jsonl(os.path.join(out_dir, "turns.jsonl")):
        by_turn.setdefault(r["comment_id"], {})[r["turn"]] = r

    if coded is None:
        coded = load_double_coded()
    if sample is None:
        sample = load_gold_sample()
    stats = compare_annotators(coded["A"], coded["B"])
    missing = [c for c in stats["disagreements"] if c not in debates]
    if missing:
        raise SystemExit(f"{len(missing)} disagreements have no finished debate: {missing[:5]}")

    n_prov = n_errors = n_unresolved = 0
    records: dict[str, dict] = {}
    for cid in stats["ids"]:
        if cid not in stats["disagreements"]:
            records[cid] = {**coded["A"][cid], "final": "AGREED"}
            continue
        d = debates[cid]
        verdict = d["consensus"] or d.get("majority")
        if verdict is None:
            # Genuine 3-way split with no majority (each participant a different
            # verdict). Don't fabricate a label or abort the whole export: flag
            # it neutral+unresolved for a human call. (Can't arise in the
            # economics run — its splits all carried a 2/3 majority.)
            n_unresolved += 1
            print(f"[export] {cid}: no consensus and no majority "
                  f"({d['verdicts']}) — flagged unresolved for a human call",
                  flush=True)
            records[cid] = {"neutral": True, "pairs": [], "final": "SPLIT",
                            "provisional": True, "unresolved": True,
                            "verdicts": d["verdicts"]}
            continue
        provisional = d["consensus"] is None
        n_prov += provisional
        if verdict in ("A", "B"):
            rec = dict(coded["A" if verdict == "A" else "B"][cid])
        elif verdict == "NEUTRAL":
            rec = {"neutral": True, "pairs": []}
        else:  # CUSTOM: the agreed (or majority-shared) pair list from the transcript
            blocks = _final_blocks(list(by_turn[cid].values()), d["n_turns"])
            names = [m for m, v in d["verdicts"].items() if v == "CUSTOM" and m in blocks]
            groups: dict[str, list[str]] = {}
            for m in names:
                groups.setdefault(blocks[m][2], []).append(m)
            group = max(groups.values(), key=len)
            if len(group) < 2:
                raise SystemExit(f"{cid}: CUSTOM verdict but no two matching pair lists")
            raw = max((blocks[m] for m in group), key=lambda b: b[0])[1]
            pairs = _parse_custom_pairs(raw.splitlines())
            ann, errors, _ = validate_annotation({"pairs": pairs}, sample[cid])
            if errors:
                n_errors += 1
                print(f"[export] {cid}: CUSTOM list fails validation — needs a human fix:")
                for e in errors:
                    print(f"    - {e}")
            rec = {"neutral": not pairs, "pairs": (ann or {}).get("pairs", pairs),
                   **({"validation_errors": errors} if errors else {})}
        rec["final"] = verdict
        if provisional:
            rec["provisional"] = True
        records[cid] = rec

    out = os.path.join(os.path.dirname(out_dir.rstrip("/")), "adjudicated.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for cid, rec in records.items():
            f.write(json.dumps({"comment_id": cid, **rec}, ensure_ascii=False) + "\n")
    print(f"[export] wrote {out}: {len(records)} comments "
          f"({len(stats['disagreements'])} adjudicated, {n_prov} provisional "
          f"majority-based, {n_unresolved} unresolved 3-way splits, "
          f"{n_errors} with validation errors)", flush=True)
    return out


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true",
                    help="canned offline turns, no API key or spend")
    ap.add_argument("--limit", type=int, help="only the first N disagreements")
    ap.add_argument("--ids", help="comma-separated comment ids to debate")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS),
                    help="two model slugs, comma-separated")
    ap.add_argument("--max-turns", type=int, default=6,
                    help="total turns per debate incl. both models (default 6)")
    ap.add_argument("--escalate-model", default="x-ai/grok-4.3",
                    help="third model that joins a debate still split after "
                         "--max-turns ('' disables). A third family, outside "
                         "both debaters', so escalation is not a tie-break by "
                         "one of the two priors already in the room")
    ap.add_argument("--max-ext-turns", type=int, default=6,
                    help="extra 3-way turns after escalation (default 6 = two "
                         "full cycles)")
    ap.add_argument("--max-tokens", type=int, default=4000,
                    help="per-turn output budget incl. hidden reasoning; a "
                         "'length' finish retries once at double (default 4000)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out-dir", help=f"default {DEBATE_DIR} (…/debates_mock with --mock)")
    # Per-subreddit gold transfer check: the two 'annotators' (A/B
    # slots) are two frontier models' .labels.jsonl instead of the economics
    # gold-tool sqlite. --sample must be that sub's gold sample JSONL. Prior
    # per-model labels are read from the parent of --out-dir.
    ap.add_argument("--sample", help="gold sample JSONL (default economics gold_300)")
    ap.add_argument("--labels-a", help="annotator-A .labels.jsonl (→ A slot)")
    ap.add_argument("--labels-b", help="annotator-B .labels.jsonl (→ B slot)")
    ap.add_argument("--report", action="store_true",
                    help="rebuild report.md from existing output, no API calls")
    ap.add_argument("--export-gold", action="store_true",
                    help="write data/labels/gold/adjudicated.jsonl from the "
                         "debate verdicts (+ auto-agreed comments), no API calls")
    args = ap.parse_args()

    models = tuple(m.strip() for m in args.models.split(",") if m.strip())
    if len(models) != 2:
        raise SystemExit("--models needs exactly two comma-separated slugs")
    out_dir = args.out_dir or (DEBATE_DIR + "_mock" if args.mock else DEBATE_DIR)
    os.makedirs(out_dir, exist_ok=True)

    if bool(args.labels_a) != bool(args.labels_b):
        raise SystemExit("--labels-a and --labels-b must be given together")
    if args.labels_a:
        coded = load_coded_from_files(args.labels_a, args.labels_b)
        # per-model gold labels live alongside the debate dir, in its parent
        gold_dir = os.path.dirname(out_dir.rstrip("/"))
    else:
        coded = load_double_coded()
        gold_dir = GOLD_DIR
    sample = load_gold_sample(args.sample) if args.sample else load_gold_sample()

    if args.report:
        build_report(out_dir, coded, sample)
        return
    if args.export_gold:
        export_gold(out_dir, coded, sample)
        return

    stats = compare_annotators(coded["A"], coded["B"])
    prior = load_prior_model_labels(models, gold_dir)
    ids = stats["disagreements"]
    if args.ids:
        want = {c.strip() for c in args.ids.split(",")}
        missing = want - set(ids)
        if missing:
            raise SystemExit(f"not disagreements (or unknown): {sorted(missing)}")
        ids = [c for c in ids if c in want]
    if args.limit:
        ids = ids[:args.limit]

    escalate = args.escalate_model.strip() or None
    known_models = models + ((escalate,) if escalate else ())
    done, partial = load_state(out_dir, known_models)
    todo = [c for c in ids if c not in done]
    print(f"[debate] {len(ids)} disagreements: {len(ids) - len(todo)} already "
          f"done, {len(todo)} to run ({sum(1 for c in todo if c in partial)} "
          f"resuming mid-debate) → {out_dir}", flush=True)
    if not todo:
        build_report(out_dir, coded, sample)
        return

    chat_fn = mock_chat if args.mock else live_chat_factory(make_client(),
                                                            args.max_tokens)

    turns_w = _JsonlWriter(os.path.join(out_dir, "turns.jsonl"))
    debates_w = _JsonlWriter(os.path.join(out_dir, "debates.jsonl"))

    def run_one(cid):
        summary = run_debate(cid, sample[cid], coded, prior, models,
                             chat_fn=chat_fn, max_turns=args.max_turns,
                             prior_turns=partial.get(cid, []),
                             record_turn=turns_w.append,
                             escalate_model=escalate,
                             max_ext_turns=args.max_ext_turns)
        debates_w.append(summary)
        return summary

    finished = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, c): c for c in todo}
        for fut in as_completed(futures):
            cid = futures[fut]
            try:
                s = fut.result()
                finished += 1
                tag = s["consensus"] or (f"majority {s['majority']}"
                                         if s.get("majority") else "split")
                esc = f" +{_short(s['escalate_model'])}" if s.get("escalated") else ""
                # Only endpoints that report a per-call price fill cost_usd.
                cost = f", ${s['cost_usd']:.4f}" if s.get("cost_usd") else ""
                print(f"[debate] {cid}: {s['outcome']} ({tag}) in {s['n_turns']} "
                      f"turns{esc}{cost}  [{finished}/{len(todo)}]", flush=True)
            except Exception as e:
                print(f"[debate] {cid} failed ({type(e).__name__}: {e}) — "
                      "its finished turns are saved; rerun to resume", flush=True)

    if finished:
        build_report(out_dir, coded, sample)


if __name__ == "__main__":
    main()
