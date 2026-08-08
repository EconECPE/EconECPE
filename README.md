# EconECPE — Span-Level Emotion-Cause Pair Extraction for Financial Social Media

Annotation spec, pipeline, labels and market-analysis code for a span-level,
multi-pair, typed emotion-cause dataset built over the complete 2018–2026
*r/economics* archive, with a cross-domain extension to *r/investing* and
*r/stocks*.

Everything needed to reproduce the study end to end is here. The Reddit
corpora and the span text are deliberately **not** redistributed — see below.

---

## Why the text is not included

Reddit's terms do not permit redistributing comment archives, and every comment
in them carries the username of whoever wrote it. So this repository ships
**identifiers and labels**, and a fetcher that rebuilds the text locally from the
public [Arctic Shift](https://arctic-shift.photon-reddit.com/) archive.

Nothing here redistributes a comment or names a user. Three rules make that
hold rather than merely intend it:

- **No text ships.** Every span is an offset (below). The only prose in the
  repository is the five constructed exemplars in `prompts/` and `spec/`, which
  are written for the spec, not drawn from the corpus.
- **No usernames ship.** `author` is dropped at rehydration
  (`tools/rehydrate.py`) and appears in no released file. The manifests carry
  ids, timestamps, thread links and checksums only.
- **`data/` is deny-by-default in `.gitignore`.** The pipeline regenerates
  text-bearing files beside the released ones — `<model>.labels.jsonl` and
  `consensus.jsonl` hold spans as literal text, the worklist and debate
  transcripts quote whole comments, and the ingester writes the raw archive with
  bodies *and* authors into `data/`. All of it is ignored; only the released
  manifests, `*.offsets.jsonl` and `data/market/*.csv` are allowed back. Adding
  a new released file takes a deliberate `.gitignore` entry.

Retrieval is read-only and rate-limit-respecting: `arctic_shift.py` and
`tools/rehydrate.py` talk to the public archive API with a fixed delay and an
identifying user-agent. Nothing scrapes reddit.com, authenticates as a user, or
works around a rate limit.

That constraint reaches further than it first appears. In span-level ECPE the
labels *are* quotations: `emotion_span` and `cause_span` are verbatim extracts of
the comment being annotated. Measured over the 300-comment gold set, the spans
alone reproduce **35% of an average comment and 100% of twenty of them** — so
shipping the labels as text would have republished a large slice of the corpus
under another name.

Each span therefore ships as an **offset** — a source (`comment`, `parent` or
`post`), a start, an end and an occurrence index — and becomes text again only
once you have fetched the corpus yourself. The encoding is lossless: all 62,807
spans in the released gold, human-gold and silver sets were round-tripped at
build time.

```bash
python tools/rehydrate.py --decode
```

That fetches the annotated comments, their parents and their linked posts,
verifies each against the SHA-256 in `data/manifest/`, and writes text-bearing
labels to `data/rehydrated/`. Expect it to take about twenty minutes for the
20,000-comment silver split; it is resumable, so a killed run costs nothing.

**Do not skip the checksum report it prints.** Reddit content is mutable: a
comment edited or deleted since the archive snapshot comes back different, or
not at all, and a few percent of any id set is unavailable at any given time.
Records that fail verification are still written but flagged `"text_mismatch":
true`, so the reported numbers can be reproduced on the verified subset with the
unverified remainder in plain view rather than silently mixed in.

The aggregate market series in `data/market/` are derived, carry no comment text
and no usernames, and ship as-is — the market results in §6 reproduce without
rehydrating anything.

## What is released

| | |
|---|---|
| `data/manifest/gold_300.manifest.jsonl` | the 300 gold comments: ids, thread links, stratum, checksums |
| `data/manifest/silver_20000.manifest.jsonl` | the 19,998 silver comments, same fields |
| `data/labels/gold/adjudicated.offsets.jsonl` | gold labels, 336 pairs, after cross-model debate adjudication |
| `data/labels/gold/human_gold_100.offsets.jsonl` | the 100-comment hand-curated human benchmark |
| `data/labels/silver/consensus.offsets.jsonl` | multi-LLM consensus silver labels |

### Who labelled what

Read this before reading any agreement number.

The **300-comment gold set is not hand-annotated.** Its two annotator slots, `A`
and `B`, hold the output of two frontier models — `A` is **Claude Sonnet 5**,
`B` is **Gemini 3.5 Flash** — each run independently over all 300 comments and
then filed into the annotation tool under the annotator names. The
disagreements were resolved by `pipeline/debate.py`, in which those same two
models argue each case to a verdict. So `adjudicated.offsets.jsonl` is
**LLM-annotated and LLM-adjudicated end to end**, and the κ / span-F1 it yields
is *model–model agreement, not human inter-annotator agreement*. Two models can
agree and both be wrong, and no LLM-vs-LLM statistic can see it.

The **100-comment set is the human one**: a single human annotator
(`annotator: "A-human"`), working in the same tool over a stratified subset of
the same 300, so every human label has a Claude label and a Gemini label
alongside it. It is the only non-LLM ground truth here, and it is what
`pipeline/human_gold_eval.py` uses to bound the shared model bias the gold set's
own numbers cannot expose.

It is **hand-curated, not blind.** The annotator worked from pooled machine
candidates rather than a blank page: per comment the tool pooled every pair from
all available label sources, deduplicated, and ranked by backing count; the
annotator kept the correct candidates, discarded the rest, rewrote fields where a
span was right but its labels were not, and marked the comment neutral where
nothing held. Every kept pair records its origin in `_accepted_from`, so the
assisted share is checkable directly from the released file. Because the
candidates come from the models, the shared-bias rate this yields is a **floor**
— it counts only readings the annotator rejected outright — and human-vs-model
agreement measures how readily a human *endorses* a model pair. A blind pass
(`run_human_gold.sh` with no `--assist`) is what would turn that floor into an
estimate. `pipeline/human_gold_eval.py` detects the assisted share and says so in
its report.

Provenance is carried in the code, not just here:
`pipeline/gold_report.py` (`ANNOTATOR_PROVENANCE`) and
`gold-tool/server/data.js` name the model behind each slot, and every generated
report prints it.
| `data/market/*.csv` | weekly emotion×cause panels, price series, significance tables |
| `spec/` | the frozen annotation spec and JSON schema |
| `prompts/` | system prompt, few-shot pool, judge prompt |

Corpus-scale label runs (millions of comments) and the distilled model weights
are too large for a repository and are not included; `pipeline/label_corpus.py`
and `pipeline/finetune.py` regenerate both.

## Layout

```
pipeline/         annotation, consensus, distillation, market analysis
  context.py            incremental SQLite index + context assembly
  sample.py             stress-stratified sampling
  prompt.py             system + few-shot prompt construction
  annotate.py           LLM call -> validate -> one repair turn -> merged labels
  validate.py           schema + verbatim-span validation; derives cause_source
  textnorm.py           span normalization (escapes, entities, quotes, case)
  consensus.py          multi-model vote, targeted third vote, judge tie-break
  debate.py             cross-model debate over the gold disagreements (Claude x Gemini)
  gold_report.py        double-coded gold: model-model agreement, worklist, adjudication
  human_gold_eval.py    the 100-comment human benchmark and its scoring
  rescore_gold.py       span matching + the relaxed pair criterion
  finetune.py           LoRA distillation of the generative student
  deberta_train.py      encoder-tagger efficiency variant  (+ _dataset, _eval)
  label_corpus.py       corpus-scale labelling with the distilled student
  market_*.py           panel construction, Granger + walk-forward, robustness
spec/             frozen annotation spec (emotions, cause taxonomy, schema)
prompts/          system prompt, few-shot pool, judge prompt
gold-tool/        the browser annotation tool used for the human gold pass
tools/            corpus rehydration and the span offset codec
data/manifest/    comment ids + checksums for each annotated split
data/labels/      offset-encoded labels
data/market/      aggregate weekly panels and price series
arctic_shift.py   full-history archive ingester (resumable)
```

## Install

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

The annotation pipeline calls models over any OpenAI-compatible chat endpoint —
`pipeline/llm.py` is the whole client, ~200 lines, no vendor SDK beyond `openai`.
Point it somewhere and give it a key:

```bash
export LLM_BASE_URL=https://your-endpoint/v1
export LLM_API_KEY=...
```

The model slugs recorded in the released labels use the `provider/model`
convention of an aggregator; substitute whatever yours expects. Every
model-calling step also takes `--mock`, which runs the full flow offline with no
key and no spend. Distillation additionally needs `requirements-finetune.txt`
and a GPU.

## Reproducing the study

**1 — corpus.** Rehydrate the annotated splits (above). To rebuild the whole
archive rather than the annotated sample, `arctic_shift.py` ingests it from
scratch; it is resumable and checkpoints after every page.

```bash
python tools/rehydrate.py --decode
python -m pipeline.context --build          # index the rehydrated JSONL
```

**2 — silver labels.** Two full votes (DeepSeek-V4-Flash, MiMo-V2.5), a targeted
third on the disagreements (MiniMax-M3), then a judge (DeepSeek-V4-Pro) on the
genuine ties:

```bash
python -m pipeline.annotate  --sample S --run silver --model deepseek/deepseek-v4-flash --reasoning high
python -m pipeline.annotate  --sample S --run silver --model xiaomi/mimo-v2.5 --reasoning medium
python -m pipeline.consensus --run silver --sample S --emit-disagreements D
python -m pipeline.annotate  --sample D --run silver --model minimax/minimax-m3
python -m pipeline.consensus --run silver --sample S     # judge defaults to deepseek/deepseek-v4-pro
```

None of these four shares a family with the two gold annotators, so
silver-vs-gold agreement never scores a model against its own output.

`--mock` runs the whole flow offline and free, which is the cheap way to check a
change before spending anything. Calls are cached on disk by model and prompt
hash, so reruns cost nothing.

**3 — gold set.** Claude Sonnet 5 (slot `A`) and Gemini 3.5 Flash (slot `B`)
each label all 300 held out from the silver pipeline, then debate their
disagreements; the report scores the result. Both slots are models — see
[Who labelled what](#who-labelled-what):

```bash
python -m pipeline.debate --mock      # free dry run
python -m pipeline.debate             # live adjudication
python -m pipeline.gold_report
```

**4 — distillation and corpus labelling.**

```bash
python -m pipeline.finetune                 # LoRA student
python -m pipeline.deberta_train            # encoder tagger variant
python -m pipeline.label_corpus --resume    # corpus-scale pass
```

**5 — market study.** The panels in `data/market/` are the inputs; the analysis
runs on CPU in a few minutes:

```bash
python -m pipeline.market_panel
python -m pipeline.market_signif      # Granger + walk-forward, BH-corrected
python -m pipeline.market_regime      # regime splits
python -m pipeline.market_ticker_analysis
```

## The annotation tool

`gold-tool/` is the browser tool the human gold pass ran in — a Node API over
SQLite plus a React client, with verbatim span capture, an adjudication view and
a per-comment discussion thread. Its deployment files (systemd units, nginx
block, installer) are **not** included: they named the host it ran on and
carried its Basic Auth credentials. Run it locally instead:

```bash
cd gold-tool/server && npm install && node index.js     # API on :10301
cd gold-tool/client && npm install && npm run dev       # client on :5173
```

## Licence

Code and labels: MIT ([`LICENSE`](LICENSE)). The Reddit comments they annotate
are not covered by it and are not redistributed here.
