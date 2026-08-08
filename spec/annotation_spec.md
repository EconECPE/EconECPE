# EconECPE annotation specification — v0.3 (frozen 2026-07-02, post-pilot-audit)

Emotion-Cause Pair Extraction (ECPE), relaxed and finance-tuned, on r/economics comments.
This document is the single source of truth for LLM annotators (Phases 1–2), human
annotators (gold set), and the paper appendix. The machine-checkable output contract is
[`schema.json`](schema.json) (same version). Taxonomy grounding and citations:
[`../research/emotion-taxonomy-litreview.md`](../research/emotion-taxonomy-litreview.md).

**Status**: v0.2 is frozen for the pilot (~250 comments, stress-stratified — §10). The
design flags raised in v0.1 were resolved 2026-07-02 (§9); what remains open is only the
**pre-committed gate criteria in §10**, evaluated on pilot evidence. After the gate, the
spec is frozen for the consensus run — changes then invalidate labels; don't make them.

---

## 1. Task

Given one Reddit **comment** plus its **context**, extract every *(emotion, cause)* pair the
**comment's author** expresses about an **economic topic**. Context = the post title
(+ selftext if any) and, for replies, the immediate parent comment. Context explains the
comment; only the comment author's emotions are annotated.

Output: one JSON object per comment, `{"pairs": [...]}` per `schema.json`.
A comment expressing no economically-caused emotion → `{"pairs": []}` (this is the
**neutral** case; neutral is not a pair label).

This is a *relaxed* ECPE: free spans instead of clause indices (Reddit prose has no clause
segmentation), and every pair is **typed** with `emotion`, `cause_category`, `target_asset`,
`intensity` — the typed-pair/triplet convention (ECQED; SemEval-2024 Task 3).

## 2. Emotion taxonomy (7 pair labels + implicit neutral)

Grounded in StockEmotions-12, GoEmotions, Ekman-6, WASSA practice (see lit review §3).

| Label | Definition — author expresses… | Typical cues | Not this label |
|---|---|---|---|
| `fear_anxiety` | worry, dread, panic about a bad outcome | "terrified", "scares me", "brace for", "this will blow up" | resigned gloom without arousal → `pessimism_despair` |
| `pessimism_despair` | low-arousal negative outlook, resignation, hopelessness | "we're cooked", "sobering", "no way out of this", "it'll never recover" | active worry → `fear_anxiety` |
| `anger_disgust` | indignation, outrage, contempt, moral disgust | "criminal", "disgusting", "they're robbing us", rants at institutions | disagreement without heat → no pair |
| `optimism_confidence` | calm positive outlook, trust that things will work out | "soft landing is likely", "the economy is resilient", "good sign" | high-arousal hype → `excitement_euphoria` |
| `excitement_euphoria` | high-arousal enthusiasm, thrill, euphoria | "to the moon", "all in", "incredible numbers!" | measured approval → `optimism_confidence` |
| `confusion_uncertainty` | not knowing what to think/expect; epistemic unease | "who knows anymore", "makes no sense", "can't tell if…" | fear of a *specific* bad outcome → `fear_anxiety` |
| `surprise` | reaction to the unexpected (either valence) | "wow, didn't see that coming", "damn", upset-expectation framing | — |

Boundary rules:
- **Arousal splits the negatives and positives**: fear vs pessimism, excitement vs optimism.
  When torn, ask: is the author *activated* (fear/excitement) or *settled* (pessimism/optimism)?
- **Greed is not a label**. Annotate the surface emotion (usually `excitement_euphoria` or
  `optimism_confidence`); greed is recovered post-hoc as excitement/optimism whose cause is
  the author's own expected gain.
- Multiple emotions in one comment → multiple pairs. Same span may anchor several pairs.

Fixed mappings (report-time comparability; never annotated directly):

| EconECPE | ← StockEmotions-12 | → Ekman-6(+neutral) |
|---|---|---|
| fear_anxiety | anxiety, panic | fear |
| pessimism_despair | depression | sadness |
| anger_disgust | anger, disgust | anger, disgust |
| optimism_confidence | optimism, belief | (joy) |
| excitement_euphoria | excitement, amusement | joy |
| confusion_uncertainty | confusion, ambiguous | — (extension) |
| surprise | surprise | surprise |
| neutral (no pairs) | — | neutral |

## 3. Cause taxonomy (`cause_category`)

The category classifies the **cause**, not the emotion, and enables aggregation for the
market study. Pick the category of the *stated* cause, not of the wider thread topic.

| Category | Covers | Example causes |
|---|---|---|
| `inflation` | price levels, CPI, cost of living, purchasing power | "groceries up 20%", "CPI print" |
| `monetary_policy` | Fed/central banks, rates, QE/QT | "the Fed keeps hiking" |
| `employment` | jobs, layoffs, wages, labor market | "tech layoffs everywhere" |
| `growth_gdp` | GDP, recession/expansion, productivity | "GDP contracted again" |
| `fiscal_taxes` | government spending, taxes, deficits, debt ceiling | "the deficit is exploding" |
| `trade_tariffs` | tariffs, trade wars, imports/exports, supply chains | "new tariffs on chips" |
| `energy` | oil, gas, electricity prices, energy policy | "OPEC cut production" |
| `housing` | home prices, rents, mortgages, construction | "rates killed affordability" |
| `geopolitics` | wars, elections, sanctions, political instability | "if the war spreads" |
| `markets_themselves` | market moves/valuations as the cause (selloffs, bubbles, VIX) | "the crash last week" |
| `crypto` | crypto assets/industry events | "the exchange collapsed" |
| `corporate_earnings` | company results, guidance, corporate news | "their guidance was awful" |
| `inequality_distribution` | wealth/income inequality, distribution between classes, capital vs labor | "the rich keep marrying the rich", "billionaires taxed like plumbers" |
| `personal_finance` | the author's own money situation: savings, personal debt, insurance, own career | "insurance is bleeding me dry", "I'll never get out of debt" |
| `interpersonal` | affect directed at **discussion participants**: other users, mods, or the subreddit itself | "you're an idiot", "this sub has gone downhill" |
| `other` | identifiable *economic* cause outside the above (AI disruption, demographics, healthcare costs…) | |
| `unclear` | no identifiable cause — **only** with `cause_span: null` | |

- Cause crosses categories → pick the **most proximate** stated cause ("Fed hiked because of
  inflation, I'm scared" → the hiking: `monetary_policy`).
- **Interpersonal affect is annotated, not dropped** (decided 2026-07-02, supersedes the v0.1
  exclusion): it gets `cause_category: "interpersonal"` and is filtered out of the market
  aggregation downstream. Comments mixing a jab with an economic stance yield **both** pairs.
  Rationale: reversible, measurable, and the released dataset stays complete.
- **Interpersonal is only for discussion participants** (pilot-audit fix, 2026-07-02): affect
  aimed at politicians, officials, companies, or any third party outside the thread takes the
  category of the underlying policy/topic ("anger at the GOP's spending" → `fiscal_taxes`,
  "anger at Trump's pandemic statement" → the relevant policy category), even when it surfaces
  mid-argument with another user.

## 4. Spans

- `emotion_span`: **verbatim** substring of the **comment** — minimal but complete
  (the emotive phrase, not the whole sentence unless the whole sentence carries it).
- `cause_span`: verbatim substring of the comment **or** its context; set `cause_source`
  to `comment` / `post` / `parent` accordingly. (`cause_source` exists because causes often
  live in the post title, and consensus span-matching needs to know which text a span came
  from.)
- **`cause_source` is validator-enforced, not trusted** (decided 2026-07-02): annotators/LLMs
  emit it, but the harness searches all three texts for the span (normalized: markdown
  escapes, smart quotes, case-insensitive) and silently corrects a wrong source. Span found
  in multiple sources → precedence `comment > parent > post`. Span found **nowhere** →
  paraphrase → repair loop (LLMs) / flagged for correction (humans).
- Spans are copied exactly (case, typos and all) — silver/gold agreement is computed by
  token-overlap/IoU per source text, so paraphrase = mismatch.
- No identifiable cause → `cause_span: null`, `cause_source: null`,
  `cause_category: "unclear"`. Use sparingly; most expressed emotion in r/economics has a
  stated cause.

## 5. `target_asset`

The asset class the emotion is *about*, when the author connects the emotion to an asset:
`equities | bonds | gold | crypto | USD | housing | none`. Macro-only feelings with no asset
tie → `none`. Housing-as-investment and housing-as-shelter both map to `housing`.

## 6. `intensity`

Anchored scale (intermediate values allowed): **0.25** mild/hedged ("a bit worried") ·
**0.5** clear, moderate ("this worries me") · **0.75** strong ("I'm terrified") ·
**1.0** extreme (all-caps panic, catastrophizing). Sarcasm does not cap intensity.

## 7. Edge cases

- **Sarcasm/irony** (endemic here): label the **intended** emotion, set `sarcasm: true`.
  "Great, another rate hike, exactly what we needed 🙄" → `anger_disgust`, sarcasm true.
- **Quoted/reported emotion** (`> quotes`, "my dad is panicking", "the article says people
  fear…"): the *author's* emotion only. Quoting someone to mock them → annotate the author's
  (often `anger_disgust`, sarcastic); the quoted person's fear is not annotated.
- **Questions & hypotheticals**: genuine information-seeking → no pair (or
  `confusion_uncertainty` if bewilderment is expressed). Rhetorical questions carrying affect
  ("Can this be the first step to eliminating the deduction?") → annotate the affect.
- **Rhetorical questions are not confusion** (pilot-audit fix, 2026-07-02):
  `confusion_uncertainty` requires the author to genuinely not know ("I can't tell from the
  way it was written"). A question used to dismiss or attack a position the author then argues
  against ("Does anyone really think X?" followed by why X is wrong) carries the underlying
  emotion — often `anger_disgust` — or none.
- **Predictions without affect** ("rates will probably stay flat through Q3") → no pair.
  Affectively loaded predictions ("this will end badly") → pair.
- **Personal anecdotes**: emotion about one's own situation caused by an economic condition
  ("lost my job in the layoffs, I'm devastated") → valid pair (`employment`).
- **Links/emoji-only comments**: usually `pairs: []`; emoji can be an emotion span if they
  are the comment's expressive content (📉😭).
- Cap: at most **8 pairs** per comment (schema-enforced); prioritize the strongest/clearest.

## 8. Worked examples

Context is shown for readability; the model receives it in the prompt (see `prompts/`).

**E1 — two pairs, cause in comment**
> *Post title:* "Fed signals two more hikes this year"
> *Comment:* "I'm honestly terrified. The Fed keeps hiking into a slowdown — and my landlord
> just raised rent 15% anyway."

```json
{"pairs": [
  {"emotion": "fear_anxiety", "emotion_span": "I'm honestly terrified",
   "cause_span": "The Fed keeps hiking into a slowdown", "cause_source": "comment",
   "cause_category": "monetary_policy", "target_asset": "none",
   "intensity": 0.75, "sarcasm": false},
  {"emotion": "anger_disgust", "emotion_span": "my landlord just raised rent 15% anyway",
   "cause_span": "raised rent 15%", "cause_source": "comment",
   "cause_category": "housing", "target_asset": "housing",
   "intensity": 0.5, "sarcasm": false}
]}
```

**E2 — cause in the post title, low-arousal negative**
> *Post title:* "US already in recession, Fed will cut rates back to zero — analyst"
> *Comment:* "Damn. I mean, we all saw it coming, but that is sobering."

```json
{"pairs": [
  {"emotion": "pessimism_despair", "emotion_span": "that is sobering",
   "cause_span": "US already in recession", "cause_source": "post",
   "cause_category": "growth_gdp", "target_asset": "none",
   "intensity": 0.5, "sarcasm": false}
]}
```

**E3 — sarcasm**
> *Comment:* "Oh fantastic, another trillion on the deficit. I'm sure our grandkids will
> thank us."

```json
{"pairs": [
  {"emotion": "anger_disgust", "emotion_span": "Oh fantastic, another trillion on the deficit",
   "cause_span": "another trillion on the deficit", "cause_source": "comment",
   "cause_category": "fiscal_taxes", "target_asset": "none",
   "intensity": 0.75, "sarcasm": true}
]}
```

**E4 — neutral (analytical, no affect)**
> *Comment:* "Many people that make 425k+ probably hide a lot of it anyways. So they never
> hit that 20% point."

```json
{"pairs": []}
```

**E5 — interpersonal affect → annotated with `interpersonal` (rule §3)**
> *Comment:* "So your whole reason against it is because it's 'not fair'. Did you even read
> the article?"

```json
{"pairs": [
  {"emotion": "anger_disgust",
   "emotion_span": "Did you even read the article?",
   "cause_span": "your whole reason against it is because it's 'not fair'",
   "cause_source": "comment", "cause_category": "interpersonal",
   "target_asset": "none", "intensity": 0.5, "sarcasm": false}
]}
```

## 9. Version log

- **v0.1** (2026-07-02): initial freeze after taxonomy lit review. Additions over the
  2026-07-01 draft, flagged ⚑ for review: `cause_source` field, `sarcasm` flag, `unclear`
  cause value + null-cause rule, interpersonal-affect exclusion, 8-pair cap.
  Valence/arousal dimensions deliberately **not** annotated (the categorical set preserves
  the arousal split; see lit review §3.5).
- **v0.2** (2026-07-02): flags resolved after discussion. (1) `cause_source` demoted from
  trusted model output to **validator-enforced** (§4). (2) `sarcasm` kept; its evaluation
  status decided by pilot κ (§10); consensus disagreements on sarcasm routed to the judge
  with an explicit irony re-read instruction. (3) Interpersonal exclusion **reversed** →
  new `cause_category: "interpersonal"`, annotated and filtered downstream (§3).
  (4) `unclear` + 8-pair cap unchanged, monitored per §10.
- **v0.3** (2026-07-02): pilot-gate revision after the DeepSeek pilot (all §10 gates passed;
  manual audit of all 126 pairs + 30 neutrals). (1) `other` split: new categories
  `inequality_distribution` + `personal_finance` (16 total) — audited `other` was 19.8% of
  pairs and dominated by these two themes. (2) Interpersonal restricted to discussion
  participants; public figures → policy category (§3). (3) Rhetorical questions ≠
  `confusion_uncertainty` (§7). (4) Sampler hygiene: moderator rule-boilerplate excluded.

## 10. Pilot gate criteria (pre-committed 2026-07-02)

The pilot: ~250 comments, stress-stratified ≈ 40% top-level comments on link posts (stresses
post-title causes), 40% replies from argumentative threads (stresses parent causes +
interpersonal), 20% pure random (honest base rates). Three-model run + joint human review.
A flag-report script computes the metrics below; the rules are fixed **before** seeing them:

| Metric | Expectation | Pre-committed action |
|---|---|---|
| Verbatim span-hit rate (after repair loop) | ≥ 95% | below → span rules or prompt are broken; fix before consensus run |
| `cause_source` ∈ {post, parent} share | ≥ 5% | below → keep field in data, drop from paper schema description |
| Sarcasm rate | ~10–25%; any model > 40% = over-triggering | κ ≥ 0.5 → evaluate in paper; 0.2–0.5 → auxiliary metadata only; < 0.2 → drop flag and audit emotion labels on sarcastic comments |
| `unclear` pairs | < 10% of pairs; similar across models | manually review **all** unclear pairs; one model overusing → prompt fix |
| Pairs per comment | ~90% of emotive comments ≤ 2 pairs; cap never hit | any comment at the 8-cap → inspect |
| `interpersonal` share | report prevalence | < 5% and noisy → may merge into `other` at analysis time (data keeps the label) |
