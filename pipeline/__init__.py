"""EconECPE annotation pipeline (Phase 1+).

Modules:
  textnorm  — span normalization + source derivation (spec §4: validator-enforced
              cause_source).
  context   — SQLite index over the scraped JSONL (post/parent lookup) + context
              assembly for the annotation prompt.
  prompt    — builds the chat messages: system prompt + few-shot + formatted item.
  validate  — JSON-schema + span-rule validation of model outputs; repair prompts.
  sample    — hygiene filters + stress-stratified pilot sampler (spec §10).
  llm       — minimal OpenAI-compatible client: concurrent, resumable label().
  annotate  — orchestrates llm.label(): call → validate → repair → merge.

No provider is assumed: `llm` talks to any OpenAI-compatible endpoint
(LLM_BASE_URL / LLM_API_KEY), and every model-calling module has a `--mock`
path that runs offline with no key.
"""
