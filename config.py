"""Central configuration for the r/economics corpus + annotation pipeline.

The corpus is ingested by arctic_shift.py from the public Arctic Shift archive —
a read-only, rate-limit-respecting API. That is the only way text enters this
project; nothing here scrapes reddit.com directly.
"""
import os

# --- .env loader (dependency-free) --------------------------------------------
def _load_dotenv():
    """Load KEY=VALUE lines from a sibling .env into os.environ (without
    overriding values already set in the real environment)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
    except FileNotFoundError:
        pass


_load_dotenv()

# --- Target subreddit ---------------------------------------------------------
# Arctic Shift is case-insensitive.
SUBREDDIT = os.environ.get("ECONECPE_SUBREDDIT", "economics")

# --- Date window --------------------------------------------------------------
# Only ingest activity created at/after this instant (Unix UTC seconds).
# 1514764800 = 2018-01-01 00:00:00 UTC. Set ECONECPE_START_TS=0 for full history.
START_TS = int(os.environ.get("ECONECPE_START_TS", "1514764800"))

# --- Paths --------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Arctic Shift historical archive (JSONL, append-only, resumable)
SUBMISSIONS_FILE = os.path.join(DATA_DIR, f"{SUBREDDIT}_submissions.jsonl")
COMMENTS_FILE = os.path.join(DATA_DIR, f"{SUBREDDIT}_comments.jsonl")
ARCTIC_CHECKPOINT = os.path.join(DATA_DIR, "arctic_checkpoint.json")

# --- Arctic Shift API ---------------------------------------------------------
ARCTIC_BASE = "https://arctic-shift.photon-reddit.com/api"
# "auto" returns up to 1000 items/page (vs 100 for a fixed number) — ~10x fewer
# requests, so the full 2018-2026 comment pull drops from ~3 days to ~8 hours.
ARCTIC_PAGE_LIMIT = os.environ.get("ECONECPE_PAGE_LIMIT", "auto")
ARCTIC_SLEEP = 1.0               # polite delay between requests; the API 422s ("slow down") if too fast
ARCTIC_MAX_RETRIES = 8           # per-request retries (429 + any 5xx, incl. Cloudflare 52x)
ARCTIC_MAX_FATAL = 30            # consecutive exhausted-retry batches before giving up
ARCTIC_USER_AGENT = "econecpe-research/1.0 (academic replication)"

# --- Annotation pipeline (Phase 1+) --------------------------------------------
SPEC_DIR = os.path.join(BASE_DIR, "spec")
SPEC_SCHEMA = os.path.join(SPEC_DIR, "schema.json")
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")
# posts + comments lookup index. Overridable so a second subreddit can be
# indexed/labeled into its own DB without touching the economics index (whose
# incremental byte-offsets are per-file and would break if streams are mixed).
INDEX_DB = os.environ.get("ECONECPE_INDEX_DB", os.path.join(DATA_DIR, "index.sqlite"))
SAMPLES_DIR = os.path.join(DATA_DIR, "samples")
LABELS_DIR = os.path.join(DATA_DIR, "labels")

# Primary annotation model (pilot + 1st consensus vote). Model slugs follow the
# provider/model convention; use whatever your endpoint expects (see
# pipeline/llm.py for endpoint configuration).
ANNOTATE_MODEL = os.environ.get("ECONECPE_ANNOTATE_MODEL", "deepseek/deepseek-v4-flash")
ANNOTATE_MAX_WORKERS = int(os.environ.get("ECONECPE_ANNOTATE_WORKERS", "8"))

# Context caps (chars) when assembling the model input. Set to Reddit's own
# maxima (comment 10k, selftext 40k) = whole texts in practice. Truncating the
# parent was observed (pilot2, dslo66j) to invite hallucinated cause spans from
# the unseen remainder; token cost of full context is negligible at these prices.
CTX_SELFTEXT_MAX = 40_000
CTX_PARENT_MAX = 10_000
