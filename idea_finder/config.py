"""Settings in one place: models, dataset names and paths.

Any value read from the environment can also be set in a .env file in the project root.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")  # never overrides variables that are already set

EXTRACT_MODEL = os.environ.get("EXTRACT_MODEL", "claude-haiku-5-5")
LABEL_MODEL = os.environ.get("LABEL_MODEL", "claude-sonnet-5-5")
AGENT_MODEL = os.environ.get("AGENT_MODEL", "claude-sonnet-5-5")
CRITIC_MODEL = os.environ.get("CRITIC_MODEL", "claude-opus-5-5")

# Models that accept a forced tool_choice ({"type": "tool", ...}).
# Claude Sonnet 5.5 and Opus 5.5 return a 400 error for it (tool-use docs, checked 2026-10-29).
FORCED_TOOL_CHOICE_MODELS = {"claude-haiku-5-5"}

# Most one agent run may spend, in US dollars.
MAX_COST_PER_RUN = float(os.environ.get("MAX_COST_PER_RUN", "2.00"))

# Prices in US dollars per million tokens.
# Checked on 2026-10-29 at https://platform.claude.com/docs/en/about-claude/pricing
# "cache_write" is the 5-minute cache rate; we only use 5-minute caching.
# Haiku 5.5 costs more when the prompt is over 100,000 tokens ("long_prompt").
PRICES: dict[str, dict] = {
    "claude-haiku-5-5": {
        "input": 0.10, "cache_write": 0.125, "cache_read": 0.01, "output": 0.50,
        "long_prompt": {"input": 0.50, "cache_write": 0.625, "cache_read": 0.05, "output": 2.50},
    },
    "claude-sonnet-5-5": {"input": 2.00, "cache_write": 2.50, "cache_read": 0.10, "output": 10.00},
    "claude-opus-5-5": {"input": 4.00, "cache_write": 5.00, "cache_read": 0.20, "output": 20.00},
}
LONG_PROMPT_TOKENS = 100_000

HF_REPO = "McAuley-Lab/Amazon-Reviews-2023"
HF_PATH = "datasets/{repo}/raw/review_categories/{category}.jsonl"

# Category names in the Amazon Reviews 2023 dataset (data covers 1996 to Sep 2023).
CATEGORIES = [
    "All_Beauty", "Amazon_Fashion", "Appliances", "Arts_Crafts_and_Sewing",
    "Automotive", "Baby_Products", "Beauty_and_Personal_Care", "Books",
    "Cell_Phones_and_Accessories", "Clothing_Shoes_and_Jewelry", "Electronics",
    "Grocery_and_Gourmet_Food", "Handmade_Products", "Health_and_Household",
    "Home_and_Kitchen", "Industrial_and_Scientific", "Office_Products",
    "Patio_Lawn_and_Garden", "Pet_Supplies", "Sports_and_Outdoors",
    "Tools_and_Home_Improvement", "Toys_and_Games", "Video_Games",
]

STAGES = ["collect", "extract", "cluster", "label", "report", "sample"]
