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
