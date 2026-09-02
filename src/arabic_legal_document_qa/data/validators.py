import os 
import json
import sys
from arabic_legal_document_qa.configs.config import Settings, get_settings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
JSON_PATH = (PROJECT_ROOT/ "data"/ "processed"/ "egyptian_civil_code.json")


if __name__ == "__main__":
    
    settings = get_settings()


    # Diagnostic CLI output runner
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"--- Legal JSON Validation Report ---")
    print(f"Total Records: {len(data)}")

    numbers = [a["article_number"] for a in data]
    repealed = [a for a in data if a["is_repealed"]]
    empty_ar = [a for a in data if not a["is_repealed"] and not a["text_ar"].strip()]
    empty_en = [a for a in data if not a["is_repealed"] and not a["text_en"].strip()]
    giant_articles = [
        a for a in data 
        if len(a["text_ar"]) > settings.MAX_CHAR_LENGTH_THRESHOLD or len(a["text_en"]) > settings.MAX_CHAR_LENGTH_THRESHOLD
    ]

    print(f"Unique Article Numbers: {len(set(numbers))}")
    print(f"Repealed Articles Count: {len(repealed)}")
    print(f"Missing Arabic Body Count: {len(empty_ar)}")
    print(f"Missing English Body Count: {len(empty_en)}")
    print(f"Giant Articles (> {settings.MAX_CHAR_LENGTH_THRESHOLD} chars): {len(giant_articles)}")

    missing_seq = sorted(list(set(range(1, settings.MAX_CIVIL_CODE_ARTICLE_NUMBER + 1)) - set(numbers)))
    if missing_seq:
        print(f"Sequence Gaps Detected ({len(missing_seq)} missing): {missing_seq[:15]}...")
    else:
        print("Sequence is 100% contiguous (1 to 1149)")

    # Only exit when running in a standalone terminal script, not IPython/Jupyter
    if "ipykernel" not in sys.modules and "IPython" not in sys.modules:
        sys.exit(0)