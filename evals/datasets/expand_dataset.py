"""Expanded evaluation dataset with in-document and out-of-document questions."""

import json
from pathlib import Path

# Load the existing Dangote dataset
existing_file = Path(__file__).parent / "dangote_prospectus.jsonl"
existing_data = []
if existing_file.exists():
    with open(existing_file) as f:
        for line in f:
            existing_data.append(json.loads(line))

# Extend with abstention test cases
abstention_cases = [
    {
        "id": "abstain-live-prices",
        "question": "What are the current market prices for Dangote shares?",
        "expected_outcome": "abstained",
        "reason": "Prospectus is a snapshot; live market data not included",
        "source_pages": [],
    },
    {
        "id": "abstain-competitor-comparison",
        "question": "How does Dangote compare to other African refineries?",
        "expected_outcome": "abstained",
        "reason": "Out of document scope; prospectus focuses only on Dangote",
        "source_pages": [],
    },
    {
        "id": "abstain-future-projections",
        "question": "Will Dangote shares go up in price after listing?",
        "expected_outcome": "abstained",
        "reason": "Investment advice and predictions are out of scope",
        "source_pages": [],
    },
    {
        "id": "abstain-external-news",
        "question": "What happened with Dangote's recent environmental violations?",
        "expected_outcome": "abstained",
        "reason": "Current events not in prospectus",
        "source_pages": [],
    },
]

# Write to a new file
output_file = Path(__file__).parent / "dangote_with_abstentions.jsonl"
with open(output_file, "w") as f:
    for case in existing_data + abstention_cases:
        f.write(json.dumps(case) + "\n")

print(f"Created {output_file} with {len(existing_data)} existing + {len(abstention_cases)} abstention cases")
