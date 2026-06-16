import argparse
import json

from humor_generation.evaluation.validators import validate_candidate
from humor_generation.utils.io import load_jsonl


def load_items_by_id(path):
    return {item["id"]: item for item in load_jsonl(path)}


def validate_outputs(input_path, output_path, report_path):
    items_by_id = load_items_by_id(input_path)
    total = 0
    valid = 0
    invalid_rows = []

    for prediction in load_jsonl(output_path):
        total += 1
        item = items_by_id[prediction["id"]]
        result = validate_candidate(prediction.get("joke", ""), item)
        if result.is_valid:
            valid += 1
        else:
            invalid_rows.append(
                {
                    "id": prediction["id"],
                    "reasons": result.reasons,
                    "joke": prediction.get("joke", ""),
                }
            )

    report = {
        "total": total,
        "valid": valid,
        "invalid": total - valid,
        "constraint_satisfaction": valid / total if total else 0.0,
        "invalid_rows": invalid_rows,
    }

    with open(report_path, "w", encoding="utf-8") as output_file:
        json.dump(report, output_file, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Validate generated jokes against task constraints.")
    parser.add_argument("--input", required=True, help="Prepared JSONL input file.")
    parser.add_argument("--predictions", required=True, help="Generated predictions JSONL file.")
    parser.add_argument("--report", required=True, help="Path to the validation report JSON.")
    return parser.parse_args()


def main():
    args = parse_args()
    validate_outputs(args.input, args.predictions, args.report)


if __name__ == "__main__":
    main()

