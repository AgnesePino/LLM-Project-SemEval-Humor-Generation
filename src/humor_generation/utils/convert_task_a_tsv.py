import argparse
import csv

from humor_generation.utils.io import write_jsonl


MISSING_VALUES = {"", "-", "nan", "none", "null"}


def has_value(value):
    return str(value).strip().casefold() not in MISSING_VALUES


def convert_task_a_tsv(input_path, output_path):
    rows = []

    with open(input_path, encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file, delimiter="\t")
        required_columns = {"id", "word1", "word2", "headline"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

        for row in reader:
            word1 = row["word1"].strip()
            word2 = row["word2"].strip()
            headline = row["headline"].strip()
            has_word1 = has_value(word1)
            has_word2 = has_value(word2)
            has_headline = has_value(headline)

            if has_word1 != has_word2:
                raise ValueError(
                    f"Row {row['id']} has only one word. Word Inclusion requires both word1 and word2."
                )

            if has_word1 and has_word2:
                rows.append(
                    {
                        "id": row["id"],
                        "type": "word_inclusion",
                        "word1": word1,
                        "word2": word2,
                    }
                )
            elif has_headline:
                rows.append(
                    {
                        "id": row["id"],
                        "type": "news_headline",
                        "headline": headline,
                    }
                )
            else:
                raise ValueError(f"Row {row['id']} has neither a word pair nor a headline.")

    write_jsonl(output_path, rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert SemEval Task A TSV data to internal JSONL.")
    parser.add_argument("--input", required=True, help="Path to a TSV file with id, word1, word2, headline.")
    parser.add_argument("--output", required=True, help="Path where the converted JSONL file will be written.")
    return parser.parse_args()


def main():
    args = parse_args()
    convert_task_a_tsv(args.input, args.output)


if __name__ == "__main__":
    main()
