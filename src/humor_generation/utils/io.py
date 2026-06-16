import json
from pathlib import Path


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")

