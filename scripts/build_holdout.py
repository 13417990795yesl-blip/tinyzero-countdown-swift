"""Rebuild the 256-question Countdown holdout without cross-split overlap."""

import argparse
import hashlib
import json
from pathlib import Path

from gen_countdown import gen

SIZE = 256
SEED = 20261126


def load_questions(path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def key(row):
    return tuple(sorted(row["nums"])), row["target"]


def build_holdout(train, validation):
    seen = {key(row) for row in train}
    seen.update(key(row) for row in validation)
    result = []
    for row in gen(2 * SIZE, SEED):
        signature = key(row)
        if signature in seen:
            continue
        result.append(row)
        seen.add(signature)
        if len(result) == SIZE:
            return result
    raise ValueError(f"Only generated {len(result)} independent test questions")


def serialize(rows):
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train = load_questions(args.train)
    validation = load_questions(args.validation)
    content = serialize(build_holdout(train, validation))
    with args.output.open("xb") as output:
        output.write(content)
    print(f"{SIZE} questions, seed {SEED}, SHA-256 {hashlib.sha256(content).hexdigest()}")


if __name__ == "__main__":
    main()
