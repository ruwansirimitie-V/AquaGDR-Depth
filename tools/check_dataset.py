"""Check that all RGB/depth paths listed in a split file exist."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/underwater_erp"))
    parser.add_argument("--split", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    split = args.split.resolve()
    if not split.is_file():
        parser.error(f"Split file does not exist: {split}")

    missing = []
    rows = 0
    for line_number, line in enumerate(split.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        columns = line.split()
        if len(columns) != 2:
            missing.append((line_number, "expected exactly two whitespace-separated paths"))
            continue
        rows += 1
        for relative_path in columns:
            path = (root / relative_path).resolve()
            if not path.is_relative_to(root):
                missing.append((line_number, f"path escapes dataset root: {relative_path}"))
            elif not path.is_file():
                missing.append((line_number, f"missing file: {relative_path}"))

    if missing:
        for line_number, message in missing[:50]:
            print(f"line {line_number}: {message}")
        raise SystemExit(f"Dataset check failed: {len(missing)} problem(s), {rows} row(s) read")

    print(f"Dataset check passed: {rows} RGB/depth pairs in {split}")


if __name__ == "__main__":
    main()
