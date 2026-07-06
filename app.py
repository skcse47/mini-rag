from pathlib import Path

document = Path("data/company").read_text(encoding="utf-8")

print(document)