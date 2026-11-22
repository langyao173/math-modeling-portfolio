"""文件读取与已存结果校核。"""

import hashlib

import json

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]



def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(path):
    return json.loads(Path(path).read_text("utf-8"))

def verify_review(path):
    record = load(path)
    assert record["status"].startswith("PASS_"), str(path)
    assert record["reviewed_source_sha256"], ("Review must bind its actual sources", str(path))
    for name, expected in record["reviewed_source_sha256"].items():
        assert sha(ROOT / name) == expected, ("Reviewed source changed", name)
    for name, expected in record.get("reviewed_evidence_sha256", {}).items():
        target = ROOT / name if "/" in name else Path(path).parent / name
        assert sha(target) == expected, ("Reviewed evidence changed", name)
    return record
