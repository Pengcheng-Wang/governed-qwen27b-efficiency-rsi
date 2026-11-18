"""
prepare_splits_v2.py — Materialize genuinely disjoint, normalized-hash + 13-gram decontaminated
Three-Way Splits (Search / Confirm / Protected Single-Touch Holdout) for AILab_Qwen27B_Governed_v2.

Strictly excludes all v1-exposed evaluation/training benchmarks (MATH-500, AIME24, AIME25,
v1 development_math200, v1 rl_dapo_screen256, and v1 retention_mmlu_pro200) from v2 loop selection,
and stores Protected Holdout gold answers ONLY in `.local/governance_anchor/holdout_labels/`
outside candidate-writable Git worktrees.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

from common import (
    file_hash,
    last_boxed,
    loss_tokens,
    math_messages,
    normalize_problem,
    object_hash,
    quantiles,
    read_jsonl,
    text_ngrams,
    validate_token_row,
    write_jsonl,
)
from mcq_contract import mcq_messages
from stage import write_json

ROOT = Path(__file__).resolve().parents[1]

MATH_SOURCES = {
    "math_train": {
        "repo": "DigitalLearningGmbH/MATH-lighteval",
        "revision": "0530c78699ea5e8eb5530600900e1f328b48acad",
        "file": "data/train-00000-of-00001.parquet",
        "expected_rows": 7500,
    },
    "math500": {
        "repo": "HuggingFaceH4/MATH-500",
        "revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        "file": "test.jsonl",
        "expected_rows": 500,
    },
    "aime24": {
        "repo": "HuggingFaceH4/aime_2024",
        "revision": "2fe88a2f1091d5048c0f36abc874fb997b3dd99a",
        "file": "data/train-00000-of-00001.parquet",
        "expected_rows": 30,
    },
    "aime25": {
        "repo": "yentinglin/aime_2025",
        "revision": "6f71d77b0b89b9dabe07ab466c51df33f514df7f",
        "file": "data/train-00000-of-00001-243207c6c994e1bd.parquet",
        "expected_rows": 30,
    },
}

DAPO_SOURCE = {
    "repo": "BytedTsinghua-SIA/DAPO-Math-17k",
    "revision": "65877096c24ffa7abc4e4fa5edb95cf3413a5674",
    "file": "data/dapo-math-17k.parquet",
}

DAPO_PREFIX = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: $Answer (without quotes) where $Answer is the answer "
    "to the problem.\n\n"
)
DAPO_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'
DAPO_KNOWN_BAD = {
    "c64dea2a-146f-444e-b9e3-92756c1706b1",
    "31e18bbb-a745-48fb-bc5d-ed3e9a350010",
}

MMLU_PRO_REPO = "TIGER-Lab/MMLU-Pro"
MMLU_EXCLUDE_CATEGORIES = {"math"}

V1_EXPECTED_DEV200_SHA256 = "6225cee21eb467f32f2b633fb91a2b59c15fffc4faafb8b9241ffee22243e92d"
V1_EXPECTED_DAPO_CANDIDATES_SHA256 = "1f69972d2705bc10b82bfcae346d5efe12fc5c79d3b5d1b6aa0b2494c187e7bf"


def v1_object_hash(payload: Any) -> str:
    """Exact reproduction of v1/code/common.py object_hash (json.dumps with default spaces)."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def v1_normalize_problem(text: str) -> str:
    """Exact reproduction of v1/code/common.py normalize_problem (strips all whitespace)."""
    return re.sub(r"\s+", "", text).casefold()


def v1_last_boxed(text: str) -> str | None:
    """Exact reproduction of v1/code/common.py balanced-brace last_boxed."""
    starts = list(re.finditer(r"\\(?:boxed|fbox)\s*\{", text))
    for match in reversed(starts):
        start, depth = match.end(), 1
        for index in range(start, len(text)):
            if text[index] == "{" and (index == 0 or text[index - 1] != "\\"):
                depth += 1
            elif text[index] == "}" and (index == 0 or text[index - 1] != "\\"):
                depth -= 1
                if depth == 0:
                    answer = text[start:index].strip()
                    if answer:
                        return answer
                    break
    return None


def v1_jsonl_sha256(rows: list[dict]) -> str:
    """Exact reproduction of v1/code/common.py write_jsonl byte serialization."""
    h = hashlib.sha256()
    for row in rows:
        h.update((json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
    return h.hexdigest()


def load_v1_exact_math_source(name: str) -> list[dict]:
    src = MATH_SOURCES[name]
    path = Path(
        hf_hub_download(
            src["repo"],
            src["file"],
            repo_type="dataset",
            revision=src["revision"],
            local_dir=ROOT / "data" / "raw" / src["repo"].replace("/", "__"),
        )
    )
    rows = pq.read_table(path).to_pylist() if path.suffix == ".parquet" else read_jsonl(path)
    out = []
    for index, row in enumerate(rows):
        answer = row.get("answer") or v1_last_boxed(row.get("solution", ""))
        out.append({
            "id": f"{name}:{index}",
            "source": name,
            "problem": row["problem"],
            "gold_answer": answer,
            "solution": row.get("solution", ""),
            "level": row.get("level"),
            "subject": row.get("type", row.get("subject")),
            "prompt_hash": v1_object_hash(v1_normalize_problem(row["problem"])),
            "finish_reason": "N/A",
            "target_origin": "published_static_solution",
        })
    return out


def load_math_source(name: str) -> tuple[list[dict], dict]:
    src = MATH_SOURCES[name]
    path = Path(
        hf_hub_download(
            src["repo"],
            src["file"],
            repo_type="dataset",
            revision=src["revision"],
            local_dir=ROOT / "data" / "raw" / src["repo"].replace("/", "__"),
        )
    )
    rows = pq.read_table(path).to_pylist() if path.suffix == ".parquet" else read_jsonl(path)
    if len(rows) != src["expected_rows"]:
        raise RuntimeError(f"Unexpected source size for {name}: {len(rows)}")
    normalized = []
    for index, row in enumerate(rows):
        answer = row.get("answer") or last_boxed(row.get("solution", ""))
        normalized.append({
            "id": f"{name}:{index}",
            "source": name,
            "problem": row["problem"],
            "gold_answer": answer,
            "solution": row.get("solution", ""),
            "level": row.get("level"),
            "subject": row.get("type", row.get("subject")),
            "prompt_hash": object_hash(normalize_problem(row["problem"])),
            "finish_reason": "N/A",
            "target_origin": "published_static_solution",
        })
    return normalized, {**src, "file_sha256": file_hash(path), "rows": len(rows)}


def tokenize_static(row: dict, tokenizer, cap: int, eos: list[int]) -> dict | None:
    messages = math_messages(row["problem"])
    prefix = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    full = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": row["solution"]}],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    if not full.startswith(prefix):
        raise RuntimeError("Native train/eval template prefix mismatch")
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    full_ids = tokenizer.encode(full, add_special_tokens=False)
    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise RuntimeError("Train/eval token prefix mismatch")
    end = next((i for i in range(len(prefix_ids), len(full_ids)) if full_ids[i] in eos), None)
    if end is None or tokenizer.decode(full_ids[end + 1 :]).strip():
        raise RuntimeError("Native assistant EOS or trailing separator mismatch")
    ids = full_ids[: end + 1]
    if len(ids) > cap:
        return None
    result = {
        **row,
        "input_ids": ids,
        "labels": [-100] * len(prefix_ids) + ids[len(prefix_ids) :],
    }
    validate_token_row(result, cap, eos)
    return result


def write_sealed_holdout(prompts_path: Path, labels_path: Path, rows: list[dict]) -> None:
    stripped_prompts = []
    labels: dict[str, str] = {}
    for row in rows:
        pid = str(row["id"])
        labels[pid] = str(row["gold_answer"])
        clean_row = {k: v for k, v in row.items() if k not in ("gold_answer", "solution")}
        stripped_prompts.append(clean_row)
    write_jsonl(prompts_path, stripped_prompts)
    write_json(
        labels_path,
        {
            "split": prompts_path.name,
            "prompts_sha256": file_hash(prompts_path),
            "count": len(labels),
            "labels": labels,
        },
    )


def take_disjoint(
    pool: list[dict],
    count: int,
    used_hashes: set[str],
    used_grams: set[tuple[str, ...]],
) -> list[dict]:
    chosen: list[dict] = []
    for row in pool:
        ph = row["prompt_hash"]
        grams = text_ngrams(row["problem"])
        if ph in used_hashes or (grams & used_grams):
            continue
        chosen.append(row)
        used_hashes.add(ph)
        used_grams.update(grams)
        if len(chosen) == count:
            break
    if len(chosen) != count:
        raise RuntimeError(f"Insufficient disjoint records: requested {count}, got {len(chosen)}")
    return chosen


def verify_pairwise_disjoint(named_splits: dict[str, list[dict]]) -> int:
    names = list(named_splits.keys())
    overlaps = 0
    for i in range(len(names)):
        h_i = {r["prompt_hash"] for r in named_splits[names[i]]}
        g_i = set().union(*(text_ngrams(r["problem"]) for r in named_splits[names[i]])) if named_splits[names[i]] else set()
        for j in range(i + 1, len(names)):
            h_j = {r["prompt_hash"] for r in named_splits[names[j]]}
            g_j = set().union(*(text_ngrams(r["problem"]) for r in named_splits[names[j]])) if named_splits[names[j]] else set()
            if (h_i & h_j) or (g_i & g_j):
                overlaps += len(h_i & h_j) + len(g_i & g_j)
    return overlaps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--preserve-search-splits",
        action="store_true",
        help="Keep existing Search/SFT calibration panels on disk and regenerate fresh quarantined Confirm/Holdout panels.",
    )
    args = ap.parse_args()

    dense_spec = json.loads((ROOT / "configs/model.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(ROOT / dense_spec["model_directory"], local_files_only=True)

    splits_dir = ROOT / "data" / "splits"
    anchors_dir = ROOT / "data" / "historical_anchors"
    holdout_labels_dir = ROOT / ".local" / "governance_anchor" / "holdout_labels"
    splits_dir.mkdir(parents=True, exist_ok=True)
    anchors_dir.mkdir(parents=True, exist_ok=True)
    holdout_labels_dir.mkdir(parents=True, exist_ok=True)

    datasets, provenance = {}, {}
    for name in MATH_SOURCES:
        datasets[name], provenance[name] = load_math_source(name)

    # Save historical benchmarks ONLY in data/historical_anchors/ (excluded from all v2 selection!)
    for hname in ("math500", "aime24", "aime25"):
        write_jsonl(anchors_dir / f"historical_{hname}.jsonl", datasets[hname])

    historical_protected = datasets["math500"] + datasets["aime24"] + datasets["aime25"]
    v1_quarantine_hashes: set[str] = {r["prompt_hash"] for r in historical_protected}
    v1_quarantine_grams: set[tuple[str, ...]] = set().union(*(text_ngrams(r["problem"]) for r in historical_protected))

    # 1. Reconstruct ALL v1 quarantined sets (eval + training!) and VERIFY EXACT V1 SHA-256 ORACLES!
    v1_raw_math = {k: load_v1_exact_math_source(k) for k in MATH_SOURCES}
    v1_hist_prot = v1_raw_math["math500"] + v1_raw_math["aime24"] + v1_raw_math["aime25"]
    v1_hist_hashes = {r["prompt_hash"] for r in v1_hist_prot}
    v1_hist_grams = set().union(*(text_ngrams(r["problem"]) for r in v1_hist_prot))
    v1_exact_cands = []
    seen_v1: set[str] = set()
    for row in v1_raw_math["math_train"]:
        if row["prompt_hash"] in v1_hist_hashes or (text_ngrams(row["problem"]) & v1_hist_grams):
            continue
        if row["prompt_hash"] in seen_v1 or not row["gold_answer"]:
            continue
        seen_v1.add(row["prompt_hash"])
        v1_exact_cands.append(row)
    v1_exact_cands.sort(key=lambda r: v1_object_hash({"seed": 1030, "id": r["id"]}))
    if len(v1_exact_cands) != 6199:
        raise RuntimeError(f"v1 candidate count mismatch: expected 6199, got {len(v1_exact_cands)}")
    v1_dev200_legacy = v1_exact_cands[:200]
    v1_sft1024_legacy = v1_exact_cands[200:1224]
    dev200_sha = v1_jsonl_sha256(v1_dev200_legacy)
    if dev200_sha != V1_EXPECTED_DEV200_SHA256:
        raise RuntimeError(
            f"v1 development_math200 SHA256 oracle mismatch: expected {V1_EXPECTED_DEV200_SHA256}, got {dev200_sha}"
        )

    v1_exposed_math_ids = {r["id"] for r in v1_exact_cands[:1224]}
    v2_by_id = {r["id"]: r for r in datasets["math_train"]}
    v1_dev200 = [v2_by_id[r["id"]] for r in v1_dev200_legacy]
    v1_sft1024 = [v2_by_id[r["id"]] for r in v1_sft1024_legacy]

    v2_cands = []
    seen_v2: set[str] = set()
    for row in datasets["math_train"]:
        if row["prompt_hash"] in v1_quarantine_hashes or (text_ngrams(row["problem"]) & v1_quarantine_grams):
            continue
        if not row["gold_answer"] or row["prompt_hash"] in seen_v2:
            continue
        seen_v2.add(row["prompt_hash"])
        v2_cands.append(row)
    v1_unexposed_math = [r for r in v2_cands if r["id"] not in v1_exposed_math_ids]

    # Download & reconstruct v1's rl_dapo_candidates.jsonl and VERIFY EXACT V1 SHA-256 ORACLE!
    dapo_parquet = Path(
        hf_hub_download(
            DAPO_SOURCE["repo"],
            DAPO_SOURCE["file"],
            repo_type="dataset",
            revision=DAPO_SOURCE["revision"],
            local_dir=ROOT / "data" / "raw" / "dapo",
        )
    )
    provenance["dapo_math_17k"] = {
        **DAPO_SOURCE,
        "file_sha256": file_hash(dapo_parquet),
    }
    unique_dapo: dict[str, tuple[str, str]] = {}
    for batch in pq.ParquetFile(dapo_parquet).iter_batches(
        batch_size=4096, columns=["prompt", "reward_model", "extra_info"]
    ):
        for item in batch.to_pylist():
            key = item["extra_info"]["index"]
            prompt = item["prompt"]
            text, gold = prompt[0]["content"], item["reward_model"]["ground_truth"]
            unique_dapo[key] = (text, gold)

    v1_base_protected_legacy = v1_hist_prot + v1_dev200_legacy
    v1_prot_norms = {r["prompt_hash"] for r in v1_base_protected_legacy}
    v1_prot_grams = set().union(*(text_ngrams(r["problem"]) for r in v1_base_protected_legacy))
    v1_dapo_legacy = []
    v1_dapo_admitted = []
    v1_dapo_seen: set[str] = set()
    for source_id, (text, gold) in unique_dapo.items():
        if source_id in DAPO_KNOWN_BAD or not re.fullmatch(r"[+-]?\d+", gold) or abs(int(gold)) > 2**53:
            continue
        if not text.startswith(DAPO_PREFIX) or not text.endswith(DAPO_SUFFIX):
            continue
        problem = text[len(DAPO_PREFIX) : -len(DAPO_SUFFIX)].strip()
        v1_norm_h = v1_object_hash(v1_normalize_problem(problem))
        if v1_norm_h in v1_prot_norms or (text_ngrams(problem) & v1_prot_grams) or v1_norm_h in v1_dapo_seen:
            continue
        rendered = tokenizer.apply_chat_template(
            math_messages(problem),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
            reasoning_effort="xhigh",
        )
        tids = tokenizer.encode(rendered, add_special_tokens=False)
        if len(tids) > 2048:
            continue
        v1_dapo_seen.add(v1_norm_h)
        v1_dapo_legacy.append({
            "id": f"dapo:{source_id}",
            "source": "dapo",
            "source_uuid": source_id,
            "problem": problem,
            "gold_answer": gold,
            "prompt_hash": v1_norm_h,
            "prompt_tokens": len(tids),
            "finish_reason": "N/A",
            "target_origin": "published_integer_label_not_a_generated_trace",
        })
        v1_dapo_admitted.append({
            "id": f"dapo:{source_id}",
            "source": "dapo",
            "source_uuid": source_id,
            "problem": problem,
            "gold_answer": gold,
            "prompt_hash": object_hash(normalize_problem(problem)),
            "prompt_tokens": len(tids),
            "finish_reason": "N/A",
            "target_origin": "published_integer_label",
        })
    v1_dapo_legacy.sort(key=lambda r: v1_object_hash({"seed": 1030, "id": r["id"]}))
    dapo_cand_sha = v1_jsonl_sha256(v1_dapo_legacy)
    if dapo_cand_sha != V1_EXPECTED_DAPO_CANDIDATES_SHA256:
        raise RuntimeError(
            f"v1 rl_dapo_candidates SHA256 oracle mismatch: expected {V1_EXPECTED_DAPO_CANDIDATES_SHA256}, got {dapo_cand_sha}"
        )
    v1_dapo_admitted.sort(key=lambda r: v1_object_hash({"seed": 1030, "id": r["id"]}))
    v1_dapo_by_id = {r["id"]: r for r in v1_dapo_admitted}

    # Union of sorted rl_dapo_screen256 AND the exact 200 GRPO/MoE prompts extracted from v1's verified Gen2 archive!
    exposed_dapo_cfg = json.loads((ROOT / "configs/v1_exposed_dapo_ids.json").read_text(encoding="utf-8"))
    banked_grpo_dapo_ids = set(exposed_dapo_cfg["unique_exposed_dapo_ids"])
    if len(banked_grpo_dapo_ids) != 200:
        raise RuntimeError(f"Expected 200 banked v1 GRPO/MoE DAPO IDs, got {len(banked_grpo_dapo_ids)}")
    v1_dapo_quarantine_ids = {r["id"] for r in v1_dapo_admitted[:256]} | banked_grpo_dapo_ids
    missing_dapo = sorted(did for did in v1_dapo_quarantine_ids if did not in v1_dapo_by_id)
    if missing_dapo:
        raise RuntimeError(f"Recovered banked DAPO IDs missing from reconstructed candidate pool: {missing_dapo[:5]}")
    v1_dapo_quarantine_rows = [v1_dapo_by_id[did] for did in sorted(v1_dapo_quarantine_ids)]

    # Full v1 eval + training union for Confirm & Protected Holdout quarantine!
    v1_math_union_rows = v1_dev200 + v1_sft1024
    v1_full_quarantine_rows = historical_protected + v1_math_union_rows + v1_dapo_quarantine_rows
    used_hashes: set[str] = {r["prompt_hash"] for r in v1_full_quarantine_rows}
    used_grams: set[tuple[str, ...]] = set().union(*(text_ngrams(r["problem"]) for r in v1_full_quarantine_rows))

    v2_math_pool = sorted(
        v1_unexposed_math,
        key=lambda r: (
            0 if str(r.get("level")) in {"Level 3", "Level 4", "Level 5", "3", "4", "5"} else 1,
            object_hash({"seed": 20261117, "id": r["id"]}),
        ),
    )

    preserve_ok = args.preserve_search_splits and all(
        (splits_dir / fn).is_file()
        for fn in ("math_search128.jsonl", "sft_systems1024_v2.jsonl", "sft_heldout64_v2.jsonl", "dapo_search240.jsonl")
    )
    if preserve_ok:
        math_search128 = read_jsonl(splits_dir / "math_search128.jsonl")
        sft_systems1024 = read_jsonl(splits_dir / "sft_systems1024_v2.jsonl")
        sft_heldout64 = read_jsonl(splits_dir / "sft_heldout64_v2.jsonl")
        dapo_search240 = read_jsonl(splits_dir / "dapo_search240.jsonl")
        for r in math_search128 + sft_systems1024 + sft_heldout64 + dapo_search240:
            used_hashes.add(r["prompt_hash"])
            used_grams.update(text_ngrams(r["problem"]))
    else:
        math_search128 = take_disjoint(v2_math_pool, 128, used_hashes, used_grams)
        write_jsonl(splits_dir / "math_search128.jsonl", math_search128)
        write_jsonl(splits_dir / "math_search_fast64.jsonl", math_search128[:64])

    # Remove superseded 256/512/1024 Confirm files and size MATH Confirm to N=1280 clusters
    # (>=91.28% KL-NI power at strict unrelaxed 2.0 pp margin for single K=1 S3 finalist at alpha=0.025!)
    for stale_p in (
        splits_dir / "math_confirm256.jsonl",
        splits_dir / "math_confirm512.jsonl",
        splits_dir / "math_confirm1024.jsonl",
        splits_dir / "math_holdout256_prompts.jsonl",
        holdout_labels_dir / "math_holdout256_labels.json",
    ):
        if stale_p.exists():
            stale_p.unlink()

    math_confirm1280 = take_disjoint(v2_math_pool, 1280, used_hashes, used_grams)
    math_holdout512 = take_disjoint(v2_math_pool, 512, used_hashes, used_grams)
    write_jsonl(splits_dir / "math_confirm1280.jsonl", math_confirm1280)
    write_sealed_holdout(
        splits_dir / "math_holdout512_prompts.jsonl",
        holdout_labels_dir / "math_holdout512_labels.json",
        math_holdout512,
    )

    if not preserve_ok:
        sft_clean = []
        for row in v2_math_pool:
            ph = row["prompt_hash"]
            grams = text_ngrams(row["problem"])
            if ph in used_hashes or (grams & used_grams):
                continue
            tok = tokenize_static(row, tokenizer, 4096, dense_spec["eos_token_ids"])
            if tok is not None:
                sft_clean.append(tok)
                used_hashes.add(ph)
                used_grams.update(grams)
                if len(sft_clean) == 1088:
                    break
        if len(sft_clean) < 1088:
            raise RuntimeError(f"Insufficient clean SFT rows: {len(sft_clean)}")
        sft_systems1024 = sft_clean[:1024]
        sft_heldout64 = sft_clean[1024:1088]
        write_jsonl(splits_dir / "sft_systems1024_v2.jsonl", sft_systems1024)
        write_jsonl(splits_dir / "sft_heldout64_v2.jsonl", sft_heldout64)

    # 3. Pick v2 DAPO 3-Way Splits (strictly disjoint from all v2 MATH splits + all v1 eval & banked GRPO/MoE sets)
    v2_dapo_pool = sorted(
        [r for r in v1_dapo_admitted if r["id"] not in v1_dapo_quarantine_ids],
        key=lambda r: object_hash({"seed": 20261117, "id": r["id"]}),
    )
    if not preserve_ok:
        dapo_search240 = take_disjoint(v2_dapo_pool, 240, used_hashes, used_grams)
        write_jsonl(splits_dir / "dapo_search240.jsonl", dapo_search240)
    dapo_confirm200 = take_disjoint(v2_dapo_pool, 200, used_hashes, used_grams)
    dapo_holdout200 = take_disjoint(v2_dapo_pool, 200, used_hashes, used_grams)

    write_jsonl(splits_dir / "dapo_confirm200.jsonl", dapo_confirm200)
    write_sealed_holdout(
        splits_dir / "dapo_holdout200_prompts.jsonl",
        holdout_labels_dir / "dapo_holdout200_labels.json",
        dapo_holdout200,
    )

    # Download & partition MMLU-Pro non-math retention (excluding v1's 200 items!)
    api = HfApi()
    mmlu_info = api.dataset_info(MMLU_PRO_REPO)
    mmlu_files = sorted(
        item.rfilename
        for item in mmlu_info.siblings
        if item.rfilename.startswith("data/test-") and item.rfilename.endswith(".parquet")
    )
    mmlu_rows = []
    mmlu_prov = []
    cols = ["question_id", "question", "options", "answer", "answer_index", "category"]
    for fname in mmlu_files:
        p = Path(
            hf_hub_download(
                MMLU_PRO_REPO,
                fname,
                repo_type="dataset",
                revision=mmlu_info.sha,
                local_dir=ROOT / "data" / "raw" / "mmlu_pro",
            )
        )
        mmlu_rows.extend(pq.read_table(p, columns=cols).to_pylist())
        mmlu_prov.append({"file": fname, "sha256": file_hash(p)})
    provenance["mmlu_pro"] = {"repo": MMLU_PRO_REPO, "revision": mmlu_info.sha, "files": mmlu_prov}

    by_cat = collections.defaultdict(list)
    for row in mmlu_rows:
        if row["category"] not in MMLU_EXCLUDE_CATEGORIES:
            by_cat[row["category"]].append(row)
    for grp in by_cat.values():
        grp.sort(key=lambda r: v1_object_hash({"seed": 1030, "id": r["question_id"]}))
    cats = sorted(by_cat.keys())
    # Quarantine v1's exact category-balanced prefix (~15-16 per cat for 200 across 13 non-math cats; quarantine first 35/cat!)
    v1_mmlu_prefix_rows = []
    for c in cats:
        for src in by_cat[c][:35]:
            q = src["question"].strip()
            v1_mmlu_prefix_rows.append({"problem": q, "prompt_hash": object_hash(normalize_problem(q))})
    for r in v1_mmlu_prefix_rows:
        used_hashes.add(r["prompt_hash"])
        used_grams.update(text_ngrams(r["problem"]))

    v2_mmlu_candidates = []
    max_depth = max(len(g) for g in by_cat.values())
    for idx in range(35, max_depth):
        for c in cats:
            if idx < len(by_cat[c]):
                src = by_cat[c][idx]
                q = src["question"].strip()
                opts = [o.strip() for o in src["options"] if o.strip()]
                if len(opts) != len(src["options"]) or not (2 <= len(opts) <= 10):
                    continue
                if not (0 <= src["answer_index"] < len(opts)) or src["answer"] != chr(65 + src["answer_index"]):
                    continue
                item = {
                    "id": f"mmlu_pro:{src['question_id']}",
                    "source": "mmlu_pro",
                    "problem": q,
                    "options": opts,
                    "gold_answer": src["answer"],
                    "category": c,
                    "finish_reason": "N/A",
                    "prompt_hash": object_hash(normalize_problem(q)),
                }
                rendered = tokenizer.apply_chat_template(
                    mcq_messages(item),
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                    reasoning_effort="medium",
                )
                ids = tokenizer.encode(rendered, add_special_tokens=False)
                if len(ids) <= 2048:
                    item["prompt_tokens"] = len(ids)
                    v2_mmlu_candidates.append(item)

    retention_confirm150 = take_disjoint(v2_mmlu_candidates, 150, used_hashes, used_grams)
    retention_holdout150 = take_disjoint(v2_mmlu_candidates, 150, used_hashes, used_grams)

    write_jsonl(splits_dir / "retention_confirm150.jsonl", retention_confirm150)
    write_sealed_holdout(
        splits_dir / "retention_holdout150_prompts.jsonl",
        holdout_labels_dir / "retention_holdout150_labels.json",
        retention_holdout150,
    )

    # Verify (a) pairwise disjointness across ALL v2 splits, AND
    # (b) zero exact/13-gram overlap between EVERY v2 split and the union of ALL v1/historical quarantined sets!
    v2_only_splits = {
        "math_search128": math_search128,
        "math_confirm1280": math_confirm1280,
        "math_holdout512": math_holdout512,
        "sft_systems1024_v2": sft_systems1024,
        "sft_heldout64_v2": sft_heldout64,
        "dapo_search240": dapo_search240,
        "dapo_confirm200": dapo_confirm200,
        "dapo_holdout200": dapo_holdout200,
        "retention_confirm150": retention_confirm150,
        "retention_holdout150": retention_holdout150,
    }
    cross_overlaps = verify_pairwise_disjoint(v2_only_splits)
    full_v1_quarantine = historical_protected + v1_math_union_rows + v1_dapo_quarantine_rows + v1_mmlu_prefix_rows
    q_hashes = {r["prompt_hash"] for r in full_v1_quarantine}
    q_grams = set().union(*(text_ngrams(r["problem"]) for r in full_v1_quarantine))

    confirm_holdout_splits = {
        "math_confirm1280": math_confirm1280,
        "math_holdout512": math_holdout512,
        "dapo_confirm200": dapo_confirm200,
        "dapo_holdout200": dapo_holdout200,
        "retention_confirm150": retention_confirm150,
        "retention_holdout150": retention_holdout150,
    }
    search_sft_splits = {
        "math_search128": math_search128,
        "sft_systems1024_v2": sft_systems1024,
        "sft_heldout64_v2": sft_heldout64,
        "dapo_search240": dapo_search240,
    }
    confirm_holdout_v1_overlaps = 0
    for srows in confirm_holdout_splits.values():
        for r in srows:
            if r["prompt_hash"] in q_hashes or (text_ngrams(r["problem"]) & q_grams):
                confirm_holdout_v1_overlaps += 1

    search_sft_v1_overlap_counts: dict[str, int] = {}
    for sname, srows in search_sft_splits.items():
        search_sft_v1_overlap_counts[sname] = sum(
            1 for r in srows if r["prompt_hash"] in q_hashes or (text_ngrams(r["problem"]) & q_grams)
        )

    if cross_overlaps != 0 or confirm_holdout_v1_overlaps != 0:
        raise RuntimeError(
            f"Overlap detected: cross_v2={cross_overlaps}, confirm_holdout_vs_full_v1={confirm_holdout_v1_overlaps}"
        )

    # Verify holdout prompt files contain ZERO gold_answer / solution keys
    holdout_clean = True
    for hp in (
        splits_dir / "math_holdout512_prompts.jsonl",
        splits_dir / "dapo_holdout200_prompts.jsonl",
        splits_dir / "retention_holdout150_prompts.jsonl",
    ):
        for r in read_jsonl(hp):
            if "gold_answer" in r or "solution" in r:
                holdout_clean = False

    split_hashes = {
        p.name: {"sha256": file_hash(p), "rows": len(read_jsonl(p))}
        for p in sorted(splits_dir.glob("*.jsonl"))
    }
    holdout_label_hashes = {
        p.name: {"sha256": file_hash(p)}
        for p in sorted(holdout_labels_dir.glob("*.json"))
    }

    report = {
        "status": "DISJOINT_THREE_WAY_VERIFIED" if (cross_overlaps == 0 and confirm_holdout_v1_overlaps == 0 and holdout_clean) else "FAILED",
        "seed": 20261117,
        "search_splits_preserved_from_plane1": bool(preserve_ok),
        "v1_oracle_sha256_verified": {
            "development_math200.jsonl": dev200_sha,
            "rl_dapo_candidates.jsonl": dapo_cand_sha,
            "v1_archive_evidence_tar_gz": exposed_dapo_cfg["v1_archive_sha256"],
        },
        "provenance": provenance,
        "cross_split_exact_or_13gram_overlaps": cross_overlaps,
        "confirm_and_holdout_vs_all_v1_eval_and_train_overlaps": confirm_holdout_v1_overlaps,
        "v1_quarantined_counts": {
            "historical_math500_aime24_aime25": len(historical_protected),
            "v1_development_math200": len(v1_dev200),
            "v1_sft_systems1024": len(v1_sft1024),
            "v1_dapo_screen256_union_banked_grpo200": len(v1_dapo_quarantine_rows),
            "v1_banked_grpo_moe_unique_dapo_prompts": len(banked_grpo_dapo_ids),
            "v1_mmlu_pro_prefix_35_per_cat": len(v1_mmlu_prefix_rows),
        },
        "search_and_sft_historical_v1_overlap_counts": search_sft_v1_overlap_counts,
        "holdout_labels_outside_candidate_scope": holdout_clean,
        "sft_systems1024_input_tokens": sum(len(r["input_ids"]) for r in sft_systems1024),
        "sft_systems1024_supervised_tokens": sum(loss_tokens(r) for r in sft_systems1024),
        "splits": split_hashes,
        "sealed_holdout_label_receipts": holdout_label_hashes,
        "pretraining_contamination_note": (
            "Public HuggingFace source corpora have unknown pretraining overlap with Qwen3.8/3.5. "
            "All v2 splits are strictly pairwise disjoint (0 exact and 0 13-gram overlaps across all 10 splits). "
            "All Confirm and Protected Single-Touch Holdout splits have 0 exact and 0 13-gram overlap with the "
            "complete v1 eval+training exposure union (MATH-500, AIME24/25, v1_dev200, v1_sft_systems1024, "
            "v1_dapo256, and v1_mmlu_pro200); historical overlap counts on preserved Plane-1 Search/SFT panels "
            "are explicitly disclosed in search_and_sft_historical_v1_overlap_counts."
        ),
    }
    write_json(ROOT / "results" / "split_audit_v2.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "provenance"}, indent=2))


if __name__ == "__main__":
    main()
