"""Dataset inspection and indexing for binary ore classification."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import load_config
from .utils import (
    CLASS_TO_IDX,
    dhash_image,
    extract_group_id,
    infer_magnification,
    is_image_path,
    load_manual_group_map,
    read_image_size,
    sha256_file,
    write_json,
    write_rows_csv,
)


def _source_and_class(root: Path, path: Path) -> tuple[str, str]:
    rel = path.relative_to(root)
    parts = rel.parts
    if not parts:
        return "unknown", "unknown"
    source = parts[0]
    if source in {"set1", "set2"} and len(parts) > 1:
        return source, parts[1]
    return source, source


def _map_target(source_class: str, mapping: dict[str, str], exclude_talc: bool) -> tuple[str | None, str]:
    value = mapping.get(source_class)
    if value is None:
        value = mapping.get(source_class.lower())
    if value in CLASS_TO_IDX:
        return value, ""
    class_lower = source_class.lower()
    if value == "exclude_talc" or ("отальк" in class_lower and exclude_talc):
        return None, "talc_excluded"
    if value and value.startswith("exclude"):
        return None, value
    return None, "unmapped_or_non_binary"


def apply_duplicate_group_ids(rows: list[dict[str, Any]], group_perceptual_duplicates: bool = True) -> dict[str, list[int]]:
    sha_to_rows: dict[str, list[int]] = defaultdict(list)
    dhash_to_rows: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        sha = row.get("sha256")
        if sha:
            sha_to_rows[str(sha)].append(index)
        dhash = row.get("dhash")
        if group_perceptual_duplicates and dhash and row.get("target") in CLASS_TO_IDX:
            dhash_to_rows[str(dhash)].append(index)

    exact_duplicates = {sha: indices for sha, indices in sha_to_rows.items() if len(indices) > 1}
    for sha, indices in exact_duplicates.items():
        duplicate_group = f"sha256:{sha[:16]}"
        for row_index in indices:
            rows[row_index]["group_id"] = duplicate_group

    if group_perceptual_duplicates:
        perceptual_duplicates = {dhash: indices for dhash, indices in dhash_to_rows.items() if len(indices) > 1}
        for dhash, indices in perceptual_duplicates.items():
            duplicate_group = f"dhash:{dhash}"
            for row_index in indices:
                rows[row_index]["group_id"] = duplicate_group
    return exact_duplicates


def build_dataset_index(config: dict[str, Any], compute_dhash: bool = True) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data_config = config["data"]
    root = Path(data_config["root"]).expanduser()
    mapping = data_config.get("class_mapping", {})
    manual_groups = load_manual_group_map(data_config.get("manual_group_csv"), root)
    image_paths = sorted(path for path in root.rglob("*") if is_image_path(path))

    rows: list[dict[str, Any]] = []
    for path in image_paths:
        rel_path = str(path.relative_to(root))
        source, source_class = _source_and_class(root, path)
        target, excluded_reason = _map_target(source_class, mapping, bool(data_config.get("exclude_talc", True)))
        width, height, read_error = read_image_size(path)
        readable = read_error is None
        sha = sha256_file(path)
        manual_group = manual_groups.get(str(path)) or manual_groups.get(rel_path)
        raw_group = manual_group or extract_group_id(path)
        rows.append(
            {
                "file_path": str(path),
                "rel_path": rel_path,
                "dataset_source": source,
                "source_class": source_class,
                "target": target or "",
                "target_index": CLASS_TO_IDX[target] if target else "",
                "excluded_reason": excluded_reason,
                "readable": readable,
                "read_error": read_error or "",
                "extension": path.suffix.lower(),
                "width": width or "",
                "height": height or "",
                "resolution": f"{width}x{height}" if width and height else "",
                "magnification": infer_magnification(path),
                "sha256": sha,
                "dhash": dhash_image(path) if compute_dhash and readable and target in CLASS_TO_IDX else "",
                "group_id_raw": raw_group,
                "group_id": f"{source}:{raw_group}",
            }
        )

    exact_duplicates = apply_duplicate_group_ids(
        rows,
        group_perceptual_duplicates=bool(data_config.get("group_perceptual_duplicates", True)),
    )

    summary = summarize_rows(rows, exact_duplicates)
    return rows, summary


def summarize_rows(rows: list[dict[str, Any]], exact_duplicates: dict[str, list[int]] | None = None) -> dict[str, Any]:
    exact_duplicates = exact_duplicates or {}
    included = [row for row in rows if row["target"] in CLASS_TO_IDX and row["readable"]]
    excluded = [row for row in rows if row["target"] not in CLASS_TO_IDX]
    unreadable = [row for row in rows if not row["readable"]]

    source_class_counts = Counter((row["dataset_source"], row["source_class"]) for row in rows)
    class_by_source = Counter((row["dataset_source"], row["target"]) for row in included)
    target_counts = Counter(row["target"] for row in included)
    excluded_counts = Counter(row["excluded_reason"] for row in excluded)
    extension_counts = Counter(row["extension"] for row in rows)
    resolution_counts = Counter(row["resolution"] for row in rows if row["resolution"])
    magnification_counts = Counter(row["magnification"] for row in rows)
    group_counts = Counter(row["group_id"] for row in included)

    dhash_buckets: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row.get("dhash"):
            dhash_buckets[row["dhash"]].append(row["rel_path"])
    perceptual_duplicate_buckets = {key: value for key, value in dhash_buckets.items() if len(value) > 1}

    duplicate_examples = []
    for sha, indices in list(exact_duplicates.items())[:100]:
        duplicate_examples.append(
            {
                "sha256": sha,
                "files": [rows[index]["rel_path"] for index in indices],
                "targets": sorted({rows[index]["target"] for index in indices if rows[index]["target"]}),
                "sources": sorted({rows[index]["dataset_source"] for index in indices}),
            }
        )

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "total_image_files": len(rows),
        "included_binary_images": len(included),
        "excluded_images": len(excluded),
        "unreadable_images": len(unreadable),
        "source_class_counts": {f"{source}/{klass}": count for (source, klass), count in source_class_counts.items()},
        "target_counts": dict(target_counts),
        "excluded_counts": dict(excluded_counts),
        "class_distribution_by_dataset_source": {
            f"{source}/{target}": count for (source, target), count in class_by_source.items()
        },
        "extension_counts": dict(extension_counts),
        "top_resolutions": dict(resolution_counts.most_common(30)),
        "magnification_counts": dict(magnification_counts),
        "unique_groups": len(group_counts),
        "groups_with_multiple_images": sum(1 for count in group_counts.values() if count > 1),
        "exact_duplicate_groups": len(exact_duplicates),
        "exact_duplicate_examples": duplicate_examples,
        "perceptual_duplicate_buckets": len(perceptual_duplicate_buckets),
        "perceptual_duplicate_examples": dict(list(perceptual_duplicate_buckets.items())[:50]),
        "unreadable_examples": [
            {"rel_path": row["rel_path"], "error": row["read_error"]} for row in unreadable[:50]
        ],
    }


def inspect_dataset(config_path: str | Path | None = None, output_dir: str | Path | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    base_output = Path(output_dir or config["data"].get("output_dir") or "runs/ore_classifier")
    report_dir = base_output / "inspection"
    rows, summary = build_dataset_index(config, compute_dhash=True)
    write_rows_csv(rows, report_dir / "dataset_index.csv")
    write_json(summary, report_dir / "dataset_summary.json")
    write_json(config["data"].get("class_mapping", {}), report_dir / "class_mapping.json")
    return {"rows": rows, "summary": summary, "report_dir": str(report_dir)}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Inspect ore-classification dataset structure.")
    parser.add_argument("--config", default="configs/classifier.yaml")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    result = inspect_dataset(args.config, args.output_dir)
    summary = result["summary"]
    print(f"Report: {result['report_dir']}")
    print(f"Images: {summary['total_image_files']}")
    print(f"Included binary: {summary['included_binary_images']}")
    print(f"Excluded: {summary['excluded_images']} ({summary['excluded_counts']})")
    print(f"Class by source: {summary['class_distribution_by_dataset_source']}")
    print(f"Exact duplicate groups: {summary['exact_duplicate_groups']}")
    print(f"Perceptual duplicate buckets: {summary['perceptual_duplicate_buckets']}")


if __name__ == "__main__":
    main()
