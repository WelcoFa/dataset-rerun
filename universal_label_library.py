from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


DEFAULT_LIBRARY_NAME = "gigahands_universal_labels"

SEED_LABELS = [
    {
        "id": "approach",
        "name": "approach",
        "aliases": ["reach", "reaching", "reaches"],
        "status": "seeded",
    },
    {
        "id": "touch",
        "name": "touch",
        "aliases": ["contact", "touches"],
        "status": "seeded",
    },
    {
        "id": "grasp",
        "name": "grasp",
        "aliases": ["grab", "grabbing", "grasps", "pick up"],
        "status": "seeded",
    },
    {
        "id": "hold",
        "name": "hold",
        "aliases": ["holding", "holds"],
        "status": "seeded",
    },
    {
        "id": "lift",
        "name": "lift",
        "aliases": ["lifting", "lifts", "raise", "raises"],
        "status": "seeded",
    },
    {
        "id": "move",
        "name": "move",
        "aliases": ["moving", "moves", "pull", "pulling", "slide", "sliding"],
        "status": "seeded",
    },
    {
        "id": "place",
        "name": "place",
        "aliases": ["places", "put down", "set down"],
        "status": "seeded",
    },
    {
        "id": "release",
        "name": "release",
        "aliases": ["let go", "releases"],
        "status": "seeded",
    },
    {
        "id": "manipulate",
        "name": "manipulate",
        "aliases": ["adjust", "close", "open", "pour", "rotate", "rotating"],
        "status": "seeded",
    },
    {
        "id": "other",
        "name": "other",
        "aliases": [],
        "status": "seeded",
    },
]

SEMANTIC_GROUPS = {
    "approach": {"approach", "reach", "reaching", "reaches"},
    "touch": {"contact", "tap", "touch", "touches"},
    "grasp": {"grab", "grabbing", "grasp", "grasps", "pick", "pickup", "pick up"},
    "hold": {"hold", "holding", "holds", "support"},
    "lift": {"elevate", "lift", "lifting", "lifts", "raise", "raising"},
    "move": {"carry", "drag", "move", "moves", "moving", "pull", "pulling", "push", "slide", "sliding"},
    "place": {"place", "placing", "put", "put down", "set down"},
    "release": {"drop", "let go", "release", "releasing"},
    "manipulate": {"adjust", "close", "manipulate", "open", "pour", "rotate", "rotating", "turn", "twist"},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_token(value: str) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def slugify_label_id(value: str) -> str:
    normalized = normalize_token(value)
    return normalized.replace(" ", "_")


def default_label_library() -> dict[str, Any]:
    return {
        "version": 1,
        "library_name": DEFAULT_LIBRARY_NAME,
        "updated_at": utc_now(),
        "labels": [dict(item) for item in SEED_LABELS],
        "discoveries": [],
        "observations": [],
    }


def _normalize_label_entry(entry: Any) -> dict[str, Any] | None:
    if isinstance(entry, str):
        name = normalize_token(entry)
        if not name:
            return None
        return {
            "id": slugify_label_id(name),
            "name": name,
            "aliases": [],
            "status": "seeded",
        }

    if not isinstance(entry, dict):
        return None

    raw_name = entry.get("name") or entry.get("label") or entry.get("id")
    name = normalize_token(raw_name)
    if not name:
        return None

    aliases: list[str] = []
    for alias in entry.get("aliases", []):
        normalized_alias = normalize_token(alias)
        if normalized_alias and normalized_alias != name and normalized_alias not in aliases:
            aliases.append(normalized_alias)

    return {
        "id": str(entry.get("id") or slugify_label_id(name)),
        "name": name,
        "aliases": aliases,
        "status": entry.get("status", "seeded"),
        "description": str(entry.get("description", "")).strip(),
        "discovered_at": entry.get("discovered_at"),
        "discovered_from_model": entry.get("discovered_from_model"),
        "evaluation": dict(entry.get("evaluation", {})) if isinstance(entry.get("evaluation", {}), dict) else {},
    }


def _normalize_library_data(data: dict[str, Any], path: Path) -> dict[str, Any]:
    raw_labels = data.get("labels")
    if raw_labels is None:
        raw_labels = data.get("canonical_labels", [])

    normalized_labels: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    for entry in raw_labels:
        normalized_entry = _normalize_label_entry(entry)
        if normalized_entry is None:
            continue
        if normalized_entry["name"] in seen_names:
            continue
        normalized_labels.append(normalized_entry)
        seen_names.add(normalized_entry["name"])

    if not normalized_labels:
        normalized_labels = [dict(item) for item in SEED_LABELS]

    return {
        "version": int(data.get("version", 1)),
        "library_name": str(data.get("library_name", DEFAULT_LIBRARY_NAME)),
        "updated_at": data.get("updated_at") or utc_now(),
        "path": str(path),
        "labels": normalized_labels,
        "discoveries": list(data.get("discoveries", [])),
        "observations": list(data.get("observations", [])),
    }


def _storage_payload(library: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": int(library.get("version", 1)),
        "library_name": str(library.get("library_name", DEFAULT_LIBRARY_NAME)),
        "updated_at": library.get("updated_at") or utc_now(),
        "labels": library.get("labels", []),
        "discoveries": library.get("discoveries", []),
        "observations": library.get("observations", []),
        "evaluation": library.get("evaluation", {}),
    }


def ensure_label_library_exists(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(default_label_library(), f, ensure_ascii=False, indent=2)


def load_label_library(path: Path) -> dict[str, Any]:
    ensure_label_library_exists(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return _normalize_library_data(data, path)


def save_label_library(library: dict[str, Any], path: Path) -> None:
    recompute_library_evaluations(library)
    library["updated_at"] = utc_now()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_storage_payload(library), f, ensure_ascii=False, indent=2)


def get_canonical_labels(library: dict[str, Any] | None) -> list[str]:
    if not library:
        return []
    return [str(item["name"]) for item in library.get("labels", [])]


def library_size(library: dict[str, Any] | None) -> int:
    return len(get_canonical_labels(library))


def build_prompt_label_block(library: dict[str, Any] | None, max_labels: int = 24) -> str:
    if not library:
        return ""

    lines: list[str] = []
    for entry in library.get("labels", [])[: max(1, max_labels)]:
        aliases = entry.get("aliases", [])
        if aliases:
            alias_preview = ", ".join(aliases[:4])
            lines.append(f"- {entry['name']} (aliases: {alias_preview})")
        else:
            lines.append(f"- {entry['name']}")
    return "\n".join(lines)


def _expanded_terms(entry: dict[str, Any]) -> set[str]:
    terms = {normalize_token(entry.get("name", ""))}
    for alias in entry.get("aliases", []):
        normalized_alias = normalize_token(alias)
        if normalized_alias:
            terms.add(normalized_alias)
    return {term for term in terms if term}


def _concept_keys(value: str) -> set[str]:
    normalized = normalize_token(value)
    if not normalized:
        return set()

    keys = {normalized}
    compact = normalized.replace(" ", "")
    if compact:
        keys.add(compact)

    singular = normalized[:-1] if normalized.endswith("s") and len(normalized) > 3 else normalized
    if singular:
        keys.add(singular)

    for concept, members in SEMANTIC_GROUPS.items():
        if normalized in members or compact in {member.replace(" ", "") for member in members}:
            keys.add(concept)

    return keys


def _term_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _best_library_match(raw_label: str, library: dict[str, Any]) -> tuple[dict[str, Any] | None, str, float]:
    normalized_raw = normalize_token(raw_label)
    if not normalized_raw:
        return None, "empty", 0.0

    raw_concepts = _concept_keys(normalized_raw)
    best_entry: dict[str, Any] | None = None
    best_kind = "new_label"
    best_score = 0.0

    for entry in library.get("labels", []):
        entry_name = normalize_token(entry.get("name", ""))
        expanded_terms = _expanded_terms(entry)

        if normalized_raw == entry_name:
            return entry, "exact_name", 1.0
        if normalized_raw in expanded_terms:
            return entry, "exact_alias", 0.99

        entry_concepts = set()
        for term in expanded_terms:
            entry_concepts.update(_concept_keys(term))

        if raw_concepts & entry_concepts:
            score = 0.94
            if score > best_score:
                best_entry = entry
                best_kind = "semantic_alias"
                best_score = score

        for term in expanded_terms:
            similarity = _term_similarity(normalized_raw, term)
            if similarity > best_score:
                best_entry = entry
                best_kind = "similarity"
                best_score = similarity

    return best_entry, best_kind, best_score


def _append_new_label(library: dict[str, Any], raw_label: str, discovered_from_model: str | None = None) -> dict[str, Any]:
    normalized_raw = normalize_token(raw_label)
    existing_names = set(get_canonical_labels(library))
    if normalized_raw in existing_names:
        for entry in library.get("labels", []):
            if entry["name"] == normalized_raw:
                return entry

    new_entry = {
        "id": slugify_label_id(normalized_raw),
        "name": normalized_raw,
        "aliases": [],
        "status": "discovered",
        "description": "",
        "discovered_at": utc_now(),
        "discovered_from_model": discovered_from_model,
        "evaluation": {},
    }
    library.setdefault("labels", []).append(new_entry)
    library["updated_at"] = utc_now()
    return new_entry


def resolve_label(
    raw_label: str,
    library: dict[str, Any] | None,
    *,
    auto_append_new_labels: bool = False,
    similarity_threshold: float = 0.86,
) -> dict[str, Any]:
    normalized_raw = normalize_token(raw_label)
    working_library = library or default_label_library()

    if not normalized_raw:
        fallback = "other" if "other" in get_canonical_labels(working_library) else get_canonical_labels(working_library)[0]
        return {
            "resolved_label": fallback,
            "raw_label": raw_label,
            "raw_label_normalized": normalized_raw,
            "match_kind": "empty",
            "match_score": 0.0,
            "label_id": slugify_label_id(fallback),
            "is_new_label": False,
        }

    matched_entry, match_kind, match_score = _best_library_match(normalized_raw, working_library)
    if matched_entry is not None and (match_kind.startswith("exact") or match_kind == "semantic_alias" or match_score >= similarity_threshold):
        return {
            "resolved_label": matched_entry["name"],
            "raw_label": raw_label,
            "raw_label_normalized": normalized_raw,
            "match_kind": match_kind,
            "match_score": round(match_score, 4),
            "label_id": matched_entry["id"],
            "is_new_label": False,
        }

    if auto_append_new_labels and normalized_raw != "other":
        new_entry = _append_new_label(working_library, normalized_raw)
        return {
            "resolved_label": new_entry["name"],
            "raw_label": raw_label,
            "raw_label_normalized": normalized_raw,
            "match_kind": "new_label",
            "match_score": round(match_score, 4),
            "label_id": new_entry["id"],
            "is_new_label": True,
        }

    return {
        "resolved_label": normalized_raw,
        "raw_label": raw_label,
        "raw_label_normalized": normalized_raw,
        "match_kind": "new_label",
        "match_score": round(match_score, 4),
        "label_id": slugify_label_id(normalized_raw),
        "is_new_label": True,
    }


def record_observation(
    library: dict[str, Any],
    *,
    raw_label: str,
    canonical_label: str,
    run_name: str,
    clip_id: Any,
    scene_name: str,
    raw_response: str,
    sub_task: str,
    interaction: str,
    current_action: str,
    match_kind: str | None,
    auto_append_new_labels: bool = False,
) -> None:
    normalized_raw = normalize_token(raw_label)
    normalized_canonical = normalize_token(canonical_label)

    observation = {
        "recorded_at": utc_now(),
        "run_name": run_name,
        "scene_name": scene_name,
        "clip_id": clip_id,
        "raw_label": normalized_raw,
        "canonical_label": normalized_canonical,
        "match_kind": match_kind,
        "sub_task": str(sub_task or "").strip(),
        "interaction": str(interaction or "").strip(),
        "current_action": str(current_action or "").strip(),
        "raw_response_excerpt": str(raw_response or "").strip()[:400],
    }
    library.setdefault("observations", []).append(observation)

    should_append = (
        auto_append_new_labels
        and match_kind == "new_label"
        and normalized_raw
        and normalized_raw != "other"
    )
    if should_append:
        new_entry = _append_new_label(library, normalized_raw)
        library.setdefault("discoveries", []).append(
            {
                "recorded_at": utc_now(),
                "label_id": new_entry["id"],
                "name": new_entry["name"],
                "run_name": run_name,
                "scene_name": scene_name,
                "clip_id": clip_id,
            }
        )

    library["updated_at"] = utc_now()


def _round(value: float) -> float:
    return round(value, 4)


def _label_quality_score(usage_count: int, match_kind_counts: dict[str, int], unique_raw_variants: int) -> float | None:
    if usage_count <= 0:
        return None

    weighted = (
        match_kind_counts.get("exact_name", 0) * 1.0
        + match_kind_counts.get("exact_alias", 0) * 0.95
        + match_kind_counts.get("semantic_alias", 0) * 0.9
        + match_kind_counts.get("similarity", 0) * 0.75
        + match_kind_counts.get("new_label", 0) * 0.6
    )
    base_score = weighted / usage_count
    if unique_raw_variants > 1:
        base_score *= max(0.6, 1.0 - min(0.3, (unique_raw_variants - 1) * 0.05))
    return _round(base_score)


def recompute_library_evaluations(library: dict[str, Any]) -> dict[str, Any]:
    observations = library.get("observations", [])
    labels = library.get("labels", [])

    known_names = {normalize_token(entry.get("name", "")) for entry in labels}
    evaluation_rows: list[dict[str, Any]] = []
    active_label_count = 0
    discovered_label_count = 0
    unused_label_count = 0

    for entry in labels:
        canonical_name = normalize_token(entry.get("name", ""))
        if not canonical_name:
            continue

        if entry.get("status") == "discovered":
            discovered_label_count += 1

        relevant_obs = [
            obs for obs in observations
            if normalize_token(obs.get("canonical_label", "")) == canonical_name
        ]
        usage_count = len(relevant_obs)
        if usage_count > 0:
            active_label_count += 1
        else:
            unused_label_count += 1

        raw_variant_counts: dict[str, int] = {}
        match_kind_counts: dict[str, int] = {}
        scene_counts: dict[str, int] = {}
        run_counts: dict[str, int] = {}

        for obs in relevant_obs:
            raw_variant = normalize_token(obs.get("raw_label", ""))
            if raw_variant:
                raw_variant_counts[raw_variant] = raw_variant_counts.get(raw_variant, 0) + 1

            match_kind = str(obs.get("match_kind") or "unknown")
            match_kind_counts[match_kind] = match_kind_counts.get(match_kind, 0) + 1

            scene_name = str(obs.get("scene_name") or "").strip()
            if scene_name:
                scene_counts[scene_name] = scene_counts.get(scene_name, 0) + 1

            run_name = str(obs.get("run_name") or "").strip()
            if run_name:
                run_counts[run_name] = run_counts.get(run_name, 0) + 1

        top_raw_variants = sorted(
            raw_variant_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[:5]
        dominant_raw_variant_count = top_raw_variants[0][1] if top_raw_variants else 0
        unique_raw_variants = len(raw_variant_counts)
        raw_variant_consistency = (
            _round(dominant_raw_variant_count / usage_count) if usage_count else None
        )
        exact_match_rate = (
            _round(match_kind_counts.get("exact_name", 0) / usage_count) if usage_count else None
        )
        alias_match_rate = (
            _round(match_kind_counts.get("exact_alias", 0) / usage_count) if usage_count else None
        )
        semantic_match_rate = (
            _round(match_kind_counts.get("semantic_alias", 0) / usage_count) if usage_count else None
        )
        similarity_match_rate = (
            _round(match_kind_counts.get("similarity", 0) / usage_count) if usage_count else None
        )
        label_quality_score = _label_quality_score(usage_count, match_kind_counts, unique_raw_variants)

        needs_review = (
            usage_count > 0 and (
                unique_raw_variants >= 4
                or (similarity_match_rate is not None and similarity_match_rate >= 0.35)
                or (raw_variant_consistency is not None and raw_variant_consistency < 0.6)
            )
        )

        evaluation = {
            "usage_count": usage_count,
            "unique_scene_count": len(scene_counts),
            "unique_run_count": len(run_counts),
            "unique_raw_variant_count": unique_raw_variants,
            "top_raw_variants": [
                {"raw_label": raw_label, "count": count}
                for raw_label, count in top_raw_variants
            ],
            "match_kind_counts": dict(sorted(match_kind_counts.items())),
            "exact_match_rate": exact_match_rate,
            "alias_match_rate": alias_match_rate,
            "semantic_match_rate": semantic_match_rate,
            "similarity_match_rate": similarity_match_rate,
            "raw_variant_consistency": raw_variant_consistency,
            "label_quality_score": label_quality_score,
            "needs_review": needs_review,
        }
        entry["evaluation"] = evaluation
        evaluation_rows.append(
            {
                "name": canonical_name,
                "status": entry.get("status", "seeded"),
                **evaluation,
            }
        )

    discovered_candidates = sorted(
        {
            normalize_token(obs.get("raw_label", ""))
            for obs in observations
            if str(obs.get("match_kind") or "") == "new_label"
            and normalize_token(obs.get("raw_label", ""))
            and normalize_token(obs.get("raw_label", "")) not in known_names
        }
    )

    library["evaluation"] = {
        "recorded_at": utc_now(),
        "total_labels": len(labels),
        "active_label_count": active_label_count,
        "unused_label_count": unused_label_count,
        "discovered_label_count": discovered_label_count,
        "total_observations": len(observations),
        "labels_needing_review": [row["name"] for row in evaluation_rows if row["needs_review"]],
        "discovered_candidates": discovered_candidates,
        "label_rows": evaluation_rows,
    }
    return library["evaluation"]


def summarize_library(library: dict[str, Any] | None) -> dict[str, Any] | None:
    if library is None:
        return None

    evaluation = recompute_library_evaluations(library)
    labels = library.get("labels", [])
    discoveries = library.get("discoveries", [])
    observations = library.get("observations", [])
    discovered_names = [str(item.get("name", "")) for item in discoveries if str(item.get("name", "")).strip()]

    return {
        "library_name": library.get("library_name", DEFAULT_LIBRARY_NAME),
        "total_labels": len(labels),
        "discovered_label_count": len({name for name in discovered_names if name}),
        "observation_count": len(observations),
        "canonical_labels": [entry["name"] for entry in labels],
        "active_label_count": evaluation.get("active_label_count"),
        "unused_label_count": evaluation.get("unused_label_count"),
        "labels_needing_review": evaluation.get("labels_needing_review", []),
        "updated_at": library.get("updated_at"),
    }
