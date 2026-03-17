"""
Caption loading from annotation.hdf5 for Xperience-10M.
"""

import json
from pathlib import Path

import h5py


def _find_nearest_frame_index(timestamp_int, name_to_index):
    """Find frame index whose name (as timestamp) is nearest to timestamp_int."""
    best_idx = -1
    best_diff = float("inf")
    for name, idx in name_to_index.items():
        stem = name.rsplit(".", 1)[0] if "." in name else name
        try:
            ts = int(stem)
        except ValueError:
            continue
        diff = abs(ts - timestamp_int)
        if diff < best_diff:
            best_diff = diff
            best_idx = idx
    return best_idx


def _build_frame_info_map_from_caption(data, name_to_index, N):
    """Build frame_info_map and segment_boundaries from caption segments."""
    frame_info_map = {}
    segments = data.get("segments", [])
    segment_boundaries = []
    for seg in segments:
        theme = seg.get("Sub Task", "")
        start_frame = seg.get("start_frame", 0)
        end_frame = seg.get("end_frame", 0)
        seg_id = seg.get("segment_id", 0)
        actions = seg.get("Current Action", [])
        action_ranges = []
        for action in actions:
            start_name = action.get("start_frame_name")
            end_name = action.get("end_frame_name")
            if start_name is None and action.get("start_frame") is not None:
                start_name = str(action["start_frame"])
            if end_name is None and action.get("end_frame") is not None:
                end_name = str(action["end_frame"])
            label = action.get("label", "")
            desc = action.get("description", "")
            if start_name and end_name:
                si = name_to_index.get(start_name, -1)
                ei = name_to_index.get(end_name, -1)
                if (si < 0 or ei < 0) and start_name.isdigit():
                    si = _find_nearest_frame_index(int(start_name), name_to_index) if si < 0 else si
                    ei = _find_nearest_frame_index(int(end_name), name_to_index) if ei < 0 else ei
                if si >= 0 and ei >= 0:
                    action_ranges.append((si, ei, label, desc))
        action_ranges.sort(key=lambda x: x[0])
        indices_in_segment = set()
        for start_idx, end_idx, label, desc in action_ranges:
            for idx in range(start_idx, min(end_idx + 1, N)):
                indices_in_segment.add(idx)
                if idx not in frame_info_map:
                    frame_info_map[idx] = {}
                frame_info_map[idx]["theme"] = theme
                frame_info_map[idx]["action_label"] = label
                frame_info_map[idx]["action_desc"] = desc
        objects_map = seg.get("objects", {})
        interaction_map = seg.get("interaction", {})
        for fname in set(objects_map.keys()) | set(interaction_map.keys()):
            idx = name_to_index.get(fname, -1)
            if idx < 0 and fname.isdigit():
                idx = _find_nearest_frame_index(int(fname), name_to_index)
            if idx < 0:
                continue
            indices_in_segment.add(idx)
            if idx not in frame_info_map:
                frame_info_map[idx] = {}
            if "theme" not in frame_info_map[idx]:
                frame_info_map[idx]["theme"] = theme
            if fname in objects_map:
                frame_info_map[idx]["objects"] = objects_map[fname]
            if fname in interaction_map:
                frame_info_map[idx]["interaction"] = interaction_map[fname]
        if indices_in_segment:
            seg_start = seg_end = -1
            try:
                sf, ef = int(start_frame), int(end_frame)
                if 0 <= sf < N and 0 <= ef < N:
                    seg_start = sf
                    seg_end = min(ef, N - 1)
                    seg_start = min(seg_start, seg_end)
            except (TypeError, ValueError):
                pass
            if seg_start < 0 or seg_end < 0:
                seg_start_key, seg_end_key = str(start_frame), str(end_frame)
                seg_start = name_to_index.get(seg_start_key, -1) if seg_start < 0 else seg_start
                seg_end = name_to_index.get(seg_end_key, -1) if seg_end < 0 else seg_end
                if seg_start < 0 and seg_start_key.isdigit():
                    seg_start = _find_nearest_frame_index(int(seg_start_key), name_to_index)
                if seg_end < 0 and seg_end_key.isdigit():
                    seg_end = _find_nearest_frame_index(int(seg_end_key), name_to_index)
            if seg_start < 0:
                seg_start = min(indices_in_segment)
            if seg_end < 0:
                seg_end = max(indices_in_segment)
            segment_boundaries.append((seg_start, seg_end, theme, seg_id))
            for idx in range(seg_start, seg_end + 1):
                if idx not in frame_info_map:
                    frame_info_map[idx] = {}
                if "theme" not in frame_info_map[idx]:
                    frame_info_map[idx]["theme"] = theme
                if not frame_info_map[idx].get("action_label"):
                    prev = None
                    for si, ei, label, desc in action_ranges:
                        if ei < idx:
                            prev = (ei, label, desc)
                        elif si > idx:
                            break
                    if prev:
                        _, label, desc = prev
                        frame_info_map[idx]["action_label"] = label
                        frame_info_map[idx]["action_desc"] = desc
                    elif action_ranges:
                        _, _, label, desc = action_ranges[0]
                        frame_info_map[idx]["action_label"] = label
                        frame_info_map[idx]["action_desc"] = desc
            frames_with_objects = sorted([name_to_index[f] for f in objects_map.keys() if f in name_to_index])
            frames_with_interaction = sorted([name_to_index[f] for f in interaction_map.keys() if f in name_to_index])
            for idx in range(seg_start, seg_end + 1):
                if idx not in frame_info_map:
                    frame_info_map[idx] = {}
                if frames_with_objects:
                    prev_o = max([f for f in frames_with_objects if f <= idx], default=None)
                    if prev_o is not None and "objects" not in frame_info_map[idx]:
                        frame_info_map[idx]["objects"] = frame_info_map[prev_o]["objects"]
                if frames_with_interaction:
                    prev_i = max([f for f in frames_with_interaction if f <= idx], default=None)
                    if prev_i is not None and "interaction" not in frame_info_map[idx]:
                        frame_info_map[idx]["interaction"] = frame_info_map[prev_i]["interaction"]
    for idx in range(N):
        if idx not in frame_info_map:
            frame_info_map[idx] = {}
    for idx in range(1, N):
        prev = frame_info_map.get(idx - 1, {})
        for key in ("theme", "action_label", "action_desc"):
            if key not in frame_info_map[idx] and key in prev:
                frame_info_map[idx][key] = prev[key]
    return frame_info_map, segment_boundaries


def load_caption_data_from_annotation_hdf5(annotation_path, data_root, img_names):
    """
    Load caption data from annotation.hdf5 only (dataset 'caption' or 'captions').
    Returns (main_task, frame_info_map, segment_boundaries, task_to_id).
    If no data found, returns ("", None, [], {}).
    """
    data = None
    data_root_path = Path(data_root)
    try:
        with h5py.File(annotation_path, "r") as f:
            for key in ("caption", "captions"):
                if key not in f:
                    continue
                raw = f[key][...]
                if hasattr(raw, "ndim") and getattr(raw, "ndim", -1) == 0 and hasattr(raw, "item"):
                    raw = raw.item()
                elif hasattr(raw, "size") and getattr(raw, "size", 0) == 1 and hasattr(raw, "item"):
                    raw = raw.item()
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="replace")
                elif not isinstance(raw, str) and hasattr(raw, "tobytes"):
                    raw = raw.tobytes().decode("utf-8", errors="replace")
                elif not isinstance(raw, str):
                    raw = str(raw)
                raw = raw.strip() if isinstance(raw, str) else ""
                if not raw:
                    continue
                if raw.startswith("{") or raw.startswith("["):
                    data = json.loads(raw)
                    break
                path_candidate = Path(raw)
                if not path_candidate.is_absolute():
                    path_candidate = data_root_path / path_candidate
                if path_candidate.exists():
                    with open(path_candidate, "r", encoding="utf-8") as fp:
                        data = json.load(fp)
                    break
                try:
                    data = json.loads(raw)
                    break
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    if data is None:
        return "", None, [], {}
    main_task = data.get("config", {}).get("Main Task", "N/A")
    N = len(img_names)
    name_to_index = {name: i for i, name in enumerate(img_names)}
    for i, name in enumerate(img_names):
        stem = name.rsplit(".", 1)[0] if "." in name else name
        if stem not in name_to_index:
            name_to_index[stem] = i
    frame_info_map, segment_boundaries = _build_frame_info_map_from_caption(data, name_to_index, N)
    unique_tasks = []
    for _, _, task_name, _ in segment_boundaries:
        if task_name not in unique_tasks:
            unique_tasks.append(task_name)
    task_to_id = {name: idx + 1 for idx, name in enumerate(unique_tasks)}
    return main_task, frame_info_map, segment_boundaries, task_to_id


__all__ = ["load_caption_data_from_annotation_hdf5"]
