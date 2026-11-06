"""Loading, discovery, and validation for saved continuous-angle embeddings."""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F


MODEL_KEYS = {"disse": "audio_space", "elsa": "audio_elsa"}
CONDITION_RE = re.compile(r"^(?P<room>[^_]+)_t30_(?P<t30>[0-9]+p[0-9]+|[0-9]+)s$")


@dataclass(frozen=True)
class EmbeddingRecord:
    """One source/model/axis/acoustic-condition embedding sweep."""

    source_id: str
    model: str
    axis: str
    room: str
    t30_s: float
    condition: str
    angles_deg: np.ndarray
    embeddings: torch.Tensor
    path: Path
    embedding_key: str
    source_norm_min: float
    source_norm_max: float

    @property
    def n_points(self) -> int:
        return int(self.embeddings.shape[0])

    @property
    def embedding_dim(self) -> int:
        return int(self.embeddings.shape[1])

    @property
    def normalized(self) -> torch.Tensor:
        return F.normalize(self.embeddings.float(), dim=-1)


def parse_condition(condition: str) -> tuple[str, float]:
    match = CONDITION_RE.match(condition)
    if match is None:
        raise ValueError(f"Invalid condition directory name: {condition!r}")
    room = match.group("room")
    t30 = float(match.group("t30").replace("p", "."))
    return room, t30


def _angle_array(obj: dict, axis: str) -> np.ndarray:
    if axis == "azimuth":
        keys = ("azimuth_deg", "angles_deg")
    else:
        keys = ("elevations_deg", "angles_deg")
    key = next((name for name in keys if name in obj and obj[name] is not None), None)
    if key is None:
        raise ValueError(f"{axis}: none of the angle keys {keys} exists")
    values = obj[key]
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError(f"{axis}: angles must be finite one-dimensional values")
    if axis == "azimuth":
        values = values % 360.0
    if len(np.unique(values)) != len(values):
        raise ValueError(f"{axis}: duplicate angles in embedding file")
    return values


def _path_parts(path: Path) -> tuple[str, str, str, str]:
    # .../<source>/<model>/<axis>/<condition>/<embedding-file>
    if len(path.parts) < 5:
        raise ValueError(f"Unexpected embedding path: {path}")
    source_id, model, axis, condition = path.parts[-5:-1]
    if model not in MODEL_KEYS:
        raise ValueError(f"Unexpected model directory {model!r} in {path}")
    if axis not in {"azimuth", "elevation"}:
        raise ValueError(f"Unexpected axis directory {axis!r} in {path}")
    return source_id, model, axis, condition


def load_embedding(path: Path) -> EmbeddingRecord:
    """Load one trusted repository .pt file with safe tensor-only loading."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    source_id, model, axis, condition = _path_parts(path)
    room, t30_s = parse_condition(condition)
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: expected a dictionary")
    saved_axis = obj.get("axis") or obj.get("sweep_name")
    if saved_axis in {"azimuth", "elevation"} and saved_axis != axis:
        raise ValueError(f"{path}: saved axis={saved_axis!r}, directory axis={axis!r}")
    angles = _angle_array(obj, axis)
    available = obj.get("embeddings")
    if not isinstance(available, dict):
        raise ValueError(f"{path}: missing embeddings dictionary")
    key = MODEL_KEYS[model]
    if key not in available:
        raise ValueError(f"{path}: missing embedding key {key!r}; available={list(available)}")
    embeddings = available[key]
    if not torch.is_tensor(embeddings):
        raise ValueError(f"{path}: embeddings[{key!r}] is not a tensor")
    embeddings = embeddings.detach().float().cpu()
    if embeddings.ndim != 2 or embeddings.shape[0] != len(angles) or embeddings.shape[1] < 1:
        raise ValueError(f"{path}: expected [N,D] matching angles, got {tuple(embeddings.shape)}")
    if not torch.isfinite(embeddings).all():
        raise ValueError(f"{path}: embeddings contain NaN or infinity")
    norms = torch.linalg.vector_norm(embeddings, dim=1)
    if torch.any(norms <= 0):
        raise ValueError(f"{path}: zero embedding vector")
    return EmbeddingRecord(
        source_id=source_id,
        model=model,
        axis=axis,
        room=room,
        t30_s=t30_s,
        condition=condition,
        angles_deg=angles,
        embeddings=embeddings,
        path=path,
        embedding_key=key,
        source_norm_min=float(norms.min()),
        source_norm_max=float(norms.max()),
    )


def embedding_filename(model: str, axis: str) -> str:
    if model == "disse":
        return f"continuous_{axis}_embeddings.pt"
    if model == "elsa":
        return "elsa_embeddings.pt"
    raise ValueError(f"Unknown model: {model}")


def discover_embedding_paths(
    input_root: Path,
    *,
    models: Sequence[str] = ("disse", "elsa"),
    axes: Sequence[str] = ("azimuth",),
    rooms: Sequence[str] | None = None,
    t30_s: Sequence[float] | None = None,
) -> list[Path]:
    input_root = input_root.expanduser().resolve()
    room_set = set(rooms) if rooms is not None else None
    t30_set = {round(float(value), 6) for value in t30_s} if t30_s is not None else None
    paths: list[Path] = []
    for model in models:
        for axis in axes:
            filename = embedding_filename(model, axis)
            for path in input_root.glob(f"*/{model}/{axis}/*/{filename}"):
                _, _, _, condition = _path_parts(path)
                room, condition_t30 = parse_condition(condition)
                if room_set is not None and room not in room_set:
                    continue
                if t30_set is not None and round(condition_t30, 6) not in t30_set:
                    continue
                paths.append(path)
    return sorted(paths)


def load_collection(paths: Iterable[Path]) -> list[EmbeddingRecord]:
    records = [load_embedding(path) for path in paths]
    if not records:
        raise ValueError("No embedding files were found")
    seen: set[tuple[str, str, str, str, str]] = set()
    for record in records:
        key = (record.model, record.axis, record.room, record.condition, record.source_id)
        if key in seen:
            raise ValueError(f"Duplicate source {record.source_id} for {key[:-1]}")
        seen.add(key)
    return records


def group_records(
    records: Sequence[EmbeddingRecord],
) -> dict[tuple[str, str, str, str], list[EmbeddingRecord]]:
    groups: dict[tuple[str, str, str, str], list[EmbeddingRecord]] = {}
    for record in records:
        key = (record.model, record.axis, record.room, record.condition)
        groups.setdefault(key, []).append(record)
    for key, group in groups.items():
        group.sort(key=lambda record: record.source_id)
        reference = group[0]
        for record in group[1:]:
            if reference.embedding_dim != record.embedding_dim:
                raise ValueError(f"Embedding dimensions differ in {key}: {reference.path} vs {record.path}")
            if not np.allclose(reference.angles_deg, record.angles_deg, atol=1e-6):
                raise ValueError(f"Angle grids differ in {key}: {reference.path} vs {record.path}")
    return groups
