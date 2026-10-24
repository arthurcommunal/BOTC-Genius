"""Settings of the application."""
from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

CORRESPONDANCE = [("BOTC_MAX_BATCH_SIZE", "max_batch_size", int),
                  ("BOTC_MAX_WAIT_MS", "max_wait_ms", float),
                  ("BOTC_MAX_QUEUE_SIZE", "max_queue_size", int),
                  ("BOTC_CACHE_CAPACITY", "cache_capacity", int),
                  ("BOTC_TOP_K", "top_k", int),
                  ("BOTC_EMBEDDING_DIM", "embedding_dim", int),
                  ("BOTC_N_SHARDS", "n_shards", int)]

@dataclass(frozen=True)
class Settings:
    """Immutable settings read by `create_app` then by the lifespan."""

    tenants_path: Path
    files_path: Path
    max_batch_size: int = 16
    max_wait_ms: float = 10.0
    max_queue_size: int = 1024
    cache_capacity: int = 1024
    top_k: int = 5
    embedding_dim: int = 384
    n_shards: int = 2
    clock: Callable[[], float] | None = None  # None = horloge monotone par défaut

def load_settings_from_env() -> Settings:
    """Makes the production settings from environment variables."""
    env = os.environ
    params = {}
    botc_tenants_raw = env.get("BOTC_TENANTS_PATH")

    if botc_tenants_raw is None:
        raise RuntimeError("BOTC_TENANTS_PATH is not defined in the environment")
    botc_tenants_path = Path(botc_tenants_raw)
    if not botc_tenants_path.is_file():
        raise FileNotFoundError(f"{botc_tenants_path} not found")
    params["tenants_path"] = botc_tenants_path

    botc_files_raw = env.get("BOTC_FILES_PATH")
    if botc_files_raw is None:
        raise RuntimeError("BOTC_FILES_PATH is not defined in the environment")
    botc_files_path = Path(botc_files_raw)
    if not botc_files_path.is_dir():
        raise FileNotFoundError(f"{botc_files_path} not found")
    params["files_path"] = botc_files_path
    
    for env_name, field, convert in CORRESPONDANCE:
        value = env.get(env_name)
        if value is not None:
            try:
                params[field] = convert(value)
            except ValueError as e:
                raise ValueError(f"{env_name} = {value!r} - should be of the type {convert.__name__}") from e

    settings = Settings(**params)
    return settings


