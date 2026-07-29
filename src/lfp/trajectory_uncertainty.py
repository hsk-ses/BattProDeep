"""Particle-based uncertainty propagation for dynamic aging trajectories."""

from __future__ import annotations

import gzip
import gc
import json
import os
import pickle
import tempfile
from contextlib import contextmanager
from datetime import datetime
from functools import partial
from typing import Any, Dict, Iterable, Tuple

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import find_peaks
from tqdm import tqdm

if os.name == "nt" and not getattr(os.path.realpath, "_battprodeep_safe_realpath", False):
    _ORIGINAL_REALPATH = os.path.realpath

    def _safe_realpath(path, *args, **kwargs):
        try:
            return _ORIGINAL_REALPATH(path, *args, **kwargs)
        except OSError:
            return os.path.abspath(path)

    _safe_realpath._battprodeep_safe_realpath = True
    os.path.realpath = _safe_realpath

from src.common.cycle_analyze_new import augment_dataframe_with_half_cycles
from src.lfp.calendar_aging_current_modified_bootstrapped import CalendarAging
from src.lfp.cycle_aging_current_modified_bootstrapped import CycleAging


INDEPENDENT_CAL_CYC_UNCERTAINTY_MODES = {
    "full_centered_minchange_reference_refsigma_independent",
    "full_centered_minchange_reference_refsigma_split",
    "sigma_e_reference_refsigma_independent",
}

CONTINUOUS_ZETA_MODES = {
    "sigma_e",
    "sigma_total",
    "sigma_e_reference_refsigma",
}

DISK_BACKED_RESULT_FORMAT = "battprodeep.disk_backed_particle_results.v1"


def _repository_root() -> str:
    """Return the repository root without resolving the mapped drive."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _path_candidates(path: str | os.PathLike[str], repo_root: str) -> list[str]:
    """Return unique absolute interpretations of a user-supplied path."""
    expanded = os.path.expandvars(os.path.expanduser(os.fspath(path)))
    if os.path.isabs(expanded):
        return [os.path.abspath(expanded)]

    candidates = [
        os.path.abspath(expanded),
        os.path.abspath(os.path.join(repo_root, expanded)),
    ]
    unique_candidates: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = os.path.normcase(os.path.normpath(candidate))
        if normalized not in seen:
            unique_candidates.append(candidate)
            seen.add(normalized)
    return unique_candidates


def _require_existing_directory(
    path: str | os.PathLike[str] | None,
    label: str,
    repo_root: str,
) -> str | None:
    """Normalize an input directory and fail before model loading if absent."""
    if path is None:
        return None

    candidates = _path_candidates(path, repo_root)
    for candidate in candidates:
        try:
            if os.path.isdir(candidate):
                print(f"Path check | {label}: {candidate}", flush=True)
                return candidate
        except OSError:
            continue

    checked = "\n  - ".join(candidates)
    raise FileNotFoundError(
        f"{label} does not exist or is not accessible. Checked:\n  - {checked}"
    )


def _check_directory_writable(path: str, label: str) -> None:
    """Verify output access using a temporary file removed immediately."""
    probe_path = None
    try:
        descriptor, probe_path = tempfile.mkstemp(
            prefix=".battprodeep_path_check_",
            dir=path,
        )
        os.close(descriptor)
    except OSError as exc:
        raise OSError(f"{label} is not writable: {path}") from exc
    finally:
        if probe_path is not None:
            try:
                os.remove(probe_path)
            except FileNotFoundError:
                pass


def _prepare_output_directory(
    path: str | os.PathLike[str],
    label: str,
    repo_root: str,
    expected_directory: str | None = None,
) -> str:
    """Select, create, and verify an output directory before computation."""
    candidates = _path_candidates(path, repo_root)
    expected_normalized = (
        os.path.normcase(os.path.normpath(os.path.abspath(expected_directory)))
        if expected_directory is not None
        else None
    )

    selected = None
    if expected_normalized is not None:
        for candidate in candidates:
            if os.path.normcase(os.path.normpath(candidate)) == expected_normalized:
                selected = candidate
                break

    if selected is None:
        existing = []
        for candidate in candidates:
            try:
                if os.path.isdir(candidate):
                    existing.append(candidate)
            except OSError:
                continue
        selected = existing[0] if len(existing) == 1 else candidates[0]

    try:
        os.makedirs(selected, exist_ok=True)
    except OSError as exc:
        raise OSError(f"Could not create {label}: {selected}") from exc
    _check_directory_writable(selected, label)
    print(f"Path check | {label}: {selected}", flush=True)
    return selected


def _write_json_atomic(path: str, payload: Dict[str, Any]) -> None:
    """Write progress metadata without exposing a partially written JSON file."""
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary_path, path)


class _DiskBackedResultWriter:
    """Write completed trajectory blocks to disk without retaining prior blocks."""

    def __init__(
        self,
        output_dir: str,
        n_particles: int,
        allocated_steps: int,
        array_keys: Iterable[str],
    ) -> None:
        if allocated_steps <= 0:
            raise ValueError("Disk-backed result storage requires at least one timestep.")

        self.output_dir = os.path.abspath(output_dir)
        self.n_particles = int(n_particles)
        self.allocated_steps = int(allocated_steps)
        self.valid_steps = 0
        self.array_keys = tuple(array_keys)
        self.manifest_path = os.path.join(self.output_dir, "stream_manifest.json")
        os.makedirs(self.output_dir, exist_ok=False)

        self.array_paths: Dict[str, str] = {}
        self.arrays: Dict[str, np.memmap] = {}
        for key in self.array_keys:
            path = os.path.join(self.output_dir, f"{key}.npy")
            self.array_paths[key] = path
            self.arrays[key] = np.lib.format.open_memmap(
                path,
                mode="w+",
                dtype=np.float32,
                shape=(self.n_particles, self.allocated_steps),
            )

        self.timestamps_path = os.path.join(self.output_dir, "timestamps.npy")
        self.timestamps = np.lib.format.open_memmap(
            self.timestamps_path,
            mode="w+",
            dtype="datetime64[ns]",
            shape=(self.allocated_steps,),
        )
        self._write_manifest(complete=False)

    def _manifest(self, complete: bool) -> Dict[str, Any]:
        return {
            "format": DISK_BACKED_RESULT_FORMAT,
            "complete": bool(complete),
            "storage_dir": self.output_dir,
            "n_particles": self.n_particles,
            "allocated_steps": self.allocated_steps,
            "valid_steps": self.valid_steps,
            "array_keys": list(self.array_keys),
            "array_files": {
                key: os.path.basename(path)
                for key, path in self.array_paths.items()
            },
            "timestamps_file": os.path.basename(self.timestamps_path),
        }

    def _write_manifest(self, complete: bool) -> None:
        _write_json_atomic(self.manifest_path, self._manifest(complete=complete))

    def write_block(
        self,
        block_arrays: Dict[str, np.ndarray],
        timestamps: np.ndarray,
    ) -> None:
        timestamps_ns = np.asarray(timestamps).astype("datetime64[ns]", copy=False)
        block_steps = len(timestamps_ns)
        start = self.valid_steps
        end = start + block_steps
        if end > self.allocated_steps:
            raise ValueError(
                f"Disk-backed result capacity exceeded: end={end}, "
                f"allocated={self.allocated_steps}."
            )

        for key in self.array_keys:
            if key not in block_arrays:
                raise KeyError(f"Missing streamed result array: {key}")
            values = np.asarray(block_arrays[key], dtype=np.float32)
            expected_shape = (self.n_particles, block_steps)
            if values.shape != expected_shape:
                raise ValueError(
                    f"Unexpected shape for {key}: {values.shape}; "
                    f"expected {expected_shape}."
                )
            self.arrays[key][:, start:end] = values
            self.arrays[key].flush()

        self.timestamps[start:end] = timestamps_ns
        self.timestamps.flush()
        self.valid_steps = end
        self._write_manifest(complete=False)

    def finalize(self) -> Dict[str, Any]:
        for array in self.arrays.values():
            array.flush()
        self.timestamps.flush()
        self._write_manifest(complete=True)
        manifest = self._manifest(complete=True)

        for array in self.arrays.values():
            mmap_handle = getattr(array, "_mmap", None)
            if mmap_handle is not None:
                mmap_handle.close()
        timestamps_mmap = getattr(self.timestamps, "_mmap", None)
        if timestamps_mmap is not None:
            timestamps_mmap.close()
        self.arrays.clear()
        return manifest


def load_particle_results(
    source: str | os.PathLike[str] | Dict[str, Any],
    mmap_mode: str = "r",
) -> Dict[str, Any]:
    """Load either a legacy result pickle or a disk-backed result descriptor."""
    if isinstance(source, (str, os.PathLike)):
        source_path = os.fspath(source)
        opener = gzip.open if source_path.lower().endswith(".gz") else open
        with opener(source_path, "rb") as handle:
            payload = pickle.load(handle)
    else:
        payload = source

    if not isinstance(payload, dict) or payload.get("format") != DISK_BACKED_RESULT_FORMAT:
        return payload
    if not payload.get("complete", False):
        raise ValueError("Disk-backed particle result is incomplete.")

    storage_dir = payload["storage_dir"]
    valid_steps = int(payload["valid_steps"])
    results: Dict[str, Any] = {}
    for key, filename in payload["array_files"].items():
        array = np.load(os.path.join(storage_dir, filename), mmap_mode=mmap_mode)
        results[key] = array[:, :valid_steps]

    timestamps = np.load(
        os.path.join(storage_dir, payload["timestamps_file"]),
        mmap_mode=mmap_mode,
    )
    results["timestamps"] = timestamps[:valid_steps]
    return results


def close_particle_results(results: Dict[str, Any]) -> None:
    """Close memory-map handles returned by :func:`load_particle_results`."""
    closed_handles: set[int] = set()
    for value in results.values():
        mmap_handle = getattr(value, "_mmap", None)
        if mmap_handle is None or id(mmap_handle) in closed_handles:
            continue
        mmap_handle.close()
        closed_handles.add(id(mmap_handle))


def _model_epistemic_mode(epistemic_mode: str) -> str:
    if epistemic_mode in INDEPENDENT_CAL_CYC_UNCERTAINTY_MODES:
        return epistemic_mode.removesuffix("_independent").removesuffix("_split")
    return epistemic_mode


def _shuffle_with_low_tail_alignment(
    values: np.ndarray,
    eps_particles: np.ndarray,
    rng: np.random.Generator,
    max_tries: int = 256,
) -> np.ndarray:
    """Shuffle zeta values while avoiding strong eps/zeta tail alignment."""
    values = np.asarray(values).copy()
    eps_particles = np.asarray(eps_particles)
    if values.size < 8 or eps_particles.size != values.size:
        rng.shuffle(values)
        return values

    eps_hi = eps_particles >= np.quantile(eps_particles, 0.9)
    eps_lo = eps_particles <= np.quantile(eps_particles, 0.1)
    best_values = None
    best_score = np.inf

    for _ in range(max_tries):
        candidate = values.copy()
        rng.shuffle(candidate)
        zeta_hi = candidate >= np.quantile(candidate, 0.9)
        zeta_lo = candidate <= np.quantile(candidate, 0.1)
        same_tail = np.mean((eps_hi & zeta_hi) | (eps_lo & zeta_lo))
        corr = np.corrcoef(eps_particles, candidate)[0, 1]
        corr = 0.0 if not np.isfinite(corr) else abs(float(corr))
        score = corr + same_tail
        if score < best_score:
            best_score = score
            best_values = candidate

    return best_values.astype(values.dtype, copy=False)


@contextmanager
def tqdm_joblib(tqdm_object):
    """Patch joblib so Parallel tasks update a tqdm progress bar."""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()


def calculate_soc_with_current_losses_vectorized_crate_limit(
    M: int,
    current_series: pd.Series,
    initial_soc: np.ndarray,
    C_nominal: float,
    soh: np.ndarray,
    eta_charge: float = 1.0,
    eta_discharge: float = 1.0,
    soc_min: float = 0.0,
    soc_max: float = 1.0,
    C_rate_max: float = 1.0,
    scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate particle SOC trajectories while respecting SoH-dependent C-rate limits."""
    if not isinstance(current_series.index, pd.DatetimeIndex):
        raise ValueError("Index must be a DatetimeIndex")

    current = current_series.to_numpy(dtype=np.float32)
    n_steps = len(current)
    dt_h = current_series.index.to_series().diff().dropna().mode()[0].total_seconds() / 3600.0

    eff_cap = np.maximum(C_nominal * np.asarray(soh, dtype=np.float32), 1e-9)
    i_max = eff_cap * C_rate_max

    current_scaled = current[None, :] * scale
    current_rated = np.clip(current_scaled, -i_max[:, None], i_max[:, None]).astype(np.float32)

    delta_ah = np.where(
        current_rated > 0,
        current_rated * eta_charge,
        current_rated / eta_discharge,
    ) * dt_h
    delta_soc = delta_ah / eff_cap[:, None]

    soc = np.empty((M, n_steps), dtype=np.float32)
    initial_soc = np.asarray(initial_soc, dtype=np.float32).reshape(M)
    soc[:, 0] = np.clip(initial_soc, soc_min, soc_max)

    for t in range(1, n_steps):
        soc[:, t] = np.clip(soc[:, t - 1] + delta_soc[:, t], soc_min, soc_max)

    return soc, current_rated


def sample_uncertainty_particles(
    M: int,
    n_boot: int = 32,
    seed: int = 42,
    stratified_eps: bool = True,
    epistemic_mode: str = "anchored",
    avoid_eps_zeta_tail_alignment: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample aleatoric particles and epistemic particles.

    ``zeta_particles`` is continuous for sigma-style modes. Bootstrap-member
    modes coerce it to integer member ids at the call site. Sigma-style modes
    pin particle 0 to ``eps=zeta=0`` so it is the deterministic reference
    trajectory. The seed fixes all samples.
    """
    rng = np.random.default_rng(seed)
    if stratified_eps:
        # Acklam-like inverse-normal via scipy is avoided here to keep imports light.
        from scipy import stats

        p = np.linspace(0.025, 0.975, M)
        eps_particles = stats.norm.ppf(p).astype(np.float32)
        rng.shuffle(eps_particles)
    else:
        eps_particles = rng.standard_normal(M).astype(np.float32)

    if epistemic_mode in CONTINUOUS_ZETA_MODES:
        if stratified_eps:
            zeta_particles = stats.norm.ppf(p).astype(np.float32)
            if avoid_eps_zeta_tail_alignment and epistemic_mode == "sigma_e_reference_refsigma":
                zeta_particles = _shuffle_with_low_tail_alignment(
                    zeta_particles,
                    eps_particles,
                    rng,
                )
            else:
                rng.shuffle(zeta_particles)
        else:
            zeta_particles = rng.standard_normal(M).astype(np.float32)

    else:
        zeta_particles = np.tile(np.arange(n_boot, dtype=np.int32), M // n_boot + 1)[:M]
        rng.shuffle(zeta_particles)

    if epistemic_mode in CONTINUOUS_ZETA_MODES and M > 0:
        eps_particles[0] = 0.0
        zeta_particles[0] = 0.0
    return eps_particles, zeta_particles


def _half_cycle_arrays_one_particle(
    soc: np.ndarray,
    current: np.ndarray,
    battery_capacity: float,
    fec_offset: float,
    depth_threshold: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fast array equivalent of the notebook's half-cycle augmentation for one particle."""
    n_steps = soc.shape[0]
    delta_fec = np.zeros(n_steps, dtype=np.float32)
    half_cycle_depth = np.zeros(n_steps, dtype=np.float32)

    peaks, _ = find_peaks(soc, height=0)
    valleys, _ = find_peaks(-soc, height=-1)
    extreme_points = np.sort(np.concatenate((peaks, valleys))).astype(np.int64)

    idx_start = 0
    for j in range(len(extreme_points) - 1):
        i_start = int(extreme_points[j])
        i_end = int(extreme_points[j + 1])
        depth = float((soc[i_end] - soc[i_start]) * battery_capacity)

        if abs(depth) <= depth_threshold and idx_start == 0:
            idx_start = i_start
            continue
        if abs(depth) <= depth_threshold:
            continue

        start = idx_start if idx_start != 0 else i_start
        cycle_depth_frac = float(soc[i_end] - soc[start])
        half_cycle_depth[start : i_end + 1] = cycle_depth_frac
        delta_fec[i_end] = abs(cycle_depth_frac) / 2.0
        idx_start = 0

    c_rate = (current / battery_capacity).astype(np.float32)
    charge_rate = np.where(c_rate > 0, c_rate, 0.0).astype(np.float32)
    discharge_rate = np.where(c_rate < 0, c_rate, 0.0).astype(np.float32)
    fec = fec_offset + np.cumsum(delta_fec, dtype=np.float32)

    return delta_fec, half_cycle_depth, charge_rate, discharge_rate, fec


def _legacy_half_cycle_arrays_one_particle(
    block: pd.DataFrame,
    soc: np.ndarray,
    current: np.ndarray,
    battery_capacity: float,
    fec_offset: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact notebook half-cycle construction for one particle."""
    block_particle = block.assign(soc=soc)
    block_particle["Current"] = current
    df_aug, _ = augment_dataframe_with_half_cycles(
        block_particle,
        profile_type="soc_profile",
        battery_capacity=battery_capacity,
        method="half_cycle",
        consider_original_current="Current",
    )
    delta_fec = df_aug["Delta_FEC"].to_numpy(dtype=np.float32)
    fec = np.float32(fec_offset) + np.cumsum(delta_fec, dtype=np.float32)
    return (
        delta_fec,
        df_aug["Half_Cycle_Depth"].to_numpy(dtype=np.float32),
        df_aug["Charge_Rate"].to_numpy(dtype=np.float32),
        df_aug["Discharge_Rate"].to_numpy(dtype=np.float32),
        fec,
    )


def prepare_half_cycle_particle_inputs(
    block: pd.DataFrame,
    soc_particles: np.ndarray,
    current_particles: np.ndarray,
    fec_t0_particles: np.ndarray,
    battery_capacity: float,
    n_jobs: int = 1,
    depth_threshold: float = 0.0,
    mode: str = "legacy",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare cycle-model particle inputs.

    The default ``legacy`` mode intentionally mirrors the original notebook
    wrapper, including ``fec_offset + Delta_FEC.cumsum()``.
    """
    M = soc_particles.shape[0]
    timestamps = block.index.to_numpy()

    if mode == "legacy":
        if n_jobs and n_jobs != 1:
            tasks = (
                delayed(_legacy_half_cycle_arrays_one_particle)(
                    block,
                    soc_particles[m],
                    current_particles[m],
                    battery_capacity,
                    float(fec_t0_particles[m]),
                )
                for m in range(M)
            )
            outputs = Parallel(n_jobs=n_jobs, prefer="threads")(tasks)
        else:
            outputs = [
                _legacy_half_cycle_arrays_one_particle(
                    block,
                    soc_particles[m],
                    current_particles[m],
                    battery_capacity,
                    float(fec_t0_particles[m]),
                )
                for m in range(M)
            ]
    elif mode == "fast":
        if n_jobs and n_jobs != 1:
            tasks = (
                delayed(_half_cycle_arrays_one_particle)(
                    soc_particles[m],
                    current_particles[m],
                    battery_capacity,
                    float(fec_t0_particles[m]),
                    depth_threshold,
                )
                for m in range(M)
            )
            outputs = Parallel(n_jobs=n_jobs, prefer="threads")(tasks)
        else:
            outputs = [
                _half_cycle_arrays_one_particle(
                    soc_particles[m],
                    current_particles[m],
                    battery_capacity,
                    float(fec_t0_particles[m]),
                    depth_threshold,
                )
                for m in range(M)
            ]
    else:
        raise ValueError(f"Unknown half_cycle_mode: {mode}")

    delta_fec, dod, charge, discharge, fec = zip(*outputs)
    return (
        np.asarray(delta_fec, dtype=np.float32),
        np.asarray(fec, dtype=np.float32),
        np.asarray(dod, dtype=np.float32),
        np.asarray(timestamps, dtype="datetime64[ns]"),
        np.asarray(charge, dtype=np.float32),
        np.asarray(discharge, dtype=np.float32),
    )

# def _time_blocks(profile: pd.DataFrame, freq: str = "3MS") -> Iterable[pd.DataFrame]:
#     start = profile.index.min().ceil("H")
#     end = profile.index.max().floor("H")
#     edges = pd.date_range(start=start, end=end, freq=freq)
#     if len(edges) == 0 or edges[-1] < end:
#         edges = edges.append(pd.DatetimeIndex([end]))

#     for t0, t1 in zip(edges[:-1], edges[1:]):
#         block = profile.loc[t0:t1].iloc[:-1]
#         if not block.empty:
#             yield block
def _time_blocks(profile: pd.DataFrame, freq: str = "3MS") -> Iterable[pd.DataFrame]:
    start = profile.index.min().ceil("H")
    end = profile.index.max().floor("H")
    if start >= end:
        return

    if freq == "3MS":
        offset = pd.DateOffset(months=3)
    else:
        offset = pd.tseries.frequencies.to_offset(freq)

    edges = [start]
    while edges[-1] < end:
        next_edge = edges[-1] + offset
        edges.append(min(next_edge, end))
    edges = pd.DatetimeIndex(edges)

    for t0, t1 in zip(edges[:-1], edges[1:]):
        block = profile.loc[t0:t1].iloc[:-1]
        if not block.empty:
            yield block

def run_variant_timebased_vectorized(
    eps_particles: np.ndarray,
    zeta_particles: np.ndarray,
    name: str,
    profile: pd.DataFrame,
    cycle_model: CycleAging,
    calendar_model: CalendarAging,
    C_nominal: float = 3.0,
    fec_window: float = 3.0,
    years: int = 1,
    max_blocks: int | None = None,
    paths: Dict[str, str] | None = None,
    soc_max: float = 1.0,
    soc_min: float = 0.0,
    C_rate_max: float = 1.0,
    scale: float = 1.0,
    half_cycle_n_jobs: int = 1,
    n_boot: int = 32,
    bootstrap_seed: int | None = 42,
    epistemic_mode: str = "anchored", #or "full"
    epistemic_support_scale: float = 1.0,
    stop_soh: float = 0.7,
    store_soc: bool = True,
    store_fec: bool = True,
    store_det: bool = False,
    eps_calendar_particles: np.ndarray | None = None,
    zeta_calendar_particles: np.ndarray | None = None,
    eps_cycle_particles: np.ndarray | None = None,
    zeta_cycle_particles: np.ndarray | None = None,
    disk_backed_output_dir: str | None = None,
) -> Dict[str, Any]:
    """Run combined cycle and calendar aging for one profile variant."""
    profile = profile.copy(deep=False)
    if "timestamp" not in profile.columns:
        profile["timestamp"] = profile.index
    if not isinstance(profile.index, pd.DatetimeIndex):
        raise ValueError("Profile index must be DatetimeIndex")

    M = len(eps_particles)
    eps_particles = np.asarray(eps_particles, dtype=np.float32).reshape(M)
    zeta_particles = np.asarray(zeta_particles, dtype=np.float32).reshape(M)
    eps_calendar_particles = (
        eps_particles
        if eps_calendar_particles is None
        else np.asarray(eps_calendar_particles, dtype=np.float32).reshape(M)
    )
    zeta_calendar_particles = (
        zeta_particles
        if zeta_calendar_particles is None
        else np.asarray(zeta_calendar_particles, dtype=np.float32).reshape(M)
    )
    eps_cycle_particles = (
        eps_particles
        if eps_cycle_particles is None
        else np.asarray(eps_cycle_particles, dtype=np.float32).reshape(M)
    )
    zeta_cycle_particles = (
        zeta_particles
        if zeta_cycle_particles is None
        else np.asarray(zeta_cycle_particles, dtype=np.float32).reshape(M)
    )
    model_epistemic_mode = _model_epistemic_mode(epistemic_mode)

    soc_init = np.full(M, profile["soc"].iloc[0], dtype=np.float32)
    vt = np.zeros(M, dtype=np.float32)
    fec_t0_particles = np.zeros(M, dtype=np.float32)
    cumulative_cycle_loss_particles = np.zeros(M, dtype=np.float32)
    cumulative_calendar_loss_particles = np.zeros(M, dtype=np.float32)
    last_std_particles = np.zeros(M, dtype=np.float32)
    max_spread_cal = 0.0
    max_spread_cyc = 0.0
    soh_particles = np.ones(M, dtype=np.float32)

    log_dir = (
        os.path.join(paths["run_base"], "logs", "Looped_run")
        if paths and "run_base" in paths
        else os.path.join("logs", "Looped_run")
    )
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cycle_log = os.path.join(log_dir, f"cycle_loss_input_feature_{name}_{timestamp}.pkl")
    calendar_log = os.path.join(log_dir, f"calendar_loss_input_feature_{name}_{timestamp}.pkl")

    result_keys = ["cycle_loss", "calendar_loss"]
    if store_soc:
        result_keys.append("soc")
    if store_fec:
        result_keys.append("fec")
    if store_det:
        result_keys.extend(["det_cycle_loss", "det_calendar_loss"])

    blocks = list(_time_blocks(profile))
    max_storage_blocks = (
        len(blocks)
        if max_blocks is None
        else min(int(max_blocks), len(blocks))
    )
    result_writer = None
    particle_results: Dict[str, list[Any]] = {}
    timestamp_chunks: list[np.ndarray] = []
    if disk_backed_output_dir is not None:
        allocated_steps = sum(len(block) for block in blocks[:max_storage_blocks])
        result_writer = _DiskBackedResultWriter(
            output_dir=disk_backed_output_dir,
            n_particles=M,
            allocated_steps=allocated_steps,
            array_keys=result_keys,
        )
    else:
        particle_results = {key: [] for key in result_keys}

    with tqdm(total=len(blocks), desc=f"{name} | FEC blocks", leave=False) as pbar_blocks:
        for block_idx, block in enumerate(blocks, start=1):
            soc_particles, i_actual = calculate_soc_with_current_losses_vectorized_crate_limit(
                M=M,
                current_series=block["Current"],
                initial_soc=soc_init,
                C_nominal=C_nominal,
                soh=soh_particles,
                soc_max=soc_max,
                soc_min=soc_min,
                C_rate_max=C_rate_max,
                scale=scale,
            )

            (
                delta_fec_particles,
                fec_particles,
                dod_particles,
                timestamps_shared,
                charge_rate_particles,
                discharge_rate_particles,
            ) = prepare_half_cycle_particle_inputs(
                block=block,
                soc_particles=soc_particles,
                current_particles=i_actual,
                fec_t0_particles=fec_t0_particles,
                battery_capacity=C_nominal,
                n_jobs=half_cycle_n_jobs,
            )

            (
                cycle_loss,
                cumulative_cycle_loss_particles,
                fec_array_particles,
                max_spread_cyc,
                det_cycle_loss,
            ) = cycle_model.lfp_cycle_BattProDeep_vectorized(
                M=M,
                profile=block,
                soc_particles=soc_particles,
                delta_FEC_particles=delta_fec_particles,
                FEC_particles=fec_particles,
                DoD_particles=dod_particles,
                timestamps_particles=timestamps_shared,
                charge_rate_particles=charge_rate_particles,
                discharge_rate_particles=discharge_rate_particles,
                years=1,
                fec_window=fec_window,
                cumulative_loss0=cumulative_cycle_loss_particles,
                temperature=37,
                temperature_type="profile",
                temperature_column="Temperature",
                max_spread0=max_spread_cyc,
                fec_scale=years,
                log_file=cycle_log,
                eps=eps_cycle_particles,
                zeta=zeta_cycle_particles,
                n_boot=n_boot,
                bootstrap_seed=bootstrap_seed,
                epistemic_mode=model_epistemic_mode,
                epistemic_support_scale=epistemic_support_scale,
                return_det=store_det,
            )

            (
                calendar_loss,
                vt,
                calendar_cum_pct,
                last_std_cal,
                _vt_trace,
                max_spread_cal,
                det_calendar_loss,
            ) = calendar_model.lfp_calendar_BattProDeep_vectorized(
                M=M,
                soc_particles=soc_particles,
                profile=block,
                variant_name=name,
                years=1,
                vt_scale=years,
                window_size=60,
                temperature=37,
                temperature_type="profile",
                temperature_column="Temperature",
                virtual_time0=vt,
                cumulative_loss0=cumulative_calendar_loss_particles * 100.0,
                last_std0=last_std_particles,
                log_file=calendar_log,
                eps=eps_calendar_particles,
                zeta=zeta_calendar_particles,
                max_spread0=max_spread_cal,
                n_boot=n_boot,
                bootstrap_seed=bootstrap_seed,
                epistemic_mode=model_epistemic_mode,
                epistemic_support_scale=epistemic_support_scale,
                return_trace=False,
                return_det=store_det,
            )

            last_std_particles = np.nan_to_num(last_std_cal, nan=0.0, posinf=0.0, neginf=0.0)
            cumulative_calendar_loss_particles = calendar_cum_pct / 100.0
            soh_particles = 1.0 - (cumulative_cycle_loss_particles + cumulative_calendar_loss_particles)
            soh_particles = np.clip(soh_particles, 1e-3, 1.0)

            block_results: Dict[str, np.ndarray] = {
                "cycle_loss": cycle_loss.astype(np.float32, copy=False),
                "calendar_loss": calendar_loss.astype(np.float32, copy=False),
            }
            if store_det and det_cycle_loss is not None and det_calendar_loss is not None:
                block_results["det_cycle_loss"] = det_cycle_loss.astype(np.float32, copy=False)
                block_results["det_calendar_loss"] = det_calendar_loss.astype(np.float32, copy=False)
            if store_soc:
                block_results["soc"] = soc_particles.astype(np.float32, copy=False)
            if store_fec:
                block_results["fec"] = fec_particles.astype(np.float32, copy=False)

            if result_writer is not None:
                result_writer.write_block(block_results, timestamps_shared)
                # Copies prevent the final columns from retaining the complete
                # block arrays after those arrays have been written to disk.
                soc_init = soc_particles[:, -1].copy()
                fec_t0_particles = fec_particles[:, -1].copy()
            else:
                for key, values in block_results.items():
                    particle_results[key].append(values)
                timestamp_chunks.append(timestamps_shared)
                soc_init = soc_particles[:, -1]
                fec_t0_particles = fec_particles[:, -1]

            reached_max_blocks = max_blocks is not None and block_idx >= max_blocks
            reached_stop_soh = soh_particles.mean(axis=0) <= stop_soh
            if reached_stop_soh:
                print(f"{name}: SoH {soh_particles.mean(axis=0):.3f} reached stop threshold at block {block_idx}", flush=True)

            if result_writer is not None:
                del block_results
                del soc_particles, i_actual
                del delta_fec_particles, fec_particles, dod_particles
                del charge_rate_particles, discharge_rate_particles
                del cycle_loss, calendar_loss, fec_array_particles
                del det_cycle_loss, det_calendar_loss, last_std_cal, calendar_cum_pct
                gc.collect()

            if reached_max_blocks or reached_stop_soh:
                break
            pbar_blocks.update(1)

    if result_writer is not None:
        return result_writer.finalize()

    stacked: Dict[str, np.ndarray] = {}
    for key, chunks in particle_results.items():
        stacked[key] = np.concatenate(chunks, axis=-1)
    stacked["timestamps"] = np.concatenate(timestamp_chunks).astype("datetime64[ns]")
    return stacked


def process_variant_vectorized(
    name: str,
    profile: pd.DataFrame,
    fec_window: float | None = None,
    paths: Dict[str, str] | None = None,
    C_nominal: float = 3.0,
    years: int = 1,
    max_blocks: int | None = None,
    soc_max: float = 1.0,
    soc_min: float = 0.0,
    eps: np.ndarray | None = None,
    zeta: np.ndarray | None = None,
    eps_calendar: np.ndarray | None = None,
    zeta_calendar: np.ndarray | None = None,
    eps_cycle: np.ndarray | None = None,
    zeta_cycle: np.ndarray | None = None,
    C_rate_max: float = 1.0,
    scale: float = 1.0,
    outdir: str | None = None,
    half_cycle_n_jobs: int = 1,
    n_boot: int = 32,
    bootstrap_seed: int | None = 42,
    epistemic_mode: str = "anchored",
    epistemic_support_scale: float = 1.0,
    avoid_eps_zeta_tail_alignment: bool = False,
    calendar_aleatoric_model_folder: str | None = None,
    cycle_aleatoric_model_folder: str | None = None,
    calendar_epistemic_model_folder: str | None = None,
    cycle_epistemic_model_folder: str | None = None,
    store_soc: bool = True,
    store_fec: bool = True,
    store_det: bool = False,
    disk_backed_streaming: bool = False,
    disk_backed_root: str | None = None,
) -> Tuple[str, str]:
    """Run and save one profile variant. Returns the lightweight result path."""
    print(f"Starting {name}", flush=True)
    if paths is not None:
        os.makedirs(paths["pickles_dir"], exist_ok=True)
    if outdir is None:
        outdir = paths["pickles_dir"] if paths else "."
    os.makedirs(outdir, exist_ok=True)

    stream_timestamp = None
    disk_backed_output_dir = None
    if disk_backed_streaming:
        stream_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        stream_root = (
            os.path.abspath(disk_backed_root)
            if disk_backed_root is not None
            else os.path.join(os.path.abspath(outdir), "disk_backed")
        )
        os.makedirs(stream_root, exist_ok=True)
        disk_backed_output_dir = os.path.join(
            stream_root,
            f"{name}_particle_results_{stream_timestamp}",
        )

    calendar_model_kwargs = {}
    cycle_model_kwargs = {}
    if calendar_aleatoric_model_folder is not None:
        calendar_model_kwargs["aleatoric_model_folder"] = calendar_aleatoric_model_folder
    if cycle_aleatoric_model_folder is not None:
        cycle_model_kwargs["aleatoric_model_folder"] = cycle_aleatoric_model_folder
    if calendar_epistemic_model_folder is not None:
        calendar_model_kwargs["epistemic_model_folder"] = calendar_epistemic_model_folder
    if cycle_epistemic_model_folder is not None:
        cycle_model_kwargs["epistemic_model_folder"] = cycle_epistemic_model_folder

    particle_results = run_variant_timebased_vectorized(
        eps_particles=eps,
        zeta_particles=zeta,
        name=name,
        profile=profile,
        calendar_model=CalendarAging(load_trained_models=True, **calendar_model_kwargs),
        cycle_model=CycleAging(load_trained_models=True, **cycle_model_kwargs),
        C_nominal=C_nominal,
        fec_window=fec_window,
        years=years,
        max_blocks=max_blocks,
        paths=paths,
        soc_max=soc_max,
        soc_min=soc_min,
        C_rate_max=C_rate_max,
        scale=scale,
        half_cycle_n_jobs=half_cycle_n_jobs,
        n_boot=n_boot,
        bootstrap_seed=bootstrap_seed,
        epistemic_mode=epistemic_mode,
        epistemic_support_scale=epistemic_support_scale,
        store_soc=store_soc,
        store_fec=store_fec,
        store_det=store_det,
        eps_calendar_particles=eps_calendar,
        zeta_calendar_particles=zeta_calendar,
        eps_cycle_particles=eps_cycle,
        zeta_cycle_particles=zeta_cycle,
        disk_backed_output_dir=disk_backed_output_dir,
    )

    timestamp = stream_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(outdir, f"{name}_particle_results_{timestamp}.pkl.gz")
    with gzip.open(out_file, "wb") as f:
        pickle.dump(particle_results, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved {name} -> {out_file}", flush=True)
    return name, out_file


def run_outer_loop_variants_timebased_vectorized(
    dfCPD_refilled: pd.DataFrame | None = None,
    variants: Dict[str, pd.DataFrame] | None = None,
    paths: Dict[str, str] | None = None,
    C_nominal: float = 3.0,
    fec_window: float = 3.0,
    years: int = 1,
    n_jobs: int = 1,
    max_blocks: int | None = None,
    soc_max: float = 1.0,
    soc_min: float = 0.0,
    seed: int = 42,
    M: int = 100,
    C_rate_max: float = 1.0,
    scale: float = 1.0,
    n_boot: int = 32,
    bootstrap_seed: int | None = None,
    epistemic_mode: str = "anchored",
    epistemic_support_scale: float = 1.0,
    avoid_eps_zeta_tail_alignment: bool = False,
    calendar_aleatoric_model_folder: str | None = None,
    cycle_aleatoric_model_folder: str | None = None,
    calendar_epistemic_model_folder: str | None = None,
    cycle_epistemic_model_folder: str | None = None,
    half_cycle_n_jobs: int = 1,
    parallel_backend: str = "threading",
    store_soc: bool = True,
    store_fec: bool = True,
    store_det: bool = False,
    disk_backed_streaming: bool = False,
    disk_backed_root: str | None = None,
    save_results_index: bool = True,
    results_index_name: str | None = None,
) -> Dict[str, str]:
    """Run all stress variants and save each result as gzip pickle."""
    repo_root = _repository_root()
    if paths is not None and "pickles_dir" not in paths:
        raise KeyError("'paths' must contain a 'pickles_dir' entry.")
    if paths is not None:
        paths = dict(paths)
        paths["pickles_dir"] = _prepare_output_directory(
            paths["pickles_dir"],
            label="pickle root directory",
            repo_root=repo_root,
        )
        if "run_base" in paths:
            paths["run_base"] = _prepare_output_directory(
                paths["run_base"],
                label="run/log root directory",
                repo_root=repo_root,
            )

    outdir_input = (
        os.path.join(paths["pickles_dir"], "9_fec_with_input_feature")
        if paths
        else "9_fec_with_input_feature"
    )
    outdir = _prepare_output_directory(
        outdir_input,
        label="result descriptor directory",
        repo_root=repo_root,
    )

    calendar_aleatoric_model_folder = _require_existing_directory(
        calendar_aleatoric_model_folder,
        label="calendar aleatoric model directory",
        repo_root=repo_root,
    )
    cycle_aleatoric_model_folder = _require_existing_directory(
        cycle_aleatoric_model_folder,
        label="cycle aleatoric model directory",
        repo_root=repo_root,
    )
    calendar_epistemic_model_folder = _require_existing_directory(
        calendar_epistemic_model_folder,
        label="calendar epistemic model directory",
        repo_root=repo_root,
    )
    cycle_epistemic_model_folder = _require_existing_directory(
        cycle_epistemic_model_folder,
        label="cycle epistemic model directory",
        repo_root=repo_root,
    )

    if disk_backed_streaming:
        disk_backed_root_input = (
            disk_backed_root
            if disk_backed_root is not None
            else os.path.join(outdir, "disk_backed")
        )
        disk_backed_root = _prepare_output_directory(
            disk_backed_root_input,
            label="disk-backed result directory",
            repo_root=repo_root,
            expected_directory=outdir if disk_backed_root is not None else None,
        )

    if bootstrap_seed is None:
        bootstrap_seed = seed

    model_epistemic_mode = _model_epistemic_mode(epistemic_mode)
    eps_particles, zeta_particles = sample_uncertainty_particles(
        M=M,
        n_boot=n_boot,
        seed=seed,
        epistemic_mode=model_epistemic_mode,
        avoid_eps_zeta_tail_alignment=avoid_eps_zeta_tail_alignment,
    )
    eps_calendar_particles = None
    zeta_calendar_particles = None
    eps_cycle_particles = None
    zeta_cycle_particles = None
    if epistemic_mode in INDEPENDENT_CAL_CYC_UNCERTAINTY_MODES:
        eps_calendar_particles, zeta_calendar_particles = sample_uncertainty_particles(
            M=M,
            n_boot=n_boot,
            seed=seed,
            epistemic_mode=model_epistemic_mode,
            avoid_eps_zeta_tail_alignment=avoid_eps_zeta_tail_alignment,
        )
        eps_cycle_particles, zeta_cycle_particles = sample_uncertainty_particles(
            M=M,
            n_boot=n_boot,
            seed=seed + 1,
            epistemic_mode=model_epistemic_mode,
            avoid_eps_zeta_tail_alignment=avoid_eps_zeta_tail_alignment,
        )
        eps_particles = eps_calendar_particles
        zeta_particles = zeta_calendar_particles
        print(
            "Using independent calendar/cycle eps and zeta assignments "
            f"for mode {epistemic_mode!r}.",
            flush=True,
        )

    if variants is None:
        if dfCPD_refilled is None:
            raise ValueError("Provide either 'variants' or 'dfCPD_refilled'.")
        variants = {
            "max_max_temp": dfCPD_refilled.assign(
                Current=dfCPD_refilled["I_in_A_cell_max"],
                Temperature=dfCPD_refilled["T_cell_max"],
                soc=dfCPD_refilled["SOC mit I-Verlust"],
            ),
            "min_min_temp": dfCPD_refilled.assign(
                Current=dfCPD_refilled["I_in_A_cell_min"],
                Temperature=dfCPD_refilled["T_cell_min"],
                soc=dfCPD_refilled["SOC mit I-Verlust"],
            ),
        }
    elif len(variants) == 0:
        raise ValueError("'variants' must contain at least one profile.")

    process_func = partial(
        process_variant_vectorized,
        paths=paths,
        fec_window=fec_window,
        C_nominal=C_nominal,
        years=years,
        max_blocks=max_blocks,
        soc_max=soc_max,
        soc_min=soc_min,
        eps=eps_particles,
        zeta=zeta_particles,
        eps_calendar=eps_calendar_particles,
        zeta_calendar=zeta_calendar_particles,
        eps_cycle=eps_cycle_particles,
        zeta_cycle=zeta_cycle_particles,
        C_rate_max=C_rate_max,
        scale=scale,
        outdir=outdir,
        half_cycle_n_jobs=half_cycle_n_jobs,
        n_boot=n_boot,
        bootstrap_seed=bootstrap_seed,
        epistemic_mode=epistemic_mode,
        epistemic_support_scale=epistemic_support_scale,
        calendar_aleatoric_model_folder=calendar_aleatoric_model_folder,
        cycle_aleatoric_model_folder=cycle_aleatoric_model_folder,
        calendar_epistemic_model_folder=calendar_epistemic_model_folder,
        cycle_epistemic_model_folder=cycle_epistemic_model_folder,
        store_soc=store_soc,
        store_fec=store_fec,
        store_det=store_det,
        disk_backed_streaming=disk_backed_streaming,
        disk_backed_root=disk_backed_root,
    )

    if n_jobs and n_jobs != 1:
        print(
            f"Running {len(variants)} variants with n_jobs={n_jobs}, backend={parallel_backend}",
            flush=True,
        )
        prefer = "threads" if parallel_backend == "threading" else None
        with tqdm_joblib(tqdm(total=len(variants), desc="Variants completed")):
            results_list = Parallel(n_jobs=n_jobs, prefer=prefer)(
                delayed(process_func)(name, profile)
                for name, profile in variants.items()
            )
    else:
        results_list = [process_func(name, profile) for name, profile in variants.items()]

    results = {name: path for name, path in results_list}
    if save_results_index:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if results_index_name is None:
            results_index_name = f"looped_particle_results_timebased_particles{timestamp}_bootstrapped_reducedmodel_short.pkl"
        elif not results_index_name.lower().endswith(".pkl"):
            results_index_name = f"{results_index_name}_{timestamp}.pkl"

        results_file = os.path.join(outdir, results_index_name)
        with open(results_file, "wb") as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

        print("\nSaved combined cycle + calendar result paths as PKL:")
        print(f"   {results_file}")
    return results


