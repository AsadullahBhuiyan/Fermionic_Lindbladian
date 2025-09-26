# generate_CI_Lindblad_histories.py
import os
import time
import importlib
import numpy as np

# Adjust this import to match your file name if needed:
import CI_Lindblad_DW
importlib.reload(CI_Lindblad_DW)
from CI_Lindblad_DW import CI_Lindblad_DW


def format_interval(seconds: float) -> str:
    seconds = int(round(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{days}d {hours:02}:{minutes:02}:{seconds:02}" if days else f"{hours:02}:{minutes:02}:{seconds:02}"


# ----------------------------
# Config (edit these)
# ----------------------------
Nx, Ny    = 30, 30
dt        = 5e-2
T_cycle   = 20  
nshell    = None           # None for untruncated
DW        = True           # keep the domain-wall slab geometry

# Which caches to build:
GENERATE_DECOH_OFF = True   # decoherence disabled
GENERATE_DECOH_ON  = True   # decoherence enabled

# Keep full history in RAM after load? (animation needs this True)
keep_history = True

# Overwrite existing caches instead of loading?
overwrite = True


# ----------------------------
# Run
# ----------------------------
t0 = time.time()
artifacts = []

def run_one(decoh_flag: bool):
    tag = "decoh_on" if decoh_flag else "decoh_off"
    print(f"\n[run] Generating temporal G history for {tag} ...")
    t1 = time.time()

    # Instantiate; __init__ tries to load matching cache,
    # else evolves once, samples at T_cycle, and saves.
    m = CI_Lindblad_DW(
        Nx, Ny,
        init=True,
        decoh=decoh_flag,
        nshell=nshell,
        DW=DW,
        dt=dt,
        T_cycle=T_cycle,
        G_init=None,             # start from zeros by default
        keep_history=keep_history,
        overwrite=overwrite,
    )

    # Compute/cache path used for this parameter set
    cache_path = m._cache_path(dt, max_steps, T_cycle)
    exists = os.path.isfile(cache_path)

    # Touch data to ensure we have summary numbers
    steps = np.asarray(m.saved_steps if m.saved_steps is not None else [], dtype=int)
    n_snap = steps.size
    spc = m._steps_per_cycle(dt, T_cycle)
    Tmax = (steps.max() if n_snap else 0) * dt

    print(f"[done] {tag:9s} | N={Nx}x{Ny} | dt={dt:g} | steps={max_steps} | "
          f"spc={spc} | snapshots={n_snap} | Tmax≈{Tmax:g} | "
          f"saved={'yes' if exists else 'no'} | elapsed={format_interval(time.time() - t1)}")

    artifacts.append((tag, cache_path, exists, n_snap, spc, Tmax))


if GENERATE_DECOH_OFF:
    run_one(False)

if GENERATE_DECOH_ON:
    run_one(True)

# ----------------------------
# Report
# ----------------------------
print("\nArtifacts produced:")
for (tag, path, ok, n_snap, spc, Tmax) in artifacts:
    status = "OK" if ok else "MISSING"
    print(f"  mode={tag:9s}  [{status:8s}]  snaps={n_snap:4d}  spc={spc:3d}  Tmax≈{Tmax:g}  ->  {os.path.abspath(path)}")

print(f"\nTotal time elapsed: {format_interval(time.time() - t0)}\n")