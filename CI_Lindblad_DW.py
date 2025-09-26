import numpy as np
import math
import time
import os
import matplotlib.animation as animation
from matplotlib import pyplot as plt
from tqdm import tqdm


class CI_Lindblad_DW:
    """
    Lindbladian evolution for a single-layer two-point function G_{ij} = <c_i^† c_j>
    on an Nx×Ny torus with a domain-wall geometry in 'alpha'. Jump operators are
    overcomplete Wannier functions (OW). Optional decoherence term can be enabled.

    Caching:
      - Snapshots of G are saved at physical times that are integer multiples of
        T_cycle (i.e., at step k where k % steps_per_cycle == 0). If T_cycle is None,
        every step is saved.
      - Plotting methods are READ-ONLY: they load cached snapshots and never evolve.
      - __init__ tries to load; on miss (and init=True) it generates, samples, saves.

    Cache key fields:
      (Nx, Ny, dt, max_steps, steps_per_cycle, nshell, DW, decoh)
    """

    # ------------------------------ Init & Setup ------------------------------

    def __init__(self,
                 Nx, Ny,
                 init=True,                    # if True: on cache miss, evolve+save
                 decoh=True,
                 nshell=None,
                 DW=True,
                 dt=5e-2,
                 T_cycle=10,                 # physical cycle time; None => save each step
                 G_init=None,                  # optional initial state
                 keep_history=True,            # keep snapshots in memory after load/gen
                 overwrite=False):             # if True, ignore cache and regenerate

        self.Nx, self.Ny = int(Nx), int(Ny)
        self.decoh = bool(decoh)
        self.nshell = None if nshell is None else int(nshell)
        self.DW = bool(DW)

        # evolution defaults/currents
        self.dt_default = float(dt)
        self.max_steps_default = int(T_cycle/dt)
        self.T_cycle_default = None if T_cycle is None else float(T_cycle)
        self.keep_history_default = bool(keep_history)

        # Internal buffers/cached data
        self.G_last = None            # final snapshot (from cache or generation)
        self.G_history = []           # list of snapshots (if keep_history=True)
        self.saved_steps = None       # integer step indices corresponding to snapshots

        # Build OW structures needed ONLY for evolution (not for plotting)
        # Also define DW_loc for "smart x-positions" in plots.
        self._construct_OW_and_DW()

        # Try to load cache; else generate if requested
        if overwrite:
            self._generate_and_save(dt=self.dt_default,
                                    max_steps=self.max_steps_default,
                                    T_cycle=self.T_cycle_default,
                                    G_init=G_init)
        else:
            loaded = self._load_history(dt=self.dt_default,
                                        max_steps=self.max_steps_default,
                                        T_cycle=self.T_cycle_default)
            if not loaded and init:
                self._generate_and_save(dt=self.dt_default,
                                        max_steps=self.max_steps_default,
                                        T_cycle=self.T_cycle_default,
                                        G_init=G_init)

    # ---------------------------- Outdir & Caching ----------------------------

    def _ensure_outdir(self, path):
        os.makedirs(path, exist_ok=True)
        return path

    def _cache_dir(self):
        return self._ensure_outdir("cache/CI_Lindblad_DW")

    def _steps_per_cycle(self, dt, T_cycle):
        if T_cycle is None:
            return 1
        return max(1, int(round(float(T_cycle) / float(dt))))

    def _cache_key(self, dt, max_steps, T_cycle):
        spc = self._steps_per_cycle(dt, T_cycle)
        nsh = "None" if self.nshell is None else str(int(self.nshell))
        DW_tag = f"DW{int(self.DW)}"
        decoh_tag = f"decoh{int(self.decoh)}"
        return (f"N{self.Nx}x{self.Ny}_dt{dt:g}_steps{int(max_steps)}"
                f"_spc{int(spc)}_nshell{nsh}_{DW_tag}_{decoh_tag}")

    def _cache_path(self, dt, max_steps, T_cycle):
        return os.path.join(self._cache_dir(), self._cache_key(dt, max_steps, T_cycle) + ".npz")

    def _save_history(self, snapshots, steps, dt, max_steps, T_cycle):
        """
        snapshots: list of arrays, each (Nx,Ny,2,Nx,Ny,2)
        steps: list/array of integer step indices corresponding to snapshots
        """
        if len(snapshots) == 0:
            raise ValueError("No snapshots to save.")

        # store as object array to avoid huge contiguous stacks
        obj = np.empty((len(snapshots),), dtype=object)
        for i, G in enumerate(snapshots):
            obj[i] = np.asarray(G, dtype=np.complex128)

        path = self._cache_path(dt, max_steps, T_cycle)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        np.savez_compressed(
            path,
            G_history_objs=obj,
            steps=np.asarray(steps, dtype=np.int64),
            Nx=int(self.Nx),
            Ny=int(self.Ny),
            dt=float(dt),
            max_steps=int(max_steps),
            steps_per_cycle=int(self._steps_per_cycle(dt, T_cycle)),
            nshell=("None" if self.nshell is None else int(self.nshell)),
            DW=bool(self.DW),
            decoh=bool(self.decoh),
            DW_loc=np.array(self.DW_loc, dtype=int),
        )

        # keep in memory if requested
        self.G_history = [obj[i] for i in range(obj.size)]
        self.saved_steps = list(int(k) for k in steps)
        self.G_last = np.asarray(self.G_history[-1], dtype=np.complex128)

        return path

    def _load_history(self, dt, max_steps, T_cycle):
        """
        Try to load a cache matching (Nx,Ny,dt,max_steps,steps_per_cycle,nshell,DW,decoh).
        On success, populate self.G_history (if keep_history_default=True), self.G_last, self.saved_steps.
        Returns True/False.
        """
        path = self._cache_path(dt, max_steps, T_cycle)
        if not os.path.isfile(path):
            return False

        data = np.load(path, allow_pickle=True)
        # quick sanity checks (we assume the key matched already)
        if int(data["Nx"]) != self.Nx or int(data["Ny"]) != self.Ny:
            return False
        if bool(data["DW"]) != self.DW or bool(data["decoh"]) != self.decoh:
            return False

        obj = data["G_history_objs"]     # (T,) object array
        steps = data["steps"].astype(int).tolist()
        self.saved_steps = steps

        if self.keep_history_default:
            self.G_history = [obj[i] for i in range(obj.size)]
        else:
            # keep only last in memory to reduce RAM
            self.G_history = []
        self.G_last = np.asarray(obj[-1], dtype=np.complex128)

        # Store DW_loc from file (for smart x positions)
        try:
            self.DW_loc = tuple(np.asarray(data["DW_loc"]).tolist())
        except Exception:
            pass

        return True

    def ensure_history(self, dt=None, max_steps=None, T_cycle=None, G_init=None, overwrite=False):
        """
        Public: ensure there is a cache for the requested (dt,max_steps,T_cycle).
        Returns list of snapshots (may be empty if keep_history_default=False).
        """
        dt = self.dt_default if dt is None else float(dt)
        max_steps = self.max_steps_default if max_steps is None else int(max_steps)
        T_cycle = self.T_cycle_default if T_cycle is None else (None if T_cycle is None else float(T_cycle))

        if overwrite:
            self._generate_and_save(dt, max_steps, T_cycle, G_init)
        else:
            loaded = self._load_history(dt, max_steps, T_cycle)
            if not loaded:
                self._generate_and_save(dt, max_steps, T_cycle, G_init)
        return list(self.G_history)  # copy-ish

    # --------------------------- Evolution (generation) -----------------------

    def _generate_and_save(self, dt, max_steps, T_cycle, G_init):
        """Run RK4 once, sample snapshots at multiples of steps_per_cycle, and save to cache."""
        steps_per_cycle = self._steps_per_cycle(dt, T_cycle)

        # Ensure OW data exist (used only in Lindbladian)
        self._ensure_OW_ready(alpha_hint=1.0)

        # Initialize G
        Nx, Ny = self.Nx, self.Ny
        expected = (Nx, Ny, 2, Nx, Ny, 2)
        if G_init is None:
            G = np.zeros(expected, dtype=np.complex128)
        else:
            G = np.asarray(G_init, dtype=np.complex128)
            if G.shape != expected:
                raise ValueError(f"G_init must have shape {expected}, got {G.shape}.")

        snapshots = []
        saved_steps = []

        # Save step 0 if it aligns with sampling rule (always do it for convenience)
        snapshots.append(G.copy())
        saved_steps.append(0)

        # Evolution loop
        tmp = {}
        pbar = tqdm(range(1, int(max_steps) + 1), desc="evolving", unit="step")
        t0 = time.time()
        for k in pbar:
            G, tmp = self.rk4_Lindblad_evolver(G, dt, tmp=tmp)
            if (k % steps_per_cycle) == 0:
                snapshots.append(G.copy())
                saved_steps.append(k)
            pbar.set_postfix({
                "dt": f"{dt:.2e}",
                "saved": len(snapshots),
                "elapsed": f"{(time.time()-t0):.2f}s",
            })
        pbar.close()

        # Ensure the final step is included even if not a multiple (avoid duplicates)
        if saved_steps[-1] != max_steps:
            snapshots.append(G.copy())
            saved_steps.append(max_steps)

        # Persist
        self._save_history(snapshots, saved_steps, dt, max_steps, T_cycle)

    # ---------------------------- Plotting (read-only) ------------------------

    def plot_spectrum_vs_time(self, times=(0, 2, 10, 20, 40), filename=None, cmap='tab10'):
        """
        READ-ONLY. Plot eigenvalues of G at requested physical times.
        Uses cached snapshots and maps each time t to the nearest saved step index.

        Notes:
          - Requires a cache produced with the current (dt_default,max_steps_default,T_cycle_default).
          - If a requested t is beyond the simulated horizon, raises an error.
        """
        if self.G_last is None and not self._load_history(self.dt_default, self.max_steps_default, self.T_cycle_default):
            raise RuntimeError("No cached history for current parameters; generate it first (init=True or ensure_history).")

        dt = self.dt_default
        steps = np.asarray(self.saved_steps, dtype=int)
        Tmax = steps.max() * dt
        times = np.asarray(times, dtype=float)
        if np.any(times < 0) or np.any(times > Tmax + 1e-12):
            raise ValueError(f"Requested times must lie in [0, {Tmax:g}] for current cache.")

        # map times -> nearest saved step
        target_steps = np.rint(times / dt).astype(int)
        # For each target step, pick nearest available saved step
        k_map = np.empty_like(target_steps)
        for i, kt in enumerate(target_steps):
            j = np.argmin(np.abs(steps - kt))
            k_map[i] = steps[j]

        # Gather snapshots
        # If we didn't keep full history in RAM, load again and keep references
        if len(self.G_history) == 0:
            self._load_history(dt, self.max_steps_default, self.T_cycle_default)
        step_to_G = {int(k): np.asarray(self.G_history[idx] if idx < len(self.G_history) else None)
                     for idx, k in enumerate(self.saved_steps)}
        Nx, Ny = self.Nx, self.Ny
        Ntot = 2 * Nx * Ny

        outdir = self._ensure_outdir('figs/spectrum_vs_time')
        fig, ax = plt.subplots(figsize=(7.2, 5.6))
        colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(times)))

        for idx, (t, k) in enumerate(zip(times, k_map)):
            Gk = step_to_G[int(k)]
            if Gk is None:
                raise RuntimeError("Missing snapshot in memory; reload cache with keep_history=True.")
            vals = np.linalg.eigvalsh(Gk.reshape(Ntot, Ntot))
            ax.plot(vals.real, linestyle='None',
                    marker='o', markersize=3,
                    markerfacecolor=colors[idx], markeredgecolor='none',
                    label=f"t≈{t:g} (step {int(k)})")

        ax.set_ylabel(r"eigvals($G$)")
        ax.set_title(f"Spectrum of $G$ vs time (N={Nx}×{Ny}, dt={dt:g})")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        fig.tight_layout()

        if filename is None:
            tdesc = "-".join(f"{tt:g}" for tt in times)
            decoh_tag = "_decoh_on" if self.decoh else "_decoh_off"
            spc = self._steps_per_cycle(self.dt_default, self.T_cycle_default)
            filename = f"spectrum_vs_time_real_N{Nx}_dt{dt:g}_t_{tdesc}_spc{spc}{decoh_tag}.pdf"
        fullpath = os.path.join(outdir, filename)
        fig.savefig(fullpath, bbox_inches='tight')
        plt.close(fig)
        return fullpath

    def _smart_x_positions(self):
        """Return list[(x,label)] using domain-wall geometry if available."""
        Nx = self.Nx
        if hasattr(self, "DW_loc") and isinstance(self.DW_loc, (list, tuple)) and len(self.DW_loc) == 2:
            xL, xR = int(self.DW_loc[0]) % Nx, int(self.DW_loc[1]) % Nx
            xs = [
                (xL // 2) % Nx,
                (xL - 1) % Nx, xL % Nx, (xL + 1) % Nx,
                ((xL + xR) // 2) % Nx,
                (xR - 1) % Nx, xR % Nx, (xR + 1) % Nx,
                (xR + (Nx // 2)) % Nx,
            ]
            seen, uniq = set(), []
            for x in xs:
                xx = int(x)
                if xx not in seen:
                    uniq.append(xx); seen.add(xx)
            return [(x, f"{x}") for x in uniq]
        else:
            xs = np.linspace(0, Nx - 1, 9, dtype=int)
            return [(int(x), f"{int(x)}") for x in xs]

    def plot_corr_y_profiles(self, x_positions=None, ry_max=None, filename=None):
        """
        READ-ONLY. Plot squared two-point correlator vs r_y at fixed x columns
        probing the DW geometry.

        Uses ONLY the final cached snapshot (no evolution).
        """
        if self.G_last is None and not self._load_history(self.dt_default, self.max_steps_default, self.T_cycle_default):
            raise RuntimeError("No cached history for current parameters; generate it first (init=True or ensure_history).")

        Nx, Ny = self.Nx, self.Ny
        G = self.G_last

        # r_y range
        if ry_max is None:
            ry_max = Ny // 2
        ry_vals = np.arange(0, int(ry_max) + 1, dtype=int)

        # x positions
        if x_positions is None:
            x_positions = self._smart_x_positions()
        # normalize format to list of ints for looping; keep labels for annotations
        try:
            x_list = [(int(x), str(lbl)) for (x, lbl) in x_positions]
        except Exception:
            x_list = [(int(x), f"{int(x)}") for x in x_positions]

        outdir = self._ensure_outdir('figs/corr_y_profiles')
        fig, ax = plt.subplots(figsize=(7, 4.5))

        for x0, lbl in x_list:
            C_vec = self.squared_two_point_corr_xslice(G, x0=int(x0), ry=ry_vals).real
            line, = ax.plot(ry_vals, C_vec, marker='o', ms=3, lw=1, label=rf"$x_0={lbl}$")

            # Inline label at right edge
            finite = np.isfinite(C_vec)
            y_right = C_vec[finite][-1] if np.any(finite) else C_vec[-1]
            x_right = ry_vals[-1] * 1.02 if ry_vals[-1] > 0 else ry_vals[-1] + 0.5
            ax.annotate(lbl, xy=(ry_vals[-1], y_right), xytext=(x_right, y_right),
                        textcoords='data', ha='left', va='center', fontsize=9,
                        color=line.get_color())

        ax.set_xlabel(r"$r_y$")
        ax.set_ylabel(r"$C_G(x_0; r_y)$")
        ax.set_title(f"Squared correlator vs $r_y$ at fixed $x_0$ (N={Nx})")
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        leg = ax.legend(
            loc='lower left',
            bbox_to_anchor=(0.02, 0.02),
            ncol=4,
            fontsize=8,
            frameon=True,
            framealpha=0.85,
            borderpad=0.4,
            handlelength=1.5,
            handletextpad=0.6,
            columnspacing=0.9,
            labelspacing=0.3
        )
        fig.tight_layout()

        if filename is None:
            xdesc = "-".join(lbl for _, lbl in x_list)
            decoh_tag = "_decoh_on" if self.decoh else ""
            filename = f"corr2_y_profiles_N{Nx}_xs_{xdesc}{decoh_tag}.pdf"
        fullpath = os.path.join(outdir, filename)
        fig.savefig(fullpath, bbox_inches='tight')
        plt.close(fig)
        return fullpath

    def chern_marker_dynamics(self, fps=12, cmap='RdBu_r',
                              outbasename=None, vmin=-1.0, vmax=1.0):
        """
        READ-ONLY. Animate local Chern marker tanh C(r) over cached snapshots.
        Saves a GIF and a final PNG.
        """
        if (self.G_last is None or (self.keep_history_default and len(self.G_history) == 0)) \
           and not self._load_history(self.dt_default, self.max_steps_default, self.T_cycle_default):
            raise RuntimeError("No cached history for current parameters; generate it first (init=True or ensure_history).")

        # Need snapshots in memory for animation
        if len(self.G_history) == 0:
            # load again with keep_history True expectation; if still empty, we can't animate
            ok = self._load_history(self.dt_default, self.max_steps_default, self.T_cycle_default)
            if not ok or len(self.G_history) == 0:
                raise RuntimeError("History not stored in RAM; regenerate with keep_history=True to animate.")

        Nx, Ny = self.Nx, self.Ny
        outdir = self._ensure_outdir('figs/chern_marker')
        if outbasename is None:
            decoh_tag = 'decoh_on' if self.decoh else 'decoh_off'
            spc = self._steps_per_cycle(self.dt_default, self.T_cycle_default)
            outbasename = f"chern_marker_dynamics_N{Nx}_spc{spc}_{decoh_tag}"
        gif_path = os.path.join(outdir, outbasename + ".gif")
        final_path = os.path.join(outdir, outbasename + "_final.png")

        fig = plt.figure(figsize=(3.6, 4.0))
        ax = fig.add_subplot(111)
        im = ax.imshow(np.zeros((Nx, Ny)), cmap=cmap, vmin=vmin, vmax=vmax,
                       origin='upper', aspect='equal')
        for sp in ax.spines.values():
            sp.set_linewidth(1.5); sp.set_color('black')
        ax.set_xlabel("y"); ax.set_ylabel("x")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        writer = animation.PillowWriter(fps=fps)

        def _C_map(G):
            return self.local_chern_marker(G)

        with writer.saving(fig, gif_path, dpi=120):
            for t, Gt in zip(self.saved_steps, self.G_history):
                im.set_data(_C_map(Gt))
                ax.set_title(f"Local Chern marker (step={t})")
                writer.grab_frame()
        plt.close(fig)

        # Save final frame
        C_last = self.local_chern_marker(self.G_last)
        fig2, ax2 = plt.subplots(figsize=(3.6, 4.0))
        im2 = ax2.imshow(C_last, cmap=cmap, vmin=vmin, vmax=vmax, origin='upper', aspect='equal')
        fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        ax2.set_xlabel("y"); ax2.set_ylabel("x")
        ax2.set_title("Local Chern marker — final")
        fig2.savefig(final_path, bbox_inches='tight', dpi=140)
        plt.close(fig2)

        return gif_path, final_path, C_last, self.G_last

    # ----------------------------- Lindbladian core ---------------------------

    def _construct_OW_and_DW(self, alpha_init=1.0):
        """
        Build OW structures (P±, W’s, V’s) and set DW_loc for plot heuristics.
        """
        # Domain-wall alpha field
        if self.DW:
            alpha = np.full((self.Nx, self.Ny), 3.0, dtype=float)  # trivial outside
            half = self.Nx // 2
            w = int(np.floor(0.2 * self.Nx))
            x0 = max(0, half - w)
            x1 = min(self.Nx, half + w + 1)  # end-exclusive slice
            alpha[x0:x1, :] = 1.0            # topological slab
            self.alpha = alpha
            # Store approximate DW edges (inclusive indices)
            self.DW_loc = (int(x0), int(x1 - 1))
        else:
            self.alpha = np.ones((self.Nx, self.Ny), dtype=float)
            self.DW_loc = (0, self.Nx - 1)

        # k-grids
        kx = 2*np.pi * np.fft.fftfreq(self.Nx, d=1.0)
        ky = 2*np.pi * np.fft.fftfreq(self.Ny, d=1.0)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')

        # model vector n(k)
        nx = np.sin(KX)[:, :, None, None]
        ny = np.sin(KY)[:, :, None, None]
        nz = self.alpha[None, None, :, :] - np.cos(KX)[:, :, None, None] - np.cos(KY)[:, :, None, None]
        nmag = np.sqrt(nx**2 + ny**2 + nz**2)
        nmag = np.where(nmag == 0, 1e-15, nmag)

        # Pauli
        sx = np.array([[0, 1], [1, 0]], dtype=complex)
        sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sz = np.array([[1, 0], [0, -1]], dtype=complex)
        Id = np.eye(2, dtype=complex)

        hk = (nx[..., None, None]*sx + ny[..., None, None]*sy + nz[..., None, None]*sz) / nmag[..., None, None]
        self.Pminus = 0.5 * (Id - hk)
        self.Pplus  = 0.5 * (Id + hk)

        # Local τA/τB spinors
        tauA = (1/np.sqrt(2)) * np.array([[1], [1]], dtype=complex)
        tauB = (1/np.sqrt(2)) * np.array([[1], [-1]], dtype=complex)

        Rx_grid = np.arange(self.Nx)
        Ry_grid = np.arange(self.Ny)
        phase_x = np.exp(1j * KX[..., None, None] * Rx_grid[None, None, :, None])
        phase_y = np.exp(1j * KY[..., None, None] * Ry_grid[None, None, None, :])
        phase   = phase_x * phase_y

        def k2_to_r2(Ak):
            return np.fft.fft2(Ak, axes=(0, 1))

        def _square_window_mask(nshell):
            Nx, Ny = self.Nx, self.Ny
            x = np.arange(Nx)[:, None, None, None]
            y = np.arange(Ny)[None, :, None, None]
            Rx = np.arange(Nx)[None, None, :, None]
            Ry = np.arange(Ny)[None, None, None, :]
            dx_wrap = ((x - Rx + Nx//2) % Nx) - Nx//2
            dy_wrap = ((y - Ry + Ny//2) % Ny) - Ny//2
            return (np.abs(dx_wrap) <= nshell) & (np.abs(dy_wrap) <= nshell)

        def make_W(Pband, tau, phase):
            tau_dag = tau[:, 0].conj()
            psi_k   = np.einsum('m,...mn->...n', tau_dag, Pband)  # (...,2)

            F0 = phase * psi_k[..., 0]
            F1 = phase * psi_k[..., 1]
            W0 = k2_to_r2(F0)
            W1 = k2_to_r2(F1)
            W  = np.moveaxis(np.stack([W0, W1], axis=-1), -1, 2)  # (Nx,Ny,2,Rx,Ry)

            if self.nshell is not None:
                mask = _square_window_mask(self.nshell)          # (Nx,Ny,Rx,Ry)
                W = W * mask[:, :, None, :, :]
                norm2 = np.sum(np.abs(W)**2, axis=(0, 1, 2), keepdims=True)
                W = np.where(norm2 > 1e-15, W / (np.sqrt(norm2) + 1e-15), W)
            else:
                denom = np.sqrt(np.sum(np.abs(W)**2, axis=(0, 1, 2), keepdims=True)) + 1e-15
                W = W / denom
            return W

        self.W_A_plus  = make_W(self.Pplus,  tauA, phase)
        self.W_B_plus  = make_W(self.Pplus,  tauB, phase)
        self.W_A_minus = make_W(self.Pminus, tauA, phase)
        self.W_B_minus = make_W(self.Pminus, tauB, phase)

        def make_V(W):
            return np.einsum('ijklm, pqrlm -> ijkpqr', W, W.conj(), optimize=True)

        self.V_A_minus = make_V(self.W_A_minus)
        self.V_B_minus = make_V(self.W_B_minus)
        self.V_minus   = self.V_A_minus + self.V_B_minus

        self.V_A_plus  = make_V(self.W_A_plus)
        self.V_B_plus  = make_V(self.W_B_plus)
        self.V_plus    = self.V_A_plus + self.V_B_plus

    def _ensure_OW_ready(self, alpha_hint=1.0):
        """Paranoid guard: rebuild OW if missing (used only by evolution)."""
        needed = ("Pminus","Pplus","W_A_plus","W_B_plus","W_A_minus","W_B_minus",
                  "V_A_plus","V_B_plus","V_A_minus","V_B_minus","V_plus","V_minus")
        for a in needed:
            if not hasattr(self, a):
                self._construct_OW_and_DW(alpha_init=alpha_hint)
                break

    def Lgain(self, G, n_a):
        Y = -(1/2)*(np.einsum('ijklmn, lmnpqr -> ijkpqr', G, self.V_minus, optimize=True) +
                    np.einsum('ijklmn, lmnpqr -> ijkpqr', self.V_minus, G, optimize=True))
        return n_a*(self.V_minus + Y)

    def Lloss(self, G, n_a):
        Y = -((1-n_a)/2)*(np.einsum('ijklmn, lmnpqr -> ijkpqr', G, self.V_plus, optimize=True) +
                          np.einsum('ijklmn, lmnpqr -> ijkpqr', self.V_plus, G, optimize=True))
        return Y

    def double_comm(self, G, W, V):
        VG = np.einsum('ijklmn, lmnpqr -> ijkpqr', V, G, optimize=True)
        GV = np.einsum('ijklmn, lmnpqr -> ijkpqr', G, V, optimize=True)
        s_ab = np.einsum('ijkab, ijkpqr, pqrab -> ab', W.conj(), G, W, optimize=True)
        nonlinear = np.einsum('ijkab, ab, pqrab -> ijkpqr', W, s_ab, W.conj(), optimize=True)
        return (VG + GV) - 2.0 * nonlinear

    def Ldecoh(self, G, n_a):
        upper = self.double_comm(G, self.W_A_plus,  self.V_A_plus)  + self.double_comm(G, self.W_B_plus,  self.V_B_plus)
        lower = self.double_comm(G, self.W_A_minus, self.V_A_minus) + self.double_comm(G, self.W_B_minus, self.V_B_minus)
        return -(1/2)*((2-n_a)*lower + (1+n_a)*upper)

    def Lcycle(self, G, n_a=0.5):
        if self.decoh:
            return self.Lgain(G, n_a) + self.Lloss(G, n_a) + self.Ldecoh(G, n_a)
        else:
            return self.Lgain(G, n_a) + self.Lloss(G, n_a)

    def rk4_Lindblad_evolver(self, G, dt, n_a=0.5, tmp=None):
        if tmp is None: tmp = {}
        k1 = tmp.get('k1'); k2 = tmp.get('k2'); k3 = tmp.get('k3'); k4 = tmp.get('k4'); Y = tmp.get('Y')
        if k1 is None: k1 = tmp['k1'] = np.empty_like(G)
        if k2 is None: k2 = tmp['k2'] = np.empty_like(G)
        if k3 is None: k3 = tmp['k3'] = np.empty_like(G)
        if k4 is None: k4 = tmp['k4'] = np.empty_like(G)
        if Y  is None: Y  = tmp['Y']  = np.empty_like(G)

        k1[:] = self.Lcycle(G, n_a)
        np.multiply(k1, 0.5*dt, out=Y); np.add(G, Y, out=Y)
        k2[:] = self.Lcycle(Y, n_a)
        np.multiply(k2, 0.5*dt, out=Y); np.add(G, Y, out=Y)
        k3[:] = self.Lcycle(Y, n_a)
        np.multiply(k3, dt, out=Y); np.add(G, Y, out=Y)
        k4[:] = self.Lcycle(Y, n_a)

        np.add(k1, k4, out=Y)
        np.add(Y, 2.0*k2, out=Y)
        np.add(Y, 2.0*k3, out=Y)
        G += (dt/6.0) * Y
        return G, tmp

    # ------------------------------ Analysis utils ----------------------------

    def squared_two_point_corr(self, G, rx=0, ry=0):
        G = np.asarray(G)
        Nx, Ny, s1, Nx2, Ny2, s2 = G.shape
        if (s1, s2) != (2, 2) or (Nx, Ny) != (Nx2, Ny2):
            raise ValueError("G must have shape (Nx, Ny, 2, Nx, Ny, 2).")
        X, Y = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')
        rx_arr = np.atleast_1d(rx).astype(int)
        ry_arr = np.atleast_1d(ry).astype(int)
        Xb = X[:, :, None, None]
        Yb = Y[:, :, None, None]
        Xp = (Xb + rx_arr[None, None, :, None]) % Nx
        Yp = (Yb + ry_arr[None, None, None, :]) % Ny
        blocks = G[Xb, Yb, :, Xp, Yp, :]
        C = np.sum(np.abs(blocks)**2, axis=(0, 1, 4, 5)) / (2.0 * Nx * Ny)
        return C

    def squared_two_point_corr_xslice(self, G, x0=0, ry=0):
        G = np.asarray(G)
        Nx, Ny, s1, Nx2, Ny2, s2 = G.shape
        if (s1, s2) != (2, 2) or (Nx, Ny) != (Nx2, Ny2):
            raise ValueError("G must have shape (Nx, Ny, 2, Nx, Ny, 2).")
        x0 = int(x0) % Nx
        Y = np.arange(Ny, dtype=np.intp)[:, None]
        ry_arr = np.atleast_1d(ry).astype(np.intp)
        R = ry_arr.size
        Yp = (Y + ry_arr[None, :]) % Ny
        Gx = G[x0, :, :, x0, :, :]                       # (Ny,2,Ny,2)
        Gx_re = np.transpose(Gx, (0, 2, 1, 3)).reshape(Ny*Ny, 2, 2)
        flat_idx = (Y * Ny + Yp).reshape(-1)
        blocks = Gx_re[flat_idx].reshape(Ny, R, 2, 2)
        C = np.sum(np.abs(blocks)**2, axis=(0, 2, 3)) / (2.0 * Ny)
        return C

    # -------------------------- Real-space Chern number -----------------------

    def _build_tripartition_masks(self, R_frac=0.4):
        """Build A,B,C masks inside a disk (on-the-fly; not stored)."""
        Nx, Ny = self.Nx, self.Ny
        R = R_frac * min(Nx, Ny)
        xref, yref = Nx // 2, Ny // 2
        inside = np.zeros((Nx, Ny), dtype=bool)
        A = np.zeros_like(inside)
        B = np.zeros_like(inside)
        C = np.zeros_like(inside)
        rr = R * R
        ymax = int(math.floor(R))
        a2 = 2*np.pi/3
        a4 = 4*np.pi/3
        for dy in range(-ymax, ymax + 1):
            y = yref + dy
            if y < 0 or y >= Ny:
                continue
            max_dx = int(math.floor(math.sqrt(max(0.0, rr - dy*dy))))
            x0 = max(0, xref - max_dx)
            x1 = min(Nx - 1, xref + max_dx)
            if x0 > x1:
                continue
            inside[x0:x1+1, y] = True
            dxs = np.arange(x0, x1+1) - xref
            dys = np.full_like(dxs, dy)
            theta = np.mod(np.arctan2(dys, dxs), 2*np.pi)
            A[x0:x1+1, y] = (theta >= 0)  & (theta < a2)
            B[x0:x1+1, y] = (theta >= a2) & (theta < a4)
            C[x0:x1+1, y] = (theta >= a4) & (theta < 2*np.pi)
        return A, B, C, inside

    def real_space_chern_number(self, G=None, A_mask=None, B_mask=None, C_mask=None):
        """
        Compute 12π i [ Tr(P_CA P_AB P_BC) - Tr(P_AC P_CB P_BA) ] with P = G^*.
        Builds tri-partition masks on demand if not supplied.
        """
        if G is None:
            if self.G_last is None and not self._load_history(self.dt_default, self.max_steps_default, self.T_cycle_default):
                raise RuntimeError("No cached final state available; generate first.")
            G = self.G_last

        Nx, Ny = self.Nx, self.Ny
        expected = (Nx, Ny, 2, Nx, Ny, 2)
        if np.asarray(G).shape != expected:
            raise ValueError(f"G must have shape {expected}.")

        if A_mask is None or B_mask is None or C_mask is None:
            A_mask, B_mask, C_mask, _ = self._build_tripartition_masks()

        P = np.asarray(G, dtype=complex, order='C').conj().reshape(2*Nx*Ny, 2*Nx*Ny)

        def sector_indices(mask_xy):
            sites = np.flatnonzero(mask_xy.ravel(order='C'))
            return np.concatenate((2*sites, 2*sites + 1))

        iA = sector_indices(A_mask); iB = sector_indices(B_mask); iC = sector_indices(C_mask)
        P_CA = P[np.ix_(iC, iA)]; P_AB = P[np.ix_(iA, iB)]; P_BC = P[np.ix_(iB, iC)]
        P_AC = P[np.ix_(iA, iC)]; P_CB = P[np.ix_(iC, iB)]; P_BA = P[np.ix_(iB, iA)]
        t1 = np.trace(P_CA @ P_AB @ P_BC)
        t2 = np.trace(P_AC @ P_CB @ P_BA)
        Y = 12 * np.pi * 1j * (t1 - t2)
        return np.real_if_close(Y, tol=1e-6)