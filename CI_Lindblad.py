import numpy as np
import math
import time
from IPython.display import clear_output
from matplotlib import pyplot as plt
import os

class CI_Lindblad:
    def __init__(self, Nx, Ny, decoh = True, alpha = 1.0, nshell = None):
        """
        1) Build overcomplete Wannier spinors for a Chern insulator model.

        2) Generate masks for each partition A,B,C for computation of real space chern number
        """
        self.Nx, self.Ny = Nx, Ny
        self.alpha = alpha
        self.decoh = decoh
        self.construct_OW_functions(self.alpha, nshell = nshell) # construct Wannier functions

        # lazy-evolution cache (filled on demand)
        self.G_last   = None
        self.G_history = []
        self.evo_dt    = 5e-2
        self.evo_steps = 500
        # evolve once so the instance is immediately usable
        self._ensure_evolved(dt=self.evo_dt, max_steps=self.evo_steps, keep_history=True)
    def _ensure_evolved(self, dt=5e-2, max_steps=500, keep_history=True, G_init=None):
        """Make sure self.G_last/self.G_history exist for the requested (dt,max_steps).
        Re-runs evolution only when necessary."""
        need = (self.G_last is None or
                getattr(self, 'evo_dt', None) != dt or
                getattr(self, 'evo_steps', None) != int(max_steps) or
                (keep_history and (self.G_history is None or len(self.G_history) < int(max_steps))))
        if need:
            G, hist = self.G_evolution(max_steps=int(max_steps), dt=dt,
                                       G_init=G_init, keep_history=keep_history)
            # G_evolution updates cache already; nothing else to do
            return G, hist
        return self.G_last, self.G_history

        # --- Build tri-partition masks A, B, C inside a circle of radius R ---
        R = 0.4 * min(Nx, Ny)
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
            max_dx = int(math.floor(math.sqrt(rr - dy*dy)))
            x0 = max(0, xref - max_dx)
            x1 = min(Nx - 1, xref + max_dx)
            if x0 > x1:
                continue

            inside[x0:x1+1, y] = True
            dxs = np.arange(x0, x1+1) - xref
            dys = np.full_like(dxs, dy)
            theta = np.mod(np.arctan2(dys, dxs), 2*np.pi)

            A[x0:x1+1, y] = (theta >= 0)   & (theta < a2)
            B[x0:x1+1, y] = (theta >= a2)  & (theta < a4)
            C[x0:x1+1, y] = (theta >= a4)  & (theta < 2*np.pi)

        # store on the instance
        self.A_mask = A
        self.B_mask = B
        self.C_mask = C
        self.inside_mask = inside
        self.R = R
        self.ref = (xref, yref)
    
        print('Class CI_Lindblad has been Initialized')

    def construct_OW_functions(self, alpha, nshell=None):
        '''
        Build overcomplete Wannier spinors for a *uniform* Chern-insulator
        with scalar alpha (no domain wall).

        Produces four fields on the instance:
          self.W_A_plus, self.W_B_plus, self.W_A_minus, self.W_B_minus
        each of shape (Nx, Ny, 2, Nx, Ny) with axes:
          (x, y, mu, R_x, R_y)
        and normalized per (R_x, R_y): sum_{x,y,mu} |W|^2 = 1.

        If nshell is provided (integer), the real-space Wannier functions W are truncated
        to a square window of size (2*nshell+1)×(2*nshell+1) centered at each (R_x, R_y)
        using minimal-image (periodic) distances on the torus, and then renormalized
        per center.
        '''
        # store scalar alpha
        self.alpha = float(alpha)
        self.nshell = nshell

        Nx, Ny = int(self.Nx), int(self.Ny)

        # --- k-grids (FFT ordering) ---
        kx = 2*np.pi * np.fft.fftfreq(Nx, d=1.0)   # (Nx,)
        ky = 2*np.pi * np.fft.fftfreq(Ny, d=1.0)   # (Ny,)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')  # (Nx,Ny)

        # --- model vector n(k) ---
        nx = np.sin(KX)          # (Nx,Ny)
        ny = np.sin(KY)          # (Nx,Ny)
        nz = self.alpha - np.cos(KX) - np.cos(KY)
        nmag = np.sqrt(nx**2 + ny**2 + nz**2)
        nmag = np.where(nmag == 0, 1e-15, nmag)

        # (optional) keep handy for debugging/inspection
        self.nx, self.ny, self.nz, self.nmag = nx, ny, nz, nmag

        # --- Pauli matrices and identity ---
        pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
        pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
        Id2 = np.eye(2, dtype=complex)

        # --- k-space single-particle h(k) = n̂ · σ, projectors P±(k) ---
        hk = (nx[..., None, None] * pauli_x +
              ny[..., None, None] * pauli_y +
              nz[..., None, None] * pauli_z) / nmag[..., None, None]   # (Nx,Ny,2,2)

        self.Pminus = 0.5 * (Id2 - hk)   # (Nx,Ny,2,2)
        self.Pplus  = 0.5 * (Id2 + hk)   # (Nx,Ny,2,2)

        # --- local orbital choices (columns) ---
        tauA = (1/np.sqrt(2)) * np.array([[1], [1]], dtype=complex)    # (2,1)
        tauB = (1/np.sqrt(2)) * np.array([[1], [-1]], dtype=complex)   # (2,1)

        # --- phases for all centers (R_x, R_y) ---
        Rx_grid = np.arange(Nx)  # (Nx,)
        Ry_grid = np.arange(Ny)  # (Ny,)
        phase_x = np.exp(1j * KX[..., None, None] * Rx_grid[None, None, :, None])  # (Nx,Ny,Nx,1)
        phase_y = np.exp(1j * KY[..., None, None] * Ry_grid[None, None, None, :])  # (Nx,Ny,1,Ny)
        phase   = phase_x * phase_y  # (Nx,Ny,Nx,Ny)

        # --- FFT over k-axes only ---
        def k2_to_r2(Ak):  # Ak shape (Nx,Ny,*,*)  (the trailing axes are broadcast centers)
            return np.fft.fft2(Ak, axes=(0, 1))

        # --- optional square window mask for truncation (Nx,Ny,Rx,Ry) ---
        def _square_window_mask(nshell_val):
            if nshell_val is None:
                return None
            x = np.arange(Nx)[:, None, None, None]   # (Nx,1,1,1)
            y = np.arange(Ny)[None, :, None, None]   # (1,Ny,1,1)
            Rx = np.arange(Nx)[None, None, :, None]  # (1,1,Nx,1)
            Ry = np.arange(Ny)[None, None, None, :]  # (1,1,1,Ny)
            dx = ((x - Rx + Nx//2) % Nx) - Nx//2     # minimal-image separation
            dy = ((y - Ry + Ny//2) % Ny) - Ny//2
            return (np.abs(dx) <= nshell_val) & (np.abs(dy) <= nshell_val)  # (Nx,Ny,Nx,Ny)

        shell_mask = _square_window_mask(nshell)

        # ------------------------------
        # helper to build a normalized W for ALL centers (uniform alpha)
        # psi_k(k) = tau^† P_band(k) → (Nx,Ny,2)
        # ------------------------------
        def make_W(Pband, tau):
            tau_dag = tau[:, 0].conj()                                   # (2,)
            psi_k   = np.einsum('m,ijmn->ijn', tau_dag, Pband)           # (Nx,Ny,2)

            F0 = phase * psi_k[:, :, 0][..., None, None]                 # (Nx,Ny,Nx,Ny)
            F1 = phase * psi_k[:, :, 1][..., None, None]                 # (Nx,Ny,Nx,Ny)
            W0 = k2_to_r2(F0)                                            # (Nx,Ny,Nx,Ny)
            W1 = k2_to_r2(F1)                                            # (Nx,Ny,Nx,Ny)

            # Stack μ=0,1 as axis 2 → (Nx,Ny,2,Nx,Ny)
            W = np.moveaxis(np.stack([W0, W1], axis=-1), -1, 2)

            # Truncation (if requested), then per-center normalization
            if shell_mask is not None:
                W = W * shell_mask[:, :, None, :, :]                     # (Nx,Ny,2,Nx,Ny)

            denom2 = np.sum(np.abs(W)**2, axis=(0, 1, 2), keepdims=True) # (1,1,1,Nx,Ny)
            W = np.where(denom2 > 1e-15, W / (np.sqrt(denom2) + 1e-15), 0.0)
            return W

        # Build four overcomplete Wannier spinors for ALL centers: (Nx,Ny,2,Nx,Ny)
        self.W_A_plus  = make_W(self.Pplus,  tauA)
        self.W_B_plus  = make_W(self.Pplus,  tauB)
        self.W_A_minus = make_W(self.Pminus, tauA)
        self.W_B_minus = make_W(self.Pminus, tauB)

        # Build V = Σ_R |W_R⟩⟨W_R| (sum over centers), for each family
        def make_V(W):
            # W: (i,j,k,l,m) = (x,y,μ,Rx,Ry)
            # V: (i,j,k,p,q,r) by contracting over centers l,m:
            return np.einsum('ijklm,pqrlm->ijkpqr', W, W.conj(), optimize=True)

        self.V_A_minus = make_V(self.W_A_minus)
        self.V_B_minus = make_V(self.W_B_minus)
        self.V_minus   = self.V_A_minus + self.V_B_minus  # sum over ν

        self.V_A_plus  = make_V(self.W_A_plus)
        self.V_B_plus  = make_V(self.W_B_plus)
        self.V_plus    = self.V_A_plus + self.V_B_plus    # sum over ν
    
    def Lgain(self, G, n_a):
        '''
        G in (Nx, Ny, 2, Nx, Ny, 2)
        '''
        Y = -(1/2)*(np.einsum('ijklmn, lmnpqr -> ijkpqr', G, self.V_minus, optimize=True) + np.einsum('ijklmn, lmnpqr -> ijkpqr', self.V_minus, G, optimize=True))   
        return n_a*(self.V_minus + Y)
    
    def Lloss(self, G, n_a):
        '''
        G in (Nx, Ny, 2, Nx, Ny, 2)            
        '''
        Y = -((1-n_a)/2)*(np.einsum('ijklmn, lmnpqr -> ijkpqr', G, self.V_plus, optimize=True) + np.einsum('ijklmn, lmnpqr -> ijkpqr', self.V_plus, G, optimize=True))   
        return Y
    
    def double_comm(self, G, W, V):
        '''
        G in (Nx, Ny, 2, Nx, Ny, 2)
        '''

        # Linear part: P G + G P
        VG = np.einsum('ijklmn, lmnpqr -> ijkpqr', V, G, optimize=True)
        GV = np.einsum('ijklmn, lmnpqr -> ijkpqr', G, V, optimize=True)
        linear_term = VG + GV

        # Scalar weights per center: s_ab = <w_ab| G |w_ab>
        s_ab = np.einsum('ijkab, ijkpqr, pqrab -> ab', W.conj(), G, W, optimize=True)

        # Nonlinear term: Σ_ab s_ab * |w_ab><w_ab|
        nonlinear_term = np.einsum('ijkab, ab, pqrab -> ijkpqr', W, s_ab, W.conj(), optimize=True)

        Y = linear_term - 2.0 * nonlinear_term
        return Y
    
    def Ldecoh(self, G, n_a):
        '''
        G in (Nx, Ny, 2, Nx, Ny, 2)            
        '''
        upperband_term = self.double_comm(G, self.W_A_plus, self.V_A_plus) + self.double_comm(G, self.W_B_plus, self.V_B_plus)
        lowerband_term = self.double_comm(G, self.W_A_minus, self.V_A_minus) + self.double_comm(G, self.W_B_minus, self.V_B_minus)
        
        Y = -(1/2)*((2-n_a)*lowerband_term + (1+n_a)*upperband_term)
        return Y
    
    def Lcycle(self, G, n_a):
        '''
        G in (Nx, Ny, 2, Nx, Ny, 2)            
        '''
        if self.decoh:
            Y = self.Lgain(G, n_a) + self.Lloss(G, n_a) + self.Ldecoh(G, n_a)
        else:
            Y = self.Lgain(G, n_a) + self.Lloss(G, n_a)
        return Y
        
    def rk4_Lindblad_evolver(self, G, dt, n_a = 0.5, tmp=None):
        """
        In-place RK4 for G' = Lindblad(G). Returns G (updated).
        tmp: optional dict of preallocated buffers {'k1','k2','k3','k4','Y'}
        """
        if tmp is None:
            tmp = {}
        k1 = tmp.get('k1'); k2 = tmp.get('k2'); k3 = tmp.get('k3'); k4 = tmp.get('k4'); Y = tmp.get('Y')
        # allocate once with the right dtype/shape
        if k1 is None: k1 = tmp['k1'] = np.empty_like(G)
        if k2 is None: k2 = tmp['k2'] = np.empty_like(G)
        if k3 is None: k3 = tmp['k3'] = np.empty_like(G)
        if k4 is None: k4 = tmp['k4'] = np.empty_like(G)
        if Y  is None: Y  = tmp['Y']  = np.empty_like(G)
        # k1 = f(G)
        k1[:] = self.Lcycle(G, n_a)
        # k2 = f(G + dt/2 * k1)
        np.multiply(k1, 0.5*dt, out=Y); np.add(G, Y, out=Y)
        k2[:] = self.Lcycle(Y, n_a)
        # k3 = f(G + dt/2 * k2)
        np.multiply(k2, 0.5*dt, out=Y); np.add(G, Y, out=Y)
        k3[:] = self.Lcycle(Y, n_a)
        # k4 = f(G + dt * k3)
        np.multiply(k3, dt, out=Y); np.add(G, Y, out=Y)
        k4[:] = self.Lcycle(Y, n_a)
        # G += dt/6 * (k1 + 2k2 + 2k3 + k4)
        # Use Y as accumulator: Y = k1 + 2k2 + 2k3 + k4
        np.add(k1, k4, out=Y)
        np.add(Y, 2.0*k2, out=Y)
        np.add(Y, 2.0*k3, out=Y)
        G += (dt/6.0) * Y
        return G, tmp
        

        
    def G_evolution(self, max_steps = 500, dt = 5e-2, G_init=None, keep_history=True, dtype=complex):
        """
        Evolve the real-space correlator/state G under G' = Lcycle(G) using RK4.

        Parameters
        ----------
        dt : float
            Time step.
        max_steps : int
            Number of RK4 steps to take.
        n_a : float, optional
            Parameter forwarded to Lcycle.
        G_init : ndarray or None, optional
            Initial state. If None, uses the zero-charge state with shape
            (Nx, Ny, 2, Nx, Ny, 2) where Nx=self.Nx, Ny=self.Ny.
        keep_history : bool, optional
            If True, records G after each step; length of the returned list is max_steps.
        dtype : dtype, optional
            Dtype for the zero state if G_init is None.

        Returns
        -------
        G : ndarray
            Final state with shape (Nx, Ny, 2, Nx, Ny, 2).
        G_history : list of ndarray
            If keep_history=True, a list of length max_steps with copies of G at each step;
            otherwise an empty list.
        """
        Nx, Ny = int(self.Nx), int(self.Ny)

        # Initialize G
        if G_init is None:
            G = np.zeros((Nx, Ny, 2, Nx, Ny, 2), dtype=dtype)
        else:
            G = np.array(G_init, copy=True)
            expected_shape = (Nx, Ny, 2, Nx, Ny, 2)
            if G.shape != expected_shape:
                raise ValueError(f"G_init must have shape {expected_shape}, got {G.shape}")

        G_history = []

        # Main evolution loop with live progress
        # Insert timer start
        t_start = time.time()
        for step in range(1, int(max_steps) + 1):
            t0 = time.time()
            G, tmp = self.rk4_Lindblad_evolver(G, dt)
            iter_time = time.time() - t0
            if keep_history:
                G_history.append(G.copy())
            # live status line
            clear_output(wait=True)
            print(f"[G_evolution] iter {step}/{int(max_steps)} | dt={dt:.3e} | itertime={iter_time:.3e}s | total={time.time()-t_start:.3e}s | alpha={self.alpha:g} | N=({self.Nx},{self.Ny})")

        # Update cache attributes
        self.G_last = G
        self.G_history = G_history if keep_history else []
        self.evo_dt = dt
        self.evo_steps = int(max_steps)
        return G, G_history

    # -----------------------------
    # Chern number 
    # -----------------------------

    def real_space_chern_number(self, G=None, A_mask=None, B_mask=None, C_mask=None):
        """
        Compute the real-space Chern number using the projector built from G.

        This implements the same formula as `chern_from_projector`, but takes the
        input as a general operator G and forms P = G.conj() before the block traces.

        Parameters
        ----------
        G : ndarray, shape (Nx, Ny, 2, Nx, Ny, 2)
            Real-space operator. The projector used here is P = G.conj().
        A_mask, B_mask, C_mask : (Nx,Ny) boolean masks, optional
            If None, use the instance's stored tri-partition masks.

        Returns
        -------
        complex
            12π i [ Tr(P_CA P_AB P_BC) - Tr(P_AC P_CB P_BA) ].
        """
        if G is None:
            self._ensure_evolved()
            G = self.G_last
        Nx, Ny = int(self.Nx), int(self.Ny)
        expected = (Nx, Ny, 2, Nx, Ny, 2)
        if G.shape != expected:
            raise ValueError(f"G must have shape {expected}, got {G.shape}")

        # Default masks from the instance unless provided
        A_mask = self.A_mask if A_mask is None else A_mask
        B_mask = self.B_mask if B_mask is None else B_mask
        C_mask = self.C_mask if C_mask is None else C_mask

        # Build projector from G: P = G.conj()
        Ntot = Nx * Ny * 2
        P = np.asarray(G, dtype=complex, order='C').conj().reshape(Ntot, Ntot)

        def sector_indices_from_mask(mask_xy):
            """
            Return flattened (x,y,s) indices that include both orbitals for all True sites.
            """
            sites = np.flatnonzero(mask_xy.ravel(order='C'))  # j = x*Ny + y
            return np.concatenate((2*sites, 2*sites + 1))     # include both orbitals s=0,1
        
        # Sector indices include both orbitals for each selected site
        iA = sector_indices_from_mask(A_mask)
        iB = sector_indices_from_mask(B_mask)
        iC = sector_indices_from_mask(C_mask)

        P_CA = P[np.ix_(iC, iA)]
        P_AB = P[np.ix_(iA, iB)]
        P_BC = P[np.ix_(iB, iC)]

        P_AC = P[np.ix_(iA, iC)]
        P_CB = P[np.ix_(iC, iB)]
        P_BA = P[np.ix_(iB, iA)]

        t1 = np.trace(P_CA @ P_AB @ P_BC)
        t2 = np.trace(P_AC @ P_CB @ P_BA)

        Y = 12 * np.pi * 1j * (t1 - t2)
        return np.real_if_close(Y, tol=1e-6)

    def squared_two_point_corr(self, G=None, rx=0, ry=0):
        """
        Squared two-point correlator on a torus for separations r = (rx, ry).

            C_G(rx, ry) = (1 / (2 Nx Ny)) * sum_{mu,mu', r'}
                           | G[(r',mu), (r'+(rx,ry), mu')] |^2

        This version returns a **2D array** over all combinations of `rx` and `ry`:
        shape = (len(rx), len(ry)). If either `rx` or `ry` is a scalar, it is
        treated as a 1-element list, yielding shape (1, len(ry)) or (len(rx), 1).

        Periodic boundary conditions are enforced by modular addition in x and y.

        Parameters
        ----------
        G : ndarray, shape (Nx, Ny, 2, Nx, Ny, 2)
            Real-space two-point function (can be complex).
        rx, ry : int or array-like of int
            Lattice separations along x and y. Scalars or 1D arrays.

        Returns
        -------
        C : ndarray, shape (len(rx), len(ry))
            2D grid of squared two-point correlators for all (rx, ry) pairs.
        """
        if G is None:
            self._ensure_evolved()
            G = self.G_last
        G = np.asarray(G)
        Nx, Ny, s1, Nx2, Ny2, s2 = G.shape
        if (s1, s2) != (2, 2) or (Nx, Ny) != (Nx2, Ny2):
            raise ValueError("G must have shape (Nx, Ny, 2, Nx, Ny, 2).")

    def squared_two_point_corr_xslice(self, G=None, x0=0, ry=0):
        if G is None:
            self._ensure_evolved()
            G = self.G_last
        G = np.asarray(G)
        Nx, Ny, s1, Nx2, Ny2, s2 = G.shape
        x0 = int(x0) % Nx
        Y  = np.arange(Ny)[:, None]
        ry_arr = np.atleast_1d(ry).astype(int)
        Yp = (Y + ry_arr[None, :]) % Ny
        Yb = np.broadcast_to(Y, Yp.shape)
        blocks = G[x0, Yb, :, x0, Yp, :]
        C = np.sum(np.abs(blocks)**2, axis=(0, 2, 3)) / (2.0 * Ny)
        return C

        # Prepare base grids of starting sites r' = (x,y)
        X, Y = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')

        # Ensure rx, ry are 1D integer arrays
        rx_arr = np.atleast_1d(rx).astype(int)
        ry_arr = np.atleast_1d(ry).astype(int)

        # Vectorized over all (rx, ry): build broadcastable index grids
        Rx, Ry = rx_arr.size, ry_arr.size
        Xb = X[:, :, None, None]                      # (Nx,Ny,1,1)
        Yb = Y[:, :, None, None]                      # (Nx,Ny,1,1)
        Xp = (Xb + rx_arr[None, None, :, None]) % Nx  # (Nx,Ny,Rx,1)
        Yp = (Yb + ry_arr[None, None, None, :]) % Ny  # (Nx,Ny,1,Ry)

        # Advanced indexing automatically broadcasts index arrays to a common shape.
        # Result has shape (Nx,Ny,Rx,Ry,2,2)
        blocks = G[Xb, Yb, :, Xp, Yp, :]

        # Sum over r'=(x,y) and spins (mu, mu') → (Rx,Ry)
        C = np.sum(np.abs(blocks)**2, axis=(0, 1, 4, 5)) / (2.0 * Nx * Ny)
        return C

    def G_CI(self, alpha = 1.0, norm='backward', k_is_centered=False):
        """
        Construct and return the real-space two-point function G = P_minuns.conj()
        for the Chern-insulator Hamiltonian defined by (Nx, Ny, alpha)
        at half-filling.

        Parameters
        ----------
        alpha : float or None
            If None, uses self.alpha. Otherwise overrides.
        norm : {'backward','ortho','forward'}
            FFT normalization passed to np.fft.ifft2.
        k_is_centered : bool
            If True, assumes nk is fftshifted (k=0 at center) before IFFT.

        Returns
        -------
        Pminus_realspace : ndarray, shape (Nx, Ny, 2, Nx, Ny, 2)
            Real-space projector onto the occupied band.
        """

        Nx, Ny = int(self.Nx), int(self.Ny)

        if alpha != self.alpha:
            self.construct_OW_functions(alpha)

        # --- vectorized: k → relative real-space for full (Nx,Ny,2,2) tensor ---
        def _k_to_r_rel_full(Pk): # Pk shape (Nx,Ny,2,2)
            # Optional un-centering in k
            if k_is_centered:
                Pk = np.fft.ifftshift(Pk, axes=(0, 1))
            # IFFT over k-axes only
            PR = np.fft.ifft2(Pk, axes=(0, 1), norm=norm) # (Nx,Ny,2,2)
            PR = np.real_if_close(PR, tol=1e3)
            # Build relative-coordinate indexers once (broadcasted)
            x = np.arange(Nx); y = np.arange(Ny)
            X  = x[:, None, None, None]
            Xp = x[None, None, :, None]
            Y  = y[None, :, None, None]
            Yp = y[None, None, None, :]
            dX = (X - Xp) % Nx # (Nx,1,Nx,1)
            dY = (Y - Yp) % Ny # (1,Ny,1,Ny)
            
            # Advanced indexing lifts (Nx,Ny,2,2) → (Nx,Ny,Nx,Ny,2,2)
            return PR[dX, dY, :, :]

        # Transform entire Pminus at once
        Pminus_rel = _k_to_r_rel_full(self.Pminus) # (Nx,Ny,Nx,Ny,2,2)
       
        # Reorder to (Nx,Ny,2,Nx,Ny,2) and use G = P.conj()
        G = np.moveaxis(Pminus_rel.conj(), 4, 2)

        return G

    def chern_dynamics_vs_time(self, alpha = 1.0, max_steps=500, G_init=None, keep_history=False, dtype=complex):
        """
        Evolve G with RK4 and compute the Chern number at each time step (fixed alpha).
        Uses cached evolution if available.
        """
        # If alpha is changed, reconstruct and invalidate cache.
        if alpha != self.alpha:
            self.construct_OW_functions(alpha)
            self.G_last = None
            self.G_history = []
        # Ensure evolution (always keep history for dynamics)
        self._ensure_evolved(dt=5e-2, max_steps=int(max_steps), keep_history=True, G_init=G_init)
        # Use cached history
        chern_vals = [self.real_space_chern_number(Gi) for Gi in self.G_history]
        ts = np.arange(1, len(chern_vals)+1, dtype=float)  # time steps (1-based)
        return ts, np.asarray(chern_vals), self.G_last, self.G_history

    def chern_steady_vs_alpha(self, alpha_list):
        """
        For each alpha in `alpha_list`, build a fresh CI_Lindblad at that alpha,
        evolve to `max_steps`, and return the (real-space) Chern number of the
        final state G.

        Parameters
        ----------
        alphas : array-like
            Sequence of alpha values to scan.
        dt : float
            Time step for evolution.
        max_steps : int
            Number of RK4 steps for the steady-state approximation.
        n_a : float, optional
            Parameter forwarded to Lcycle.
        dtype : dtype, optional
            Dtype for zero initial state.

        Returns
        -------
        alpha_list : ndarray
            Array of alpha values.
        chern_list : ndarray
            Chern number (complex) for the final G at each alpha.
        """
        alpha_list = np.asarray(alpha_list, dtype=float)
        chern_list = np.empty_like(alpha_list, dtype=complex)

        for i, a in enumerate(alpha_list):
            solver = CI_Lindblad(Nx=self.Nx, Ny=self.Ny, decoh=self.decoh, alpha=a)
            solver._ensure_evolved(dt=5e-2, max_steps=500, keep_history=False)
            chern_list[i] = solver.real_space_chern_number(solver.G_last)

        return chern_list
    
    def find_saturation_timestep(self, dt=5e-2, tol=1e-3, max_steps=100000, G_init=None, dtype=complex):
        """
        Find the earliest RK4 step at which the real-space Chern number saturates to 1
        within a specified tolerance, using alpha fixed to 1.

        This method time-evolves G starting from a zero-charge state (or provided G_init)
        with the RK4 integrator and checks after each step whether
            abs(Re(Chern(G)) - 1) <= tol.
        It stops at the first step that satisfies the criterion and returns that step index
        (1-based) and the corresponding physical time (step*dt).

        Parameters
        ----------
        dt : float
            Time step for RK4.
        tol : float, optional
            Convergence tolerance on |Re(Chern) - 1|. Default 1e-4.
        max_steps : int, optional
            Maximum number of steps to attempt before giving up (default 100000).
        n_a : float, optional
            Parameter forwarded to Lcycle.
        G_init : ndarray or None, optional
            If provided, must have shape (Nx, Ny, 2, Nx, Ny, 2). If None, uses zeros.
        dtype : dtype, optional
            Dtype for zero initial state if G_init is None.

        Returns
        -------
        step_idx : int
            The first step index (1-based) at which the tolerance is met. If not met,
            returns max_steps.
        t_final : float
            The physical time corresponding to that step (step_idx * dt).
        Chern_val : complex
            The Chern number at the returned step.
        G : ndarray
            The state G at the returned step.
        """
        Nx, Ny = int(self.Nx), int(self.Ny)

        # Initialize G
        if G_init is None:
            G = np.zeros((Nx, Ny, 2, Nx, Ny, 2), dtype=dtype)
        else:
            expected = (Nx, Ny, 2, Nx, Ny, 2)
            if G_init.shape != expected:
                raise ValueError(f"G_init must have shape {expected}, got {G_init.shape}")
            G = np.array(G_init, copy=True)

        step_idx = int(max_steps)
        ch_val = None

        # Insert timer start
        t_start = time.time()
        for step in range(1, int(max_steps) + 1):
            t0 = time.time()
            G, _ = self.rk4_Lindblad_evolver(G, dt)
            ch = self.real_space_chern_number(G)
            err = abs(ch.real - 1.0)
            iter_time = time.time() - t0
            print(f"[iter {step:6d}/{int(max_steps)}] t={step*dt:.6f} | itertime={iter_time:.3e}s | total={time.time()-t_start:.3e}s | |Re(Chern)-1|={err:.6e} | trG/(2NxNy)={(np.reshape(G,(2*Nx*Ny, 2*Nx*Ny)).trace().real)/(2*Nx*Ny):.3f}")
            clear_output(wait = True)
            # Check saturation against target Chern = 1 using real part
            if err <= tol:
                step_idx = step
                ch_val = ch
                break
        else:
            # If never broke out, record the last values
            ch_val = ch  # from the final iteration

        t_final = step_idx * dt
        return step_idx, t_final, ch_val, G


    # -----------------------------
    # Plotting helpers
    # -----------------------------
    
    def _ensure_outdir(self, subpath):
        """Create (if needed) and return an output directory path under CWD.
        subpath can include nested folders like 'figs/chern_dynamics'."""
        outdir = os.path.join(os.getcwd(), subpath)
        os.makedirs(outdir, exist_ok=True)
        return outdir
    
    def plot_chern_dynamics_vs_time(self, N_list=(11, 21, 31), alpha=None,
                                    dt=5e-2, max_steps=500, filename=None):
        """
        Plot real-space Chern number vs time for fixed alpha and varying system size.

        Parameters
        ----------
        N_list : iterable of int
            System sizes (square lattices N×N) to plot.
        alpha : float or None
            Chern-insulator mass parameter. If None, uses self.alpha.
        dt : float
            RK4 time step.
        max_steps : int
            Number of steps.
        filename : str or None
            PDF filename to save. If None, a descriptive one is generated.
        """
        alpha_val = float(self.alpha if alpha is None else alpha)
        outdir = self._ensure_outdir('figs/chern_dynamics')

        fig, ax = plt.subplots(figsize=(7, 4.5))
        marker_list = ['o', 'x', '*']
        for idx, N in enumerate(N_list):
            solver = CI_Lindblad(Nx=N, Ny=N, decoh=self.decoh, alpha=alpha_val)
            solver._ensure_evolved(dt=dt, max_steps=int(max_steps), keep_history=True)
            ch = np.array([solver.real_space_chern_number(Gi).real for Gi in solver.G_history])
            ts = np.arange(1, len(ch)+1) * dt
            ax.plot(ts, ch, label=f"N={N}", marker=marker_list[idx], markevery=int(1/dt))

        ax.set_xlabel("t")
        ax.set_ylabel("real-space Chern number")
        ax.set_title(f"Chern dynamics vs t (alpha={alpha_val:g})")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()

        if filename is None:
            Ns = "-".join(str(n) for n in N_list)
            if self.decoh:
                filename = f"chern_dynamics_vs_time_alpha{alpha_val:g}_Ns{Ns}_decoh_on.pdf"
            else:
                filename = f"chern_dynamics_vs_time_alpha{alpha_val:g}_Ns{Ns}_decoh_off.pdf"
        # Save to outdir (filename is just the basename, not a path)
        fullpath = os.path.join(outdir, filename)
        plt.tight_layout()
        plt.show()
        fig.savefig(fullpath, bbox_inches='tight')
        plt.close(fig)
        return fullpath

    def plot_chern_dynamics_vs_time_nshell(self, nshell_list=(None, 1, 2, 3), alpha=None,
                                           dt=5e-2, max_steps=500, filename=None):
        """
        Plot real-space Chern number vs time for a *fixed* system size (self.Nx,self.Ny),
        fixed alpha, and varying truncation window `nshell` used to build the overcomplete
        Wannier functions.

        Parameters
        ----------
        nshell_list : iterable of int or None
            Sequence of truncation radii to compare. Use None to denote "no truncation".
        alpha : float or None
            Chern-insulator mass parameter to use for all curves. If None, uses self.alpha.
        dt : float
            RK4 time step.
        max_steps : int
            Number of RK4 steps.
        filename : str or None
            PDF filename to save. If None, a descriptive one is generated and saved under
            figs/chern_dynamics_nshell/ .

        Returns
        -------
        fullpath : str
            Full path to the saved PDF.
        """
        Nx, Ny = int(self.Nx), int(self.Ny)
        alpha_val = float(self.alpha if alpha is None else alpha)

        outdir = self._ensure_outdir('figs/chern_dynamics_nshell')

        fig, ax = plt.subplots(figsize=(7, 4.5))
        # a few distinct markers to rotate through
        marker_cycle = ['o', 's', 'x', '^', 'D', '*', 'P']
        mevery = max(1, int(round(1.0/dt)))  # show a marker about once per unit time

        for i, ns in enumerate(nshell_list):
            # Build a solver with the requested nshell (same size, decoh flag, and alpha)
            solver = CI_Lindblad(Nx=Nx, Ny=Ny, decoh=self.decoh, alpha=alpha_val, nshell=ns)
            # Use the built-in chern dynamics routine (handles its own progress prints)
            ts, ch, _, _ = solver.chern_dynamics_vs_time(alpha=alpha_val,
                                                         max_steps=int(max_steps),
                                                         keep_history=False)
            label_ns = '∞' if ns is None else str(int(ns))
            ax.plot(ts * dt, ch.real,
                    label=f"nshell={label_ns}",
                    marker=marker_cycle[i % len(marker_cycle)],
                    markevery=mevery,
                    lw=1.2)

        ax.set_xlabel("t")
        ax.set_ylabel("real-space Chern number")
        ax.set_title(f"Chern dynamics vs t (N={Nx}, alpha={alpha_val:g})")
        ax.grid(True, alpha=0.3)
        ax.legend(title="truncation")
        plt.tight_layout()

        if filename is None:
            # Encode nshell list succinctly in the filename
            def _ns_desc(vals):
                out = []
                for v in vals:
                    out.append('inf' if v is None else str(int(v)))
                return '-'.join(out)
            ns_desc = _ns_desc(nshell_list)
            decoh_tag = '_decoh_on' if self.decoh else '_decoh_off'
            filename = f"chern_dyn_vs_time_N{Nx}_alpha{alpha_val:g}_nshells_{ns_desc}_steps{int(max_steps)}{decoh_tag}.pdf"

        fullpath = os.path.join(outdir, filename)
        plt.show()
        fig.savefig(fullpath, bbox_inches='tight')
        plt.close(fig)
        return fullpath

    def plot_steady_chern_vs_alpha(self, N_list=(11, 21, 31), alpha_list=None,
                                   dt=5e-2, max_steps=500, filename=None):
        """
        Plot steady-state (after max_steps) Chern number vs alpha for varying N.

        Parameters
        ----------
        N_list : iterable of int
            System sizes (square lattices N×N).
        alpha_list : iterable of float, optional
            Sequence of alpha values. If None, defaults to linspace(-4, 4, 161).
        dt : float
            RK4 time step.
        max_steps : int
            Number of steps (default 500).
        filename : str or None
            PDF filename to save. If None, a descriptive one is generated.
        """
        if alpha_list is None:
            alphas = np.linspace(-4.0, 4.0, 161)
        else:
            alphas = np.asarray(alpha_list, dtype=float)

        outdir = self._ensure_outdir('figs/chern_vs_alpha')
        # Start total timing
        t_total_start = time.time()

        fig, ax = plt.subplots(figsize=(7, 4.5))
        nN = len(N_list)
        nA = len(alphas)
        for iN, N in enumerate(N_list, start=1):
            ch_vals = []
            N_start = time.time()
            for ia, a in enumerate(alphas, start=1):
                a_start = time.time()
                solver = CI_Lindblad(Nx=N, Ny=N, decoh=self.decoh, alpha=a)
                solver._ensure_evolved(dt=dt, max_steps=int(max_steps), keep_history=False)
                ch_val = solver.real_space_chern_number(solver.G_last)
                ch_vals.append(ch_val.real)
                # Alpha (inner outer-loop) progress line
                alpha_iter_time = time.time() - a_start
                total_elapsed   = time.time() - t_total_start
                clear_output(wait=True)
                print(
                    f"[steady_vs_alpha] N_iter {iN}/{nN} | alpha_iter {ia}/{nA} | "
                    f"alpha={a:g} | N={N} | alpha_itertime={alpha_iter_time:.3e}s | total={total_elapsed:.3e}s"
                )
            # Optionally, report per-N block duration
            N_time = time.time() - N_start
            ax.plot(alphas, np.array(ch_vals), label=f"N={N}", marker = 'o')

        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("real-space Chern number")
        ax.set_title(f"Steady-state Chern vs $\\alpha$ (steps={int(max_steps)})")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()

        if filename is None:
            Ns = "-".join(str(n) for n in N_list)
            if self.decoh:
                filename = f"chern_steady_vs_alpha_steps{int(max_steps)}_Ns{Ns}_decoh_on.pdf"
            else:
                filename = f"chern_steady_vs_alpha_steps{int(max_steps)}_Ns{Ns}_decoh_off.pdf"

        fullpath = os.path.join(outdir, filename)
        plt.tight_layout()
        plt.show()
        fig.savefig(fullpath, bbox_inches='tight')
        plt.close(fig)
        return fullpath

    def plot_squared_corr_vs_alpha(self, direction='x', alpha_list=None,
                                   dt=5e-2, max_steps=500, filename=None):
        """
        Plot 1D squared correlator C(r) vs r along a chosen path, overlaying
        curves for multiple alpha values.

        Direction options
        -----------------
        direction='x'   : use separations (r, 0) with r = 0..Nx//2  → C(r) = C_G(r, 0)
        direction='y'   : use separations (0, r) with r = 0..Ny//2  → C(r) = C_G(0, r)
        direction='diag': use separations (r, r) with r = 0..min(Nx,Ny)//2 → C(r) = C_G(r, r)

        Parameters
        ----------
        direction : {'x','y','diag'}
            Which path in (r_x,r_y) to plot along.
        alpha_list : iterable of float, optional
            Sequence of alpha values. If None, defaults to linspace(-4,4,161).
        dt : float
            RK4 time step.
        max_steps : int
            Number of steps for steady-state approximation.
        filename : str or None
            Basename of the PDF to save. If None, a descriptive name is generated
            that reflects the chosen direction and r-range.

        Returns
        -------
        fullpath : str
            Full path to the saved PDF.
        """
        # Alphas to compare
        if alpha_list is None:
            alphas = np.linspace(-4.0, 4.0, 161)
        else:
            alphas = np.asarray(alpha_list, dtype=float)

        if direction not in ('x', 'y', 'diag'):
            raise ValueError("direction must be 'x', 'y', or 'diag'")

        Nx, Ny = int(self.Nx), int(self.Ny)
        if direction == 'x':
            r_vals = np.arange(0, Nx//2 + 1, dtype=int)  # r = |rx|
            rx_arr = r_vals
            ry_arr = np.array([0], dtype=int)
            ylabel = r"$C_G(r, 0)$"
            title_dir = 'x'
        elif direction == 'y':
            r_vals = np.arange(0, Ny//2 + 1, dtype=int)  # r = |ry|
            rx_arr = np.array([0], dtype=int)
            ry_arr = r_vals
            ylabel = r"$C_G(0, r)$"
            title_dir = 'y'
        else:  # 'diag'
            rmax = min(Nx, Ny)//2
            r_vals = np.arange(0, rmax + 1, dtype=int)  # r = |rx| = |ry|
            rx_arr = r_vals
            ry_arr = r_vals
            ylabel = r"$C_G(r, r)$"
            title_dir = 'diag'

        outdir = self._ensure_outdir('figs/corr_vs_r')

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for a in alphas:
            solver = CI_Lindblad(Nx=Nx, Ny=Ny, decoh=self.decoh, alpha=a)
            solver._ensure_evolved(dt=dt, max_steps=int(max_steps), keep_history=False)
            C_grid = solver.squared_two_point_corr(G=solver.G_last, rx=rx_arr, ry=ry_arr)

            if direction == 'diag':
                # C_grid has shape (len(r), len(r)); take the diagonal entries
                L = r_vals.size
                C_vec = C_grid[np.arange(L), np.arange(L)].astype(float)
            else:
                # C_grid is (len(r),1) or (1,len(r)); flatten to vector of length len(r)
                C_vec = C_grid.reshape(-1).astype(float)

            ax.plot(r_vals, C_vec, marker='o', ms=3, lw=1, label=f"alpha={a:g}")

        ax.set_xlabel(r"$r$")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Steady-state |G|$^2$ vs r along {title_dir}-path (steps={int(max_steps)})")
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()

        # Build descriptive filename
        if filename is None:
            # Encode alpha list succinctly
            if alphas.size <= 6:
                a_desc = '-'.join(f"{v:g}" for v in alphas)
            else:
                a_desc = f"{alphas.size}vals_{alphas.min():g}to{alphas.max():g}"
            r_desc = f"r0-{r_vals[-1]}"
            if self.decoh:
                filename = f"corr2_1D_vs_r_dir{title_dir}_N{Nx}_{r_desc}_alphas_{a_desc}_steps{int(max_steps)}_decoh_on.pdf"
            else:
                filename = f"corr2_1D_vs_r_dir{title_dir}_N{Nx}_{r_desc}_alphas_{a_desc}_steps{int(max_steps)}.pdf"

        fullpath = os.path.join(outdir, filename)
        plt.show()
        fig.savefig(fullpath, bbox_inches='tight')
        plt.close(fig)
        return fullpath

    # -----------------------------
    # Pretty scatter-plotter (amp→size, phase→color) for a chosen center
    # -----------------------------
    def plot_wanniers(self, R=(0, 0), alpha=None, nshell = None, size_max=450, gamma=0.8,
                      cmap='hsv', figsize=(12, 6), savepath=None, show=True):
        """
        Visualize the four overcomplete Wannier spinors W_{ν,±}(r, μ; R) at a chosen center R=(Rx,Ry)
        as scatter plots: point size encodes amplitude, color encodes phase.

        Parameters
        ----------
        R : tuple(int,int)
            Center (Rx,Ry) at which to display the Wannier functions. 0-based indices on the torus.
        alpha : float or None
            If provided and different from self.alpha, rebuild the OW functions for this alpha.
        nshell : positive integer < min(Nx, Ny)
            Define square truncation window about Wannier center
        size_max : float
            Max scatter marker size (points^2).
        gamma : float
            Nonlinearity for size scaling: size ∝ (|W|/max|W|)^gamma.
        cmap : str or Colormap
            Colormap for phase (default 'hsv').
        figsize : tuple
            Figure size passed to plt.subplots.
        savepath : str or None
            If given, save the figure to this path.
        show : bool
            If True (default), display the figure with plt.show().

        Returns
        -------
        fig, axes : Matplotlib Figure and Axes array
        """
        Nx, Ny = int(self.Nx), int(self.Ny)
        Rx, Ry = R

        # Rebuild if alpha requested is different from current
        if alpha is None:
            alpha = self.alpha
        
        self.construct_OW_functions(alpha, nshell)

        # The constructor stores W_* with shape (Nx, Ny, 2, Nx, Ny), axes (x,y,μ,Rx,Ry).
        # Slice out the chosen center (Rx,Ry), leaving (Nx,Ny,2).
        W_A_minus = self.W_A_minus[:, :, :, Rx, Ry]  # (Nx,Ny,2)
        W_B_minus = self.W_B_minus[:, :, :, Rx, Ry]
        W_A_plus  = self.W_A_plus[:,  :, :, Rx, Ry]
        W_B_plus  = self.W_B_plus[:,  :, :, Rx, Ry]

        # fftshift each component so the chosen center appears visually centered.
        fftshift = np.fft.fftshift
        panels = [
            (fftshift(W_A_minus[:, :, 0]), r'$W_{R,A,-}(\mathbf{r},1)$', 'a'),
            (fftshift(W_B_minus[:, :, 0]), r'$W_{R,B,-}(\mathbf{r},1)$', 'b'),
            (fftshift(W_A_plus [:, :, 0]), r'$W_{R,A,+}(\mathbf{r},1)$', 'c'),
            (fftshift(W_B_plus [:, :, 0]), r'$W_{R,B,+}(\mathbf{r},1)$', 'd'),
            (fftshift(W_A_minus[:, :, 1]), r'$W_{R,A,-}(\mathbf{r},2)$', 'e'),
            (fftshift(W_B_minus[:, :, 1]), r'$W_{R,B,-}(\mathbf{r},2)$', 'f'),
            (fftshift(W_A_plus [:, :, 1]), r'$W_{R,A,+}(\mathbf{r},2)$', 'g'),
            (fftshift(W_B_plus [:, :, 1]), r'$W_{R,B,+}(\mathbf{r},2)$', 'h'),
        ]

        # Coordinates centered about zero for a nicer view
        xs = np.arange(Nx) - Nx // 2
        ys = np.arange(Ny) - Ny // 2
        X, Y = np.meshgrid(xs, ys, indexing='ij')

        fig, axes = plt.subplots(2, 4, figsize=figsize, constrained_layout=True)
        norm = plt.Normalize(vmin=-np.pi, vmax=np.pi)
        cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap

        for ax, (W, title, letter) in zip(axes.flat, panels):
            amp = np.abs(W)
            max_amp = amp.max() + 1e-15
            sizes = (amp / max_amp)**gamma * size_max
            phase = np.angle(W)

            ax.scatter(X.ravel(), Y.ravel(), c=phase.ravel(), s=sizes.ravel(),
                       cmap=cmap_obj, norm=norm, linewidths=0, alpha=0.95)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(title, fontsize=12)
            ax.text(0.04, 0.94, f'({letter})', transform=ax.transAxes,
                    ha='left', va='top', fontsize=12)

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.9)
        cbar.set_label('Phase', fontsize=11)
        cbar.set_ticks([-np.pi, 0, np.pi])
        cbar.set_ticklabels([r'$-\pi$', r'$0$', r'$\pi$'])

        # Optional save
        if savepath is not None:
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
            fig.savefig(savepath, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig, axes