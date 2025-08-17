import numpy as np
import math
from IPython.display import clear_output

class CI_Lindblad:
    def __init__(self, Nx, Ny, alpha = 1.0):
        """
        1) Build overcomplete Wannier spinors for a Chern insulator model.

        2) Generate masks for each partition A,B,C for computation of real space chern number
        """
        self.Nx, self.Ny = Nx, Ny

        self.construct_OW_functions(alpha) # construct wannier functions

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

    def construct_OW_functions(self, alpha):
        '''
        Produces four fields:
          self.W_A_plus, self.W_B_plus, self.W_A_minus, self.W_B_minus,
        each of shape (Nx, Ny, Nx, Ny, 2) with axes:
          (x, y, R_x, R_y, mu)
        and normalized per (R_x, R_y): sum_{x,y,mu} |W|^2 = 1.
        '''

        # k-grids in radians per lattice spacing (FFT ordering)
        kx = 2*np.pi * np.fft.fftfreq(self.Nx, d=1.0)     # shape (Nx,)
        ky = 2*np.pi * np.fft.fftfreq(self.Ny, d=1.0)     # shape (Ny,)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')  # shape (Nx, Ny)

        # model vector n(k)
        self.nx = np.sin(KX)
        self.ny = np.sin(KY)
        self.nz = alpha - np.cos(KX) - np.cos(KY)
        self.nmag = np.sqrt(self.nx**2 + self.ny**2 + self.nz**2)
        self.nmag = np.where(self.nmag == 0, 1e-15, self.nmag)  # avoid divide-by-zero

        # --- Pauli matrices ---
        pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
        pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)

        # construct 2x2 identity (broadcasts over k-grid)
        Id = np.eye(2, dtype=complex)

        # construct k-space single-particle h(k) = n̂ · σ (unit vector)
        hk = (self.nx[..., None, None] * pauli_x +
              self.ny[..., None, None] * pauli_y +
              self.nz[..., None, None] * pauli_z) / self.nmag[..., None, None]  # (Nx,Ny,2,2)

        # construct upper and lower band projectors
        self.Pminus = 0.5 * (Id - hk)   # (Nx,Ny,2,2)
        self.Pplus  = 0.5 * (Id + hk)   # (Nx,Ny,2,2)

        # local choice of orbitals
        tauA = (1/np.sqrt(2)) * np.array([[1], [1]], dtype=complex)   # (2,1)
        tauB = (1/np.sqrt(2)) * np.array([[1], [-1]], dtype=complex)  # (2,1)

        # --- phases for all centers (R_x, R_y) ---
        Rx_grid = np.arange(self.Nx)                                   # (Nx,)
        Ry_grid = np.arange(self.Ny)                                   # (Ny,)
        phase_x = np.exp(1j * KX[..., None, None] * Rx_grid[None, None, :, None])  # (Nx,Ny,Nx,1)
        phase_y = np.exp(1j * KY[..., None, None] * Ry_grid[None, None, None, :])  # (Nx,Ny,1,Ny)
        phase   = phase_x * phase_y                                # (Nx,Ny,Nx,Ny)

        # Helper: do FFT over k-axes only (0,1), for every (R_x,R_y)
        def k2_to_r2(Ak):  # Ak shape (Nx, Ny, Nx, Ny)
            return np.fft.fft2(Ak, axes=(0, 1))

        # ------------------------------
        # helper to build a normalized W for every center
        # row spinor in k-space: psi_k = tau^† P(k)
        # ------------------------------
        def make_W(Pband, tau, phase, Nx, Ny):
            # tau^\dagger P(k): (2,)^* with (...,2,2) over left index -> (...,2)
            tau_dag = tau[:, 0].conj()                              # (2,)
            psi_k   = np.einsum('m,...mn->...n', tau_dag, Pband)    # (Nx,Ny,2)

            # Broadcast psi_k components over centers, then FFT over k-axes
            F0 = phase * psi_k[:, :, 0][..., None, None]            # (Nx,Ny,Nx,Ny)
            F1 = phase * psi_k[:, :, 1][..., None, None]            # (Nx,Ny,Nx,Ny)
            W0 = k2_to_r2(F0)                                       # (Nx,Ny,Nx,Ny)
            W1 = k2_to_r2(F1)                                       # (Nx,Ny,Nx,Ny)

            # Stack μ=0,1 as the 3rd axis → (Nx,Ny,2,Nx,Ny)
            W  = np.moveaxis(np.stack([W0, W1], axis=-1), -1, 2)

            # Normalize per center (R_x,R_y) across (x,y,μ)
            denom = np.sqrt(np.sum(np.abs(W)**2, axis=(0, 1, 2), keepdims=True)) + 1e-15
            return W / denom

        # Build the four overcomplete Wannier spinors for ALL centers: (Nx,Ny,2,Nx,Ny)
        self.W_A_plus  = make_W(self.Pplus,  tauA, phase, self.Nx, self.Ny)
        self.W_B_plus  = make_W(self.Pplus,  tauB, phase, self.Nx, self.Ny)
        self.W_A_minus = make_W(self.Pminus, tauA, phase, self.Nx, self.Ny)
        self.W_B_minus = make_W(self.Pminus, tauB, phase, self.Nx, self.Ny)


    def make_V(self, W):
        V = np.einsum('ijklm, pqrlm -> ijkpqr', W, W.conj(), optimize=True) # sum over Wannier centers 
        return V
    
    def Lgain(self, G, n_a):
        '''
        G in (Nx, Ny, 2, Nx, Ny, 2)
        '''
        V_A_minus = self.make_V(self.W_A_minus)
        V_B_minus = self.make_V(self.W_B_minus)
        V = V_A_minus + V_B_minus # sum over local orbital choices
        Y = -(1/2)*(np.einsum('ijklmn, lmnpqr -> ijkpqr', G, V, optimize=True) + np.einsum('ijklmn, lmnpqr -> ijkpqr', V, G, optimize=True))   
        return n_a*(V + Y)
    
    def Lloss(self, G, n_a):
        '''
        G in (Nx, Ny, 2, Nx, Ny, 2)            
        '''
        V_A_plus = self.make_V(self.W_A_plus)
        V_B_plus = self.make_V(self.W_B_plus)
        V = V_A_plus + V_B_plus # sum over local orbital choices
        Y = -((1-n_a)/2)*(np.einsum('ijklmn, lmnpqr -> ijkpqr', G, V, optimize=True) + np.einsum('ijklmn, lmnpqr -> ijkpqr', V, G, optimize=True))   
        return Y
    
    def double_comm(self, G, W):
        '''
        G in (Nx, Ny, 2, Nx, Ny, 2)
        Implements [P, [P, G]] with P = sum_ab |w_ab><w_ab| where w_ab = W[...,ab].
        linear_term = P G + G P
        nonlinear_term = sum_ab W[...,ab] ( W[...,ab]^† G W[...,ab] ) W[...,ab]^†
        '''
        # Build the projector P = Σ_ab |w_ab><w_ab| from W
        P = self.make_V(W)  # (Nx,Ny,2,Nx,Ny,2)

        # Linear part: P G + G P
        PG = np.einsum('ijklmn, lmnpqr -> ijkpqr', P, G, optimize=True)
        GP = np.einsum('ijklmn, lmnpqr -> ijkpqr', G, P, optimize=True)
        linear_term = PG + GP

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
        upperband_term = self.double_comm(G, self.W_A_plus) + self.double_comm(G, self.W_B_plus)
        lowerband_term = self.double_comm(G, self.W_A_minus) + self.double_comm(G, self.W_B_minus)
        
        Y = -(1/2)*((2-n_a)*lowerband_term + (1+n_a)*upperband_term)
        return Y
    
    def Lcycle(self, G, n_a):
        '''
        G in (Nx, Ny, 2, Nx, Ny, 2)            
        '''
        Y = self.Lgain(G, n_a) + self.Lloss(G, n_a) + self.Ldecoh(G, n_a)
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
        

        
    def G_evolution(self, max_steps, dt=1e-2, n_a=0.5, G_init=None, keep_history=True, dtype=complex):
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

        # Preallocated buffers threaded across steps
        tmp = None

        # Main evolution loop
        for _ in range(int(max_steps)):
            G, tmp = self.rk4_Lindblad_evolver(G, dt, n_a=n_a, tmp=tmp)
            if keep_history:
                G_history.append(G.copy())

        return G, G_history

    # -----------------------------
    # Chern number utilities & scans
    # -----------------------------
    def _sector_indices_from_mask(self, mask_xy):
        """Return flattened (x,y,s) indices that include both orbitals for all True sites."""
        Nx, Ny = int(self.Nx), int(self.Ny)
        sites = np.flatnonzero(mask_xy.ravel(order='C'))  # j = x*Ny + y
        return np.concatenate((2*sites, 2*sites + 1))     # include both orbitals s=0,1

    def real_space_chern_number(self, G, A_mask=None, B_mask=None, C_mask=None):
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

        # Sector indices include both orbitals for each selected site
        iA = self._sector_indices_from_mask(A_mask)
        iB = self._sector_indices_from_mask(B_mask)
        iC = self._sector_indices_from_mask(C_mask)

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

    def G_CI(self, alpha=1.0, norm='backward', k_is_centered=False):
        """
        Construct and return the real-space two-point function
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

    def chern_dynamics_vs_time(self, max_steps, dt=1e-2, n_a=0.5, G_init=None, keep_history=False, dtype=complex):
        """
        Evolve G with RK4 and compute the Chern number at each time step (fixed alpha).

        Parameters
        ----------
        dt : float
            Time step.
        max_steps : int
            Number of RK4 steps.
        n_a : float, optional
            Parameter forwarded to Lcycle.
        G_init : ndarray or None, optional
            Initial state. If None, uses zeros with shape (Nx,Ny,2,Nx,Ny,2).
        keep_history : bool, optional
            If True, returns a list of G copies for each step; otherwise returns an empty list.
        dtype : dtype, optional
            Dtype for zero initial state if G_init is None.

        Returns
        -------
        ts : ndarray, shape (max_steps,)
            Times at the end of each step (dt, 2dt, ..., max_steps*dt).
        chern_vals : ndarray, shape (max_steps,)
            Chern number at each step (real part may be taken by caller).
        G : ndarray
            Final state after max_steps.
        G_history : list[ndarray]
            If keep_history, list of length max_steps with copies of G after each step.
        """
        Nx, Ny = int(self.Nx), int(self.Ny)
        if G_init is None:
            G = np.zeros((Nx, Ny, 2, Nx, Ny, 2), dtype=dtype)
        else:
            expected = (Nx, Ny, 2, Nx, Ny, 2)
            if G_init.shape != expected:
                raise ValueError(f"G_init must have shape {expected}, got {G_init.shape}")
            G = np.array(G_init, copy=True)

        chern_vals = np.empty(int(max_steps), dtype=complex)
        ts = np.arange(1, int(max_steps)+1, dtype=float) * dt
        G_history = [] if keep_history else []
        tmp = None

        for step in range(int(max_steps)):
            G, tmp = self.rk4_Lindblad_evolver(G, dt, n_a=n_a, tmp=tmp)
            chern_vals[step] = self.real_space_chern_number(G)
            if keep_history:
                G_history.append(G.copy())

        return ts, chern_vals, G, G_history

    def chern_steady_vs_alpha(self, alphas, max_steps, dt=1e-2, n_a=0.5, dtype=complex):
        """
        For each alpha in `alphas`, build a fresh CI_Lindblad at that alpha,
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
        alphas : ndarray
            Array of alpha values.
        chern_vals : ndarray
            Chern number (complex) for the final G at each alpha.
        """
        alphas = np.asarray(alphas, dtype=float)
        Nx, Ny = int(self.Nx), int(self.Ny)
        cherns = np.empty_like(alphas, dtype=complex)

        for i, a in enumerate(alphas):
            # zero initial state
            G0 = np.zeros((Nx, Ny, 2, Nx, Ny, 2), dtype=dtype)
            # evolve (no need to keep history)
            G, _ = self.G_evolution(dt, max_steps, n_a=n_a, G_init=G0, keep_history=False, dtype=dtype)
            cherns[i] = self.real_space_chern_number(G)

        return alphas, cherns
    
    def find_saturation_timestep(self, dt=5e-2, tol=1e-4, max_steps=100000, n_a=0.5, G_init=None, dtype=complex):
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

        tmp = None
        step_idx = int(max_steps)
        ch_val = None

        for step in range(1, int(max_steps) + 1):
            G, tmp = self.rk4_Lindblad_evolver(G, dt, n_a=n_a, tmp=tmp)
            ch = self.real_space_chern_number(G)
            err = abs(abs(ch.real) - 1.0)
            print(f"[iter {step:6d}] t = {step*dt:.6f} |Re(Chern)-1| = {err:.6e} | trG/(2NxNy) = {(np.reshape(G,(2*Nx*Ny, 2*Nx*Ny)).trace().real)/(2*Nx*Ny):.3f}" )
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
    

