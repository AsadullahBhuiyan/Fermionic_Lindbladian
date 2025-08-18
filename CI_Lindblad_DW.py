import numpy as np
import math
import time
from IPython.display import clear_output
from matplotlib import pyplot as plt

class CI_Lindblad_DW:
    def __init__(self, Nx, Ny, decoh = True, alpha = 1.0):
        """
        1) Build overcomplete Wannier spinors for a Chern insulator model.

        2) Generate masks for each partition A,B,C for computation of real space chern number
        """
        self.Nx, self.Ny = Nx, Ny
        self.alpha = alpha
        self.decoh = decoh
        self.construct_OW_functions(self.alpha) # construct Wannier functions

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

    def construct_OW_functions(self, alpha, DW = True):
        '''
        Produces four fields:
          self.W_A_plus, self.W_B_plus, self.W_A_minus, self.W_B_minus,
        each of shape (Nx, Ny, Nx, Ny, 2) with axes:
          (x, y, R_x, R_y, mu)
        and normalized per (R_x, R_y): sum_{x,y,mu} |W|^2 = 1.
        '''

        if DW: 
            # Create topological domain wall
            alpha = np.empty((self.Nx, self.Ny), dtype = complex)
            alpha[0:self.Nx//2, :] = 1
            alpha[self.Nx//2:self.Nx+1, :] = 3
        else:
            alpha = np.ones((self.Nx, self.Ny))
            
            self.alpha = alpha

        # k-grids in radians per lattice spacing (FFT ordering)
        kx = 2*np.pi * np.fft.fftfreq(self.Nx, d=1.0)     # shape (Nx,)
        ky = 2*np.pi * np.fft.fftfreq(self.Ny, d=1.0)     # shape (Ny,)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')  # shape (Nx, Ny)

        # model vector n(k)
        self.nx = np.sin(KX)[:, :, None, None]
        self.ny = np.sin(KY)[:, :, None, None]
        self.nz = alpha[None, None, :, :] - np.cos(KX)[:, :, None, None] - np.cos(KY)[:, :, None, None]
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
        def make_W(Pband, tau, phase):
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
        self.W_A_plus  = make_W(self.Pplus,  tauA, phase)
        self.W_B_plus  = make_W(self.Pplus,  tauB, phase)
        self.W_A_minus = make_W(self.Pminus, tauA, phase)
        self.W_B_minus = make_W(self.Pminus, tauB, phase)

        # Build \sum_{R,nu} W_{R,\nu,n}W_{R,\nu,n}^*
        def make_V(W):
            V = np.einsum('ijklm, pqrlm -> ijkpqr', W, W.conj(), optimize=True) # sum over Wannier centers 
            return V
        
        self.V_A_minus = make_V(self.W_A_minus) 
        self.V_B_minus = make_V(self.W_B_minus)
        self.V_minus = self.V_A_minus + self.V_B_minus # sum over local orbital choices

        self.V_A_plus = make_V(self.W_A_plus)
        self.V_B_plus = make_V(self.W_B_plus)     
        self.V_plus =  self.V_A_plus + self.V_B_plus # sum over local orbital choices
    
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

        return G, G_history

    # -----------------------------
    # Chern number 
    # -----------------------------

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

    def squared_two_point_corr(self, G, rx=0, ry=0):
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
        G = np.asarray(G)
        Nx, Ny, s1, Nx2, Ny2, s2 = G.shape
        if (s1, s2) != (2, 2) or (Nx, Ny) != (Nx2, Ny2):
            raise ValueError("G must have shape (Nx, Ny, 2, Nx, Ny, 2).")

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
        
        if alpha != self.alpha:
            self.construct_OW_functions(alpha)

        chern_vals = np.empty(int(max_steps), dtype=complex)
        ts = np.arange(1, int(max_steps)+1, dtype=float) 
        G_history = [] if keep_history else []

        # Determine an effective dt if available from docstring/defaults
        dt = 5e-2  # use same default as G_evolution for timing label

        # Insert timer start
        t_start = time.time()
        for step in range(1, int(max_steps) + 1):
            t0 = time.time()
            G, _ = self.rk4_Lindblad_evolver(G, dt)
            ch = self.real_space_chern_number(G)
            chern_vals[step-1] = ch
            if keep_history:
                G_history.append(G.copy())
            iter_time = time.time() - t0
            clear_output(wait=True)
            print(f"[chern_dyn] iter {step}/{int(max_steps)} | dt={dt:.3e} | itertime={iter_time:.3e}s | total={time.time()-t_start:.3e}s | alpha={self.alpha:g} | N=({self.Nx},{self.Ny}) | Chern={ch.real:.6f}{' + ' + str(ch.imag) + 'i' if abs(ch.imag) > 0 else ''}")

        return ts, chern_vals, G, G_history

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
            self.construct_OW_functions(alpha = a)   # updates self.alpha
            G, _ = self.G_evolution()
            chern_list[i] = self.real_space_chern_number(G)

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

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for N in N_list:
            solver = CI_Lindblad(Nx=N, Ny=N, decoh=self.decoh, alpha=alpha_val)
            # Use the built-in chern-dynamics simulator (handles printing/progress)
            ts, ch, _, _ = solver.chern_dynamics_vs_time(alpha=alpha_val,
                                                         max_steps=int(max_steps),
                                                         keep_history=False)
            # ch is complex; plot real part. Scale ts by the requested dt for x-axis units
            ax.plot(ts * dt, ch.real, label=f"N={N}")

        ax.set_xlabel("t")
        ax.set_ylabel("real-space Chern number")
        ax.set_title(f"Chern dynamics vs t (alpha={alpha_val:g})")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()

        if filename is None:
            Ns = "-".join(str(n) for n in N_list)
            filename = f"chern_dynamics_vs_time_alpha{alpha_val:g}_Ns{Ns}.pdf"
        fig.savefig(filename, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        return filename

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

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for N in N_list:
            ch_vals = []
            for a in alphas:
                solver = CI_Lindblad(Nx=N, Ny=N, decoh=self.decoh, alpha=a)
                G_final, _ = solver.G_evolution(max_steps=int(max_steps), dt=dt, keep_history=False)
                ch_vals.append(solver.real_space_chern_number(G_final).real)
            ax.plot(alphas, np.array(ch_vals), label=f"N={N}")

        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("real-space Chern number")
        ax.set_title(f"Steady-state Chern vs $\\alpha$ (steps={int(max_steps)})")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()

        if filename is None:
            Ns = "-".join(str(n) for n in N_list)
            filename = f"chern_steady_vs_alpha_steps{int(max_steps)}_Ns{Ns}.pdf"
        fig.savefig(filename, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        return filename

    def plot_squared_corr_vs_alpha(self, rx=0, ry=0, alpha_list=None,
                                   dt=5e-2, max_steps=500, filename=None):
        """
        Plot steady-state squared two-point correlator C_G(rx, ry) vs alpha for
        the *current* lattice size (self.Nx,self.Ny).

        Parameters
        ----------
        rx, ry : int
            Separation components.
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

        Nx, Ny = int(self.Nx), int(self.Ny)
        rx_arr = np.atleast_1d(rx)
        ry_arr = np.atleast_1d(ry)
        averaged = (rx_arr.size > 1) or (ry_arr.size > 1)

        fig, ax = plt.subplots(figsize=(7, 4.5))

        Cvals = []
        for a in alphas:
            solver = CI_Lindblad(Nx=Nx, Ny=Ny, decoh=self.decoh, alpha=a)
            G_final, _ = solver.G_evolution(max_steps=int(max_steps), dt=dt, keep_history=False)
            C = solver.squared_two_point_corr(G_final, rx=rx, ry=ry)
            C = np.asarray(C, dtype=float)
            # Reduce to a scalar per alpha if a grid was requested
            Cvals.append(C.mean() if C.ndim > 0 else float(C))

        Cvals = np.asarray(Cvals, dtype=float)
        ax.plot(alphas, Cvals, marker='o', ms=3, lw=1)
        ax.set_xlabel(r"$\alpha$")
        if averaged:
            ax.set_ylabel(fr"$\langle C_G \rangle$ over rx,ry grid")
        else:
            ax.set_ylabel(fr"$C_G(r_x={rx}, r_y={ry})$")
        if averaged:
            ax.set_title(f"Steady-state |G|^2 vs $\\alpha$ (avg over rx,ry grid)")
        else:
            ax.set_title(f"Steady-state |G|^2 vs $\\alpha$ at separation (rx,ry)=({rx},{ry})")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if filename is None:
            filename = f"corr2_vs_alpha_N{Nx}_rx{rx}_ry{ry}_steps{int(max_steps)}.pdf"
        fig.savefig(filename, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        return filename