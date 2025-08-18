import numpy as np
import math
import time
from IPython.display import clear_output
from matplotlib import pyplot as plt

class CI_Lindblad_DW:
    def __init__(self, Nx, Ny, decoh = True):
        """
        1) Build overcomplete Wannier spinors for a Chern insulator model.

        2) Generate masks for each partition A,B,C for computation of real space chern number
        """
        self.Nx, self.Ny = Nx, Ny
        self.decoh = decoh
        self.alpha = None

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

    def construct_OW_functions(self, alpha, nshell = None, DW = True):
        '''
        Produces four fields:
          self.W_A_plus, self.W_B_plus, self.W_A_minus, self.W_B_minus,
        each of shape (Nx, Ny, Nx, Ny, 2) with axes:
          (x, y, R_x, R_y, mu)
        and normalized per (R_x, R_y): sum_{x,y,mu} |W|^2 = 1.
        '''

        if DW: 
            # Create topological domain wall
            # Fill with the 'outside' value, then override the middle slab.
            alpha = np.full((self.Nx, self.Ny), 3, dtype=complex)   # Chern = 0 region
            alpha[self.Nx//4 : 3*self.Nx//4, :] = 1                 # Chern = 1 region
        else:
            alpha = np.ones((self.Nx, self.Ny))
            self.alpha = alpha

        # k-grids in radians per lattice spacing (FFT ordering)
        kx = 2*np.pi * np.fft.fftfreq(self.Nx, d=1.0)     # shape (Nx,)
        ky = 2*np.pi * np.fft.fftfreq(self.Ny, d=1.0)     # shape (Ny,)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')  # shape (Nx, Ny)

        # model vector n(k)
        self.nx = np.sin(KX)[:, :, None, None] # (Nx, Ny, Rx, Ry)
        self.ny = np.sin(KY)[:, :, None, None] # (Nx, Ny, Rx, Ry)
        self.nz = alpha[None, None, :, :] - np.cos(KX)[:, :, None, None] - np.cos(KY)[:, :, None, None] # (Nx, Ny, Rx, Ry)
        self.nmag = np.sqrt(self.nx**2 + self.ny**2 + self.nz**2) # (Nx, Ny, Rx, Ry)
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
              self.nz[..., None, None] * pauli_z) / self.nmag[..., None, None]  # (Nx, Ny, Rx, Ry, 2, 2)

        # construct upper and lower band projectors
        self.Pminus = 0.5 * (Id - hk)   # (Nx, Ny, Rx, Ry, 2, 2)
        self.Pplus  = 0.5 * (Id + hk)   # (Nx, Ny, Rx, Ry, 2, 2)

        # local choice of orbitals
        tauA = (1/np.sqrt(2)) * np.array([[1], [1]], dtype=complex)   # (2,1)
        tauB = (1/np.sqrt(2)) * np.array([[1], [-1]], dtype=complex)  # (2,1)

        # --- phases for all centers (R_x, R_y) ---
        Rx_grid = np.arange(self.Nx)                                   # (Rx,)
        Ry_grid = np.arange(self.Ny)                                   # (Ry,)
        phase_x = np.exp(1j * KX[..., None, None] * Rx_grid[None, None, :, None])  # (Nx,Ny,Rx,1)
        phase_y = np.exp(1j * KY[..., None, None] * Ry_grid[None, None, None, :])  # (Nx,Ny,1,Ry)
        phase   = phase_x * phase_y                                # (Nx,Ny,Rx,Ry)

        # Helper: do FFT over k-axes only (0,1), for every (R_x,R_y)
        def k2_to_r2(Ak):  # Ak shape (Nx, Ny, Rx, Ry)
            return np.fft.fft2(Ak, axes=(0, 1))

        # ------------------------------
        # helper to build a normalized W for every center
        # row spinor in k-space: psi_k = tau^† P(k)
        # ------------------------------
        def make_W(Pband, tau, phase):
            # tau^\dagger P(k): (2,)^* with (...,2,2) over left index -> (...,2)
            tau_dag = tau[:, 0].conj()                              # (2,)
            psi_k   = np.einsum('m,...mn->...n', tau_dag, Pband)    # (Nx,Ny,Rx,Ry,2)

            # Broadcast psi_k components over centers, then FFT over k-axes
            F0 = phase * psi_k[..., 0]  # (Nx,Ny,Rx,Ry)
            F1 = phase * psi_k[..., 1]  # (Nx,Ny,Rx,Ry)
            W0 = k2_to_r2(F0)           # (Nx,Ny,Rx,Ry)
            W1 = k2_to_r2(F1)           # (Nx,Ny,Rx,Ry)

            # Stack μ=0,1 as the 3rd axis → (Nx,Ny,2,Nx,Ny)
            W  = np.moveaxis(np.stack([W0, W1], axis=-1), -1, 2) # (Nx,Ny,2,Rx,Ry)

            # Normalize per center (R_x,R_y) across (x,y,μ)
            denom = np.sqrt(np.sum(np.abs(W)**2, axis=(0, 1, 2), keepdims=True)) + 1e-15
            return W / denom

        # Build the four overcomplete Wannier spinors for ALL centers: (Nx,Ny,2,Rx,Ry)
        self.W_A_plus  = make_W(self.Pplus,  tauA, phase)
        self.W_B_plus  = make_W(self.Pplus,  tauB, phase)
        self.W_A_minus = make_W(self.Pminus, tauA, phase)
        self.W_B_minus = make_W(self.Pminus, tauB, phase)

        # Build \sum_{R} W_{R,\nu,n}W_{R,\nu,n}^*
        def make_V(W):
            V = np.einsum('ijklm, pqrlm -> ijkpqr', W, W.conj(), optimize=True) # sum over Wannier centers 
            return V
        
        self.V_A_minus = make_V(self.W_A_minus) 
        self.V_B_minus = make_V(self.W_B_minus)
        self.V_minus = self.V_A_minus + self.V_B_minus # sum over \nu

        self.V_A_plus = make_V(self.W_A_plus)
        self.V_B_plus = make_V(self.W_B_plus)     
        self.V_plus =  self.V_A_plus + self.V_B_plus # sum over \nu
    
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
            G_final, _ = solver.G_evolution(max_steps=int(max_steps), dt=dt, keep_history=False)
            C_grid = solver.squared_two_point_corr(G_final, rx=rx_arr, ry=ry_arr)

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
    
    def local_chern_marker(self, G):
        """
        Local Chern marker C(r) (Bianco–Resta) for a pure Gaussian state G.

        C(r) = 2π i ∑_μ [ G X G Y G − G Y G X G ]_{(r,μ),(r,μ)}

        Parameters
        ----------
        G : ndarray, shape (Nx, Ny, 2, Nx, Ny, 2)

        Returns
        -------
        tanh(C) : ndarray, shape (Nx, Ny)
            Tanh ofLocal Chern marker on the torus.
        """
        G = np.asarray(G)
        Nx, Ny, s1, Nx2, Ny2, s2 = G.shape
        if (s1, s2) != (2, 2) or (Nx, Ny) != (Nx2, Ny2):
            raise ValueError("G must have shape (Nx, Ny, 2, Nx, Ny, 2).")

        # Coordinate fields (float for safe products)
        x = np.arange(Nx, dtype=float)
        y = np.arange(Ny, dtype=float)
        Xgrid, Ygrid = np.meshgrid(x, y, indexing='ij')   # (Nx, Ny)

        # Helpers: scale by X or Y on the left/right (no full matrix build)
        def left_X(A):   # (X ∘) : multiply using the left real-space index
            return Xgrid[:, :, None, None, None] * A

        def right_X(A):  # (∘ X) : multiply using the right real-space index
            return A * Xgrid[None, None, None, :, :, None]

        def left_Y(A):
            return Ygrid[:, :, None, None, None] * A

        def right_Y(A):
            return A * Ygrid[None, None, None, :, :, None]

        # Generic contraction: (A @ B) over the shared (x',y',s') = (l,m,n)
        # A: (i,j,s, l,m,n) ; B: (l,m,n, o,p,r)  → (i,j,s, o,p,r)
        def mm(A, B):
            return np.einsum('ijslmn,lmnopr->ijsopr', A, B, optimize=True)

        # --- Build the two ordered products ---
        # T1 = G X G Y G
        T = right_X(G)     # G X
        T = mm(T, G)       # G X G
        T = right_Y(T)     # (G X G) Y
        T = mm(T, G)       # G X G Y G

        # T2 = G Y G X G
        U = right_Y(G)     # G Y
        U = mm(U, G)       # G Y G
        U = right_X(U)     # (G Y G) X
        U = mm(U, G)       # G Y G X G

        # M = 2π i (T1 − T2)
        M = (2.0 * np.pi * 1j) * (T - U)  # (Nx,Ny,2, Nx,Ny,2)

        # Extract diagonal element at equal position & spin, then sum over spin μ
        ix = np.arange(Nx)[:, None, None]
        iy = np.arange(Ny)[None, :, None]
        ispin = np.arange(2)[None, None, :]

        # Shape (Nx, Ny, 2): element (x,y,μ; x,y,μ)
        diag_vals = M[ix, iy, ispin, ix, iy, ispin]
        C = diag_vals.sum(axis=2)  # sum over μ → (Nx, Ny)

        return np.tanh(np.real_if_close(C, tol=1e-9))