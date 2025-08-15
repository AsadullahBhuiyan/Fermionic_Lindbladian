import numpy as np

class CI_Lindblad:
    def __init__(self, Nx, Ny, alpha, n_a=1/2, dt=1e-2, G_history = False,):
        """
        Build overcomplete Wannier spinors for a Chern insulator model.

        Produces four fields:
          self.W_A_plus, self.W_B_plus, self.W_A_minus, self.W_B_minus,
        each of shape (Nx, Ny, Nx, Ny, 2) with axes:
          (x, y, R_x, R_y, mu)
        and normalized per (R_x, R_y): sum_{x,y,mu} |W|^2 = 1.
        """
        self.Nx, self.Ny, self.alpha, self.n_a = Nx, Ny, alpha, n_a

        # --- k-grids in FFT's native ordering (radians per site) ---
        kx = 2*np.pi * np.fft.fftfreq(Nx, d=1.0)            # (Nx,)
        ky = 2*np.pi * np.fft.fftfreq(Ny, d=1.0)            # (Ny,)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')         # (Nx, Ny)

        # --- center grids ---
        Rx = np.arange(Nx)                                   # (Nx,)
        Ry = np.arange(Ny)                                   # (Ny,)

        # --- Hamiltonian vector n(k) ---
        nx = np.sin(KX)
        ny = np.sin(KY)
        nz = alpha - np.cos(KX) - np.cos(KY)
        n_mag = np.sqrt(nx**2 + ny**2 + nz**2)
        n_mag = np.where(n_mag == 0, 1e-15, n_mag)          # avoid divide-by-zero

        # prefactor
        norm = 0.5 * (1/np.sqrt(2.0)) * (1.0/n_mag)         # (Nx, Ny)

        # --- phase e^{ i (kx Rx + ky Ry) } using broadcasting (no 4D meshgrid) ---
        # shapes: (Nx, Ny, Nx, Ny)
        phase_x = np.exp(1j * KX[..., None, None] * Rx[None, None, :, None])
        phase_y = np.exp(1j * KY[..., None, None] * Ry[None, None, None, :])
        phase   = phase_x * phase_y

        # helper: 2D FFT over (kx, ky) axes only
        def k2_to_r2(Ak):  # Ak shape (Nx, Ny, Nx, Ny)
            return np.fft.fft2(Ak, axes=(0, 1))  # keep your fft2 convention

        # builder that stacks μ=1,2 and normalizes per (Rx,Ry)
        def build_and_normalize(comp0, comp1):
            # broadcast (Nx,Ny) → (Nx,Ny,Nx,Ny)
            F0 = phase * (norm * comp0)[..., None, None]
            F1 = phase * (norm * comp1)[..., None, None]
            # FFT over kx,ky → real space (x,y)
            W0 = k2_to_r2(F0)   # (Nx,Ny,Nx,Ny)
            W1 = k2_to_r2(F1)
            # stack components μ=1,2 on last axis
            W  = np.moveaxis(np.stack([W0, W1], axis=-1), -1, 2) # (Nx,Ny,2,Nx,Ny)

            # normalize per center (Rx,Ry) across x,y,μ
            denom = np.sqrt(np.sum(np.abs(W)**2, axis=(0, 1, 4), keepdims=True)) + 1e-15
            return W / denom

        # --- (ν, n) combinations --- (Nx,Ny,2,Nx,Ny)
        # ν = A, n = +
        self.W_A_plus  = build_and_normalize(
            n_mag + nz + nx + 1j*ny,
            n_mag - nz + nx - 1j*ny
        )
        # ν = B, n = +
        self.W_B_plus  = build_and_normalize(
            n_mag + nz - nx - 1j*ny,
            -n_mag + nz + nx - 1j*ny
        )
        # ν = A, n = −
        self.W_A_minus = build_and_normalize(
            n_mag - nz - nx - 1j*ny,
            n_mag - nz + nx + 1j*ny
        )
        # ν = B, n = −
        self.W_B_minus = build_and_normalize(
            n_mag - nz + nx + 1j*ny,
            -n_mag - nz - nx + 1j*ny
        )

        
        self.G = np.zeros(Nx, Nx, 2, Nx, Ny, 2)

        if G_history:
            self.G_list = []
            self.G_list.append(self.G)

    def make_V(self, W):
        V = np.einsum('ijklm, pqrlm -> ijkpqr', W, W.conj(), optimize=True) # sum over Wannier centers 
        return V
    
    def Lgain(self, G):
        '''
        G in (Nx, Ny, 2, Nx, Ny, 2)
        '''
        V_A_minus = self.make_V(self.W_A_minus)
        V_B_minus = self.make_V(self.W_B_minus)
        V = V_A_minus + V_B_minus #sum over local orbital choices
        Y = -(1/2)*(np.einsum('ijklmn, lmnpqr -> ijkpqr', G, V, optimize=True) + np.einsum('ijklmn, lmnpqr -> ijkpqr', V, G, optimize=True))   
        return self.n_a*(V + Y)
    
    def Lloss(self, G):
        '''
        G in (Nx, Ny, 2, Nx, Ny, 2)            
        '''
        V_A_plus = self.make_V(self.W_A_plus)
        V_B_plus = self.make_V(self.W_B_plus)
        V = V_A_plus + V_B_plus #sum over local orbital choices
        Y = -((1-self.n_a)/2)*(np.einsum('ijklmn, lmnpqr -> ijkpqr', G, V, optimize=True) + np.einsum('ijklmn, lmnpqr -> ijkpqr', V, G, optimize=True))   
        return Y
    
    def double_comm(self, G, W):
        '''
        G in (Nx, Ny, 2, Nx, Ny, 2)
        Summing over Wannier coordinates but not trial orbital choices
        '''
        linear_term = np.einsum('ijkab, lmnab, lmnpqr -> ijkpqr', W, W.conj(), G, optimize=True) + np.einsum('ijkab, lmnab, lmnpqr -> ijkpqr', G, W, W.conj(), optimize=True)
        nonlinear_term = np.einsum('mnxab,pqyab,ijwab,ijwzkl,klzab -> mnxpqy', W, W.conj(), W.conj(), G, W, optimize=True)
        Y = linear_term - 2*nonlinear_term
        return Y
    
    def Ldecoh(self, G):
        '''
        G in (Nx, Ny, 2, Nx, Ny, 2)            
        '''
        upperband_term = self.double_comm(G, self.W_A_plus) + self.double_comm(G, self.W_B_plus)
        lowerband_term = self.double_comm(G, self.W_A_minus) + self.double_comm(G, self.W_B_minus)
        
        Y = -(1/2)*((2-self.n_a)*lowerband_term + (1+self.n_a)*upperband_term)
        return Y
    
    def Lcycle(self, G):
        '''
        G in (Nx, Ny, 2, Nx, Ny, 2)            
        '''
        Y = self.Lgain(G) + self.Lloss(G) + self.Ldecoh(G)
        return Y
        
    def rk4_step_inplace(self, G, dt, tmp=None):
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
        k1[:] = self.Lcycle(G)
        # k2 = f(G + dt/2 * k1)
        np.multiply(k1, 0.5*dt, out=Y); np.add(G, Y, out=Y)
        k2[:] = self.Lcycle(Y)
        # k3 = f(G + dt/2 * k2)
        np.multiply(k2, 0.5*dt, out=Y); np.add(G, Y, out=Y)
        k3[:] = self.Lcycle(Y)
        # k4 = f(G + dt * k3)
        np.multiply(k3, dt, out=Y); np.add(G, Y, out=Y)
        k4[:] = self.Lcycle(Y)
        # G += dt/6 * (k1 + 2k2 + 2k3 + k4)
        # Use Y as accumulator: Y = k1 + 2k2 + 2k3 + k4
        np.add(k1, k4, out=Y)
        np.add(Y, 2.0*k2, out=Y)
        np.add(Y, 2.0*k3, out=Y)
        G += (dt/6.0) * Y
        return G, tmp
        

