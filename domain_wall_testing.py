#!/usr/bin/env python
# coding: utf-8

# In[3]:


import importlib
import numpy as np
import matplotlib.pyplot as plt

import CI_Lindblad_DW           # import the module (file: CI_Lindblad_DW.py)
importlib.reload(CI_Lindblad_DW)  # hot‑reload if you’re editing the class
from CI_Lindblad_DW import CI_Lindblad_DW

plt.rcParams["figure.dpi"] = 140


# In[5]:


# Domain‑wall CI, decoherence ON: animate Chern marker and save final frame
N = 31

solver = CI_Lindblad_DW(Nx=N, Ny=N, decoh=True)

gif_path, final_path, C_last, G = solver.chern_marker_dynamics(dt=5e-2, max_steps=250, n_a=0.5,
                              G_init=None, fps=12, cmap='RdBu_r',
                              outbasename=None, vmin=-1.0, vmax=1.0)
print("Saved:", gif_path, "and", final_path)


# In[ ]:


# Domain‑wall CI, decoherence OFF
N = 31

solver = CI_Lindblad_DW(Nx=N, Ny=N, decoh=False)

gif_path, final_path, C_last, G = solver.chern_marker_dynamics(dt=5e-2, max_steps=250, n_a=0.5,
                              G_init=None, fps=12, cmap='RdBu_r',
                              outbasename=None, vmin=-1.0, vmax=1.0)
print("Saved:", gif_path, "and", final_path)


# In[ ]:


# Correlation profiles along y for several fixed x columns (probes both bulks + both walls)
N = 31
dt = 5e-2
max_steps = 250
ry_max = N

solver = CI_Lindblad_DW(Nx=N, Ny=N, decoh=True)

# (Optional) choose your own x columns (x0, label). If omitted, method picks sensible ones.
# half = Nx // 2; w = int(np.floor(0.2 * Nx))
# x0 = max(0, half - w); x1 = min(Nx, half + w + 1)
# x_positions = [( (x0+x1)//2, "topo bulk"), (x0//2, "trivial L"), ((x1+Nx)//2, "trivial R"),
#                ((x0-1)%Nx, "wall L"), (x1%Nx, "wall R")]

pdf = solver.plot_corr_y_profiles(
    dt=dt,
    max_steps=max_steps,
    ry_max=ry_max,
    # x_positions=x_positions,
    filename=f"corr2_y_profiles_N={N}_decoh_on.pdf",
)
print("Saved:", pdf)


# In[ ]:


# Correlation profiles along y for several fixed x columns (probes both bulks + both walls)
N = 31
dt = 5e-2
max_steps = 250
ry_max = N

solver = CI_Lindblad_DW(Nx=N, Ny=N, decoh=False)

# (Optional) choose your own x columns (x0, label). If omitted, method picks sensible ones.
# half = Nx // 2; w = int(np.floor(0.2 * Nx))
# x0 = max(0, half - w); x1 = min(Nx, half + w + 1)
# x_positions = [( (x0+x1)//2, "topo bulk"), (x0//2, "trivial L"), ((x1+Nx)//2, "trivial R"),
#                ((x0-1)%Nx, "wall L"), (x1%Nx, "wall R")]

pdf = solver.plot_corr_y_profiles(
    dt=dt,
    max_steps=max_steps,
    ry_max=ry_max,
    # x_positions=x_positions,
    filename=f"corr2_y_profiles_N={N}_decoh_off.pdf",
)
print("Saved:", pdf)


# In[ ]:




