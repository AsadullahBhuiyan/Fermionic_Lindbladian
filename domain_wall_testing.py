#!/usr/bin/env python
# coding: utf-8

# In[17]:


import importlib
import numpy as np
import matplotlib.pyplot as plt

import CI_Lindblad_DW           # import the module (file: CI_Lindblad_DW.py)
importlib.reload(CI_Lindblad_DW)  # hot‑reload if you’re editing the class
from CI_Lindblad_DW import CI_Lindblad_DW

plt.rcParams["figure.dpi"] = 140


# In[20]:


# Domain‑wall CI, decoherence ON: Chern Marker and Correlations functions
N = 31
dt = 5e-2
max_steps = 250
ry_max = N/2

solver = CI_Lindblad_DW(Nx=N, Ny=N, max_steps_init = max_steps, decoh=True)

gif_path, final_path, C_last, G = solver.chern_marker_dynamics(G_init=None, fps=12, cmap='RdBu_r', outbasename=None, vmin=-1.0, vmax=1.0)
solver.plot_corr_y_profiles(ry_max=ry_max, filename=f"corr2_y_profiles_N={N}_decoh_on.pdf")


# In[ ]:


# Domain‑wall CI, decoherence OFF: Chern Marker and Correlations functions
N = 31
dt = 5e-2
max_steps = 250
ry_max = N/2

solver = CI_Lindblad_DW(Nx=N, Ny=N, max_steps_init = max_steps, decoh=False)

gif_path, final_path, C_last, G = solver.chern_marker_dynamics(G_init=None, fps=12, cmap='RdBu_r', outbasename=None, vmin=-1.0, vmax=1.0)
solver.plot_corr_y_profiles(ry_max=ry_max, filename=f"corr2_y_profiles_N={N}_decoh_on.pdf")


# In[ ]:




