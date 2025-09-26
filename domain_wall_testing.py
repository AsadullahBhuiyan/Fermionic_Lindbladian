#!/usr/bin/env python
# coding: utf-8

# In[1]:


import importlib
import numpy as np
import matplotlib.pyplot as plt

import CI_Lindblad_DW           # import the module (file: CI_Lindblad_DW.py)
importlib.reload(CI_Lindblad_DW)  # hot‑reload if you’re editing the class
from CI_Lindblad_DW import CI_Lindblad_DW

plt.rcParams["figure.dpi"] = 140


# In[ ]:


# Domain‑wall CI, decoherence ON: Chern Marker and Correlations functions
#N = 31
#dt = 5e-2
#max_steps = 250
#ry_max = N/2
#
#solver = CI_Lindblad_DW(Nx=N, Ny=N, max_steps_init = max_steps, decoh=True)
#
#gif_path, final_path, C_last, G = solver.chern_marker_dynamics(G_init=None, fps=12, cmap='RdBu_r', outbasename=None, vmin=-1.0, vmax=1.0)
#solver.plot_corr_y_profiles(ry_max=ry_max, x_positions = np.arange(16, 27), filename=f"corr2_y_profiles_N={N}_decoh_on_xpos=16_27.pdf")


# In[ ]:


#G_last = solver.G_last
#print(np.shape(G_last))
#spec, _ = np.linalg.eigh(G_last.reshape(2*N**2, 2*N**2))
#plt.plot(spec,'o')
#plt.show()


# In[ ]:


# Domain‑wall CI, decoherence OFF: Chern Marker and Correlations functions
#N = 31
#dt = 5e-2
#max_steps = 500
#ry_max = N/2
#
#solver = CI_Lindblad_DW(Nx=N, Ny=N, max_steps_init = max_steps, decoh=False, DW=True)
#
#gif_path, final_path, C_last, G = solver.chern_marker_dynamics(G_init=None, fps=12, cmap='RdBu_r', outbasename=None, vmin=-1.0, vmax=1.0)
#solver.plot_corr_y_profiles(ry_max=ry_max, x_positions = [4, 8, 9, 10, 12, 14, 16, 18, 20, 21, 22, 24, 26], filename=f"corr2_y_profiles_N={N}_decoh_off.pdf")


# In[ ]:


#G_last = solver.G_last
#spec, _ = np.linalg.eigh(G_last.reshape(2*N**2, 2*N**2))
#plt.plot(spec, 'o')


# In[2]:


solver = CI_Lindblad_DW(Nx=21, Ny=21, init = False, decoh=False)
solver.plot_spectrum_vs_time(times=(0, 2, 10, 20, 40), dt=0.05)


# In[ ]:




