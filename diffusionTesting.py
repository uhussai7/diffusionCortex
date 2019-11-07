import diffusion
import numpy as np

dvol=diffusion.diffVolume()
dvol.getVolume("K:\\Datasets\\HCP_diffusion\\101006\\Diffusion\\Diffusion")

dvol.shells()


for j in range(2):
    for i in range(3):
        test.append(D[i][j])