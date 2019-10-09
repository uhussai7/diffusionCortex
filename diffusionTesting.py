import diffusion
import numpy as np

dvol=diffusion.diffVolume()
dvol.getVolume("K:\\Datasets\\HCP_diffusion\\101006\\Diffusion\\Diffusion")

inds=np.where(np.logical_and(dvol.gtab.bvals >= 800, dvol.gtab.bvals <=1200 ))