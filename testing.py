import meshes
import ioFunctions
import igl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# import igl.pyigl
# import nibabel
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from mayavi import mlab
# import vtk
# import ioFunctions as io



subject="100610"
pial=meshes.Surface()
pial.getSurf(subject=subject, hemi="lh", surf="pial")
pial.getAparc(subject=subject, hemi="lh")



# # someGyrus = meshes.Submesh()
# # someGyrus.getSubmesh(pial, 24)
# #
# # dyad1=meshes.Volume()
# # #dyad1.getVolume("C:\\Users\\uhussain\\Documents\\ShareVM\\Cortex\\101006\\Diffusion\\Diffusion.bedpostX\\dyads1.nii.gz")
# # dyad1.getVolume("K:\\Datasets\\HCP_diffusion\\101006\\Diffusion\\Diffusion.bedpostX\\dyads1.nii.gz")
# # dyad1.makeInterpolator()
mgz=ioFunctions.loadMgz(subject=subject)
# #
fol=meshes.Foliation(10)
fol.getFoliation(subject=subject, hemi="lh")
# #folProj=meshes.FoliationProjection(fol,dyad1,mgz.header)
# #
# #
# # vertices=np.row_stack([vertex.coords for vertex in pial.vertices ])
# # faces=np.row_stack([face.vertex_ids for face in pial.faces ])
# # normalVectors=igl.per_vertex_normals(vertices,faces, weighting=1)
# #
# # dotProduct=[]
# # temp= [0 for j in range(len(fol.surfaces[0].normals.vectors))]
# # for i in range(10):
# #     for j in range(len(fol.surfaces[0].normals.vectors)):
# #         vec1=np.asarray(fol.surfaces[i].normals.vectors[j])
# #         vec2 =np.asarray(folProj.projection[i].vector[j])
# #         temp[j]=abs(vec1 @ vec2)
# #     dotProduct.append(temp)
# #
# #
dispersion=meshes.Volume()
dispersion.getVolume("K:\\Datasets\\HCP_diffusion\\100610\\Diffusion\\NODDI_7t\\NODDI_IC.kappa.nii.gz")
dispersion.makeInterpolator()
disProj=meshes.FoliationProjection(fol,dispersion,mgz.header)
# #
# # disp=[]
# # temp= [0 for j in range(len(fol.surfaces[0].normals.vectors))]
# # for i in range(10):
#     #for j in range(len(fol.surfaces[0].normals.vectors)):
#     #    temp[j]=disProj.projection[i].scalar[j]
#     #print(len(temp))
disp=np.row_stack([projection.scalar for projection in disProj.projection])
# gcurv=np.row_stack([surface.gaussian_curvature for surface in fol.surfaces])
# fcurv=np.row_stack([surface.freesurfer_curvature for surface in fol.surfaces])
curv1=np.row_stack([surface.gaussian_curvature[2] for surface in fol.surfaces])
curv2=np.row_stack([surface.gaussian_curvature[3] for surface in fol.surfaces])



for surf in range(10):
     plt.scatter(disp[surf], curv2[surf],s=50,alpha=0.08)
     plt.scatter(disp[surf], curv1[surf],s=50,alpha=0.08)
     plt.xlim([0,10])
     plt.ylim([-5, 5])
     plt.show()
#
#
# #
# #
surf=0
surface=surf
plot=meshes.Vision()
plot.processMesh(mesh=fol.surfaces[surface])
plot.addScalar(scalar=curv2[surface])
plot.show()
#
#
#
# #
# #
# plt.hist2d(disp[surf], curv[surf],bins=1000, range=[[0,2],[-0.1,0.1]])
# plt.show()
