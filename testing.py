import meshes
import ioFunctions
import igl
import numpy as np
import matplotlib.pyplot as plt
# import igl.pyigl
# import nibabel
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from mayavi import mlab
# import vtk
# import ioFunctions as io


pial=meshes.Surface()

pial.getSurf(subject="101006", hemi="lh", surf="pial")
pial.getAparc(subject="101006", hemi="lh")



someGyrus = meshes.Submesh()
someGyrus.getSubmesh(pial, 24)

dyad1=meshes.Volume()
#dyad1.getVolume("C:\\Users\\uhussain\\Documents\\ShareVM\\Cortex\\101006\\Diffusion\\Diffusion.bedpostX\\dyads1.nii.gz")
dyad1.getVolume("K:\\Datasets\\HCP_diffusion\\101006\\Diffusion\\Diffusion.bedpostX\\dyads1.nii.gz")
dyad1.makeInterpolator()
mgz=ioFunctions.loadMgz(subject="101006")

fol=meshes.Foliation(10)
fol.getFoliation(subject="101006", hemi="lh")
folProj=meshes.FoliationProjection(fol,dyad1,mgz.header)


vertices=np.row_stack([vertex.coords for vertex in pial.vertices ])
faces=np.row_stack([face.vertex_ids for face in pial.faces ])
normalVectors=igl.per_vertex_normals(vertices,faces, weighting=1)

dotProduct=[]
temp= [0 for j in range(len(fol.surfaces[0].normals.vectors))]
for i in range(10):
    for j in range(len(fol.surfaces[0].normals.vectors)):
        vec1=np.asarray(fol.surfaces[i].normals.vectors[j])
        vec2 =np.asarray(folProj.projection[i].vector[j])
        temp[j]=abs(vec1 @ vec2)
    dotProduct.append(temp)


dispersion=meshes.Volume()
dispersion.getVolume("K:\\Datasets\\HCP_diffusion\\101006\\Diffusion\\NODDI\\NODDI_EC.kappa.nii.gz")
dispersion.makeInterpolator()
disProj=meshes.FoliationProjection(fol,dispersion,mgz.header)

disp=[]
temp= [0 for j in range(len(fol.surfaces[0].normals.vectors))]
for i in range(10):
    #for j in range(len(fol.surfaces[0].normals.vectors)):
    #    temp[j]=disProj.projection[i].scalar[j]
    #print(len(temp))
disp=np.row_stack([projection.scalar for projection in disProj.projection])



surface=4
plot=meshes.Vision()
plot.processMesh(mesh=fol.surfaces[surface])
plot.addScalar(scalar=disp[surface])
plot.show()


n, bins, patches = plt.hist(disp[9], 500, density=True, facecolor='g', alpha=0.75)
plt.show()
