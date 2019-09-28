import meshes
import numpy as np
import igl.pyigl
import nibabel
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mayavi import mlab
import vtk
import ioFunctions

pial=meshes.Surface()

pial.getSurf(subject="101006", hemi="lh", surf="equi0.555555555556.pial")
pial.getAparc(subject="101006", hemi="lh")

someGyrus = meshes.Submesh()
someGyrus.getSubmesh(pial, 24)



x=[]
y=[]
z=[]
for vertex in pial.vertices:
    x.append(vertex.coords[0])
    y.append(vertex.coords[1])
    z.append(vertex.coords[2])


triangles = np.row_stack([face.vertex_ids for face in pial.faces])
vertices = np.row_stack([vertex.coords for vertex in pial.vertices])


# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_trisurf(x, y, z, triangles=triangles, linewidth=0.2, antialiased=True)
# plt.show()

#nibabel.freesurfer.io.write_geometry('test',someGyrus.coords,someGyrus.faces)


#[PD1, PD2, PV1, PV2]=igl.principal_curvature(vertices,triangles,radius=5)



dyad1=ioFunctions.loadDyad(subject="101006", dyad="1")
mgz=ioFunctions.loadMgz(subject="101006")
projection=meshes.Projection()
projection.project(vol=dyad1, mesh=pial, header=mgz.header)
dyad1_vectors = np.row_stack([vector for vector in projection.vector])


mlab.quiver3d(x,y,z,dyad1_vectors[:,0],dyad1_vectors[:,1],dyad1_vectors[:,2])
#mlab.quiver3d(x,y,z,PD2[:,0],PD2[:,1],PD2[:,2],scalars=PV2)
#mlab.triangular_mesh(x,y,z,triangles,scalars=PV1+PV2)



mlab.show()



