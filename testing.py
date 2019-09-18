import meshes
import numpy as np
import igl.pyigl
import nibabel
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mayavi import mlab
import vtk

pial=meshes.Surface()

pial.getSurf(subject="101006", hemi="lh", surf="pial")
pial.getAparc(subject="101006", hemi="lh")

someGyrus = meshes.Gyrus()
someGyrus.getGyrus(pial, 24)



x=[]
y=[]
z=[]
for vertex in someGyrus.vertices:
    x.append(vertex.coords[0])
    y.append(vertex.coords[1])
    z.append(vertex.coords[2])


triangles = np.row_stack([face.vertex_ids for face in someGyrus.faces])
vertices = np.row_stack([vertex.coords for vertex in someGyrus.vertices])


# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_trisurf(x, y, z, triangles=triangles, linewidth=0.2, antialiased=True)
# plt.show()

#nibabel.freesurfer.io.write_geometry('test',someGyrus.coords,someGyrus.faces)


[PD1, PD2, PV1, PV2]=igl.principal_curvature(vertices,triangles,radius=5)



mlab.quiver3d(x,y,z,PD1[:,0],PD1[:,1],PD1[:,2],scalars=PV1)
mlab.quiver3d(x,y,z,PD2[:,0],PD2[:,1],PD2[:,2],scalars=PV2)
mlab.triangular_mesh(x,y,z,triangles,scalars=PV1+PV2)
mlab.show()

