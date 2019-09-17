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
someGyrus.getGyrus(pial, 25)



x=[]
y=[]
z=[]
for vertex in someGyrus.vertices:
    x.append(vertex.coords[0])
    y.append(vertex.coords[1])
    z.append(vertex.coords[2])


triangles=np.row_stack([face.vertex_ids for face in someGyrus.faces])


mlab.triangular_mesh(x,y,z,triangles)
mlab.show()


# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_trisurf(x, y, z, triangles=triangles, linewidth=0.2, antialiased=True)
# plt.show()

#nibabel.freesurfer.io.write_geometry('test',someGyrus.coords,someGyrus.faces)


#igl.pyigl.principal_curvature(pial.coords[1:50], pial.faces[1:50],2)


