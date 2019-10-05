import meshes
import ioFunctions
import igl
import numpy as np
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
dyad1.getVolume("C:\\Users\\uhussain\\Documents\\ShareVM\\Cortex\\101006\\Diffusion\\Diffusion.bedpostX\\dyads1.nii.gz")
# #dyad1.getVolume("K:\\Datasets\\HCP_diffusion\\101006\\Diffusion\\Diffusion.bedpostX\\dyads1.nii.gz")
dyad1.makeInterpolator()
mgz=ioFunctions.loadMgz(subject="101006")
# dyad1_proj=meshes.Projection()
# dyad1_proj.project(mesh=someGyrus,volume=dyad1, header=mgz.header)


# somePlot=meshes.Vision()
# somePlot.processMesh(mesh=someGyrus)
# somePlot.addVector(dyad1_proj.vector)
# somePlot.show()
# #
#reload(meshes)
fol=meshes.Foliation(10)
fol.getFoliation(subject="101006", hemi="lh")
folProj=meshes.FoliationProjection(fol,dyad1,mgz.header)


somePlot=meshes.Vision()
somePlot.processMesh(mesh=fol.surfaces[5])
somePlot.addVector(folProj.projection[5].vector)
somePlot.show()

# x=[]
# y=[]
# z=[]
# for vertex in pial.vertices:
#     x.append(vertex.coords[0])
#     y.append(vertex.coords[1])
#     z.append(vertex.coords[2])
#
#
# triangles = np.row_stack([face.vertex_ids for face in pial.faces])
# vertices = np.row_stack([vertex.coords for vertex in pial.vertices])
#
#
# # fig = plt.figure()
# # ax = fig.gca(projection='3d')
# # ax.plot_trisurf(x, y, z, triangles=triangles, linewidth=0.2, antialiased=True)
# # plt.show()
#
# #nibabel.freesurfer.io.write_geometry('test',someGyrus.coords,someGyrus.faces)
#
#
# #[PD1, PD2, PV1, PV2]=igl.principal_curvature(vertices,triangles,radius=5)
#
#
#
# dyad1=ioFunctions.loadDyad(subject="101006", dyad="1")
# mgz=ioFunctions.loadMgz(subject="101006")
# projection=meshes.Projection()
# projection.project(vol=dyad1, mesh=pial, header=mgz.header)
# dyad1_vectors = np.row_stack([vector for vector in projection.vector])
#
#
# mlab.quiver3d(x,y,z,dyad1_vectors[:,0],dyad1_vectors[:,1],dyad1_vectors[:,2])
# #mlab.quiver3d(x,y,z,PD2[:,0],PD2[:,1],PD2[:,2],scalars=PV2)
# #mlab.triangular_mesh(x,y,z,triangles,scalars=PV1+PV2)
#
#
#
# mlab.show()

normalVectors= np.row_stack([thisVector for thisVector in vector])

vertices=np.row_stack([vertex.coords for vertex in pial.vertices ])
faces=np.row_stack([face.vertex_ids for face in pial.faces ])
normalVectors=igl.per_vertex_normals(vertices,faces, weighting=1)

somePlot=meshes.Vision()
somePlot.processMesh(mesh=pial)
somePlot.addVector(normalVectors)
somePlot.show()




