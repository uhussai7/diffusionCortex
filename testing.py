import meshes
import numpy as np
import igl.pyigl
import nibabel

pial=meshes.Surface()

pial.getSurf(subject="101006", hemi="lh", surf="pial")
pial.getAparc(subject="101006", hemi="lh")

someGyrus = meshes.Gyrus()
someGyrus.getGyrus(pial, 1)

#nibabel.freesurfer.io.write_geometry('test',someGyrus.coords,someGyrus.faces)


#igl.pyigl.principal_curvature(pial.coords[1:50], pial.faces[1:50],2)


