import ioFunctions
import numpy as np

class Face:
    def __init__(self,vertex_ids=None):
        self.vertex_ids = vertex_ids

class Vertex:
    def __init__(self,coords=None):
        self.coords = coords
        self.faces = []


class Annot:
    def __init__(self):
        self.labels = []
        self.ctab = []
        self.names= []

class Gyrus:
    def __init__(self):
        self.vertices = []
        self.faces = []

    def getGyrus(self, surface, gyrusIndex): #filter out a gyrus based on
        if surface.aparc is None:
            raise ValueError("Aparc not loaded")
        inds = np.where(surface.aparc.labels == gyrusIndex)
        inds = inds[0][:]
        self.vertices = [Vertex() for ind in inds]
        f=0
        for i in range(inds.shape[0]):
            ind=inds[i]
            self.vertices[i].coords = surface.vertices[ind].coords
            faceids= surface.vertices[ind].faces
            for faceid in faceids:
                vert_ids_in_face = surface.faces[faceid].vertex_ids
                if (set(vert_ids_in_face) & set(inds)) == set(vert_ids_in_face):
                    ids_in_face = np.array([0, 0, 0])
                    for k in range(3):
                        ind_in_face=np.where(inds == vert_ids_in_face[k])
                        ind_in_face=ind_in_face[0][0]
                        self.vertices[ind_in_face].faces.append(f)
                        if vert_ids_in_face[k] != ind:
                            surface.vertices[vert_ids_in_face[k]].faces.remove(faceid)
                        ids_in_face[k]=ind_in_face
                    f=f+1
                    self.faces.append(Face(vertex_ids=ids_in_face))


class Surface:
    def __init__(self):
        self.vertices= []
        self.faces = []
        self.aparc = Annot() #this will be for lh/rh.aparc.annot

    def getSurf(self,subject=None, hemi=None, surf=None,**kwargs):
        coords, faces = ioFunctions.loadSurf(subject,hemi,surf)
        self.vertices=[Vertex(acoord) for acoord in coords]
        f=0 #there is probably a better way to do this
        for aface in faces:
            self.faces.append(Face(vertex_ids=aface))
            for node in range(3):
                self.vertices[aface[node]].faces.append(f)
            f=f+1
        del coords
        del faces

    def getAparc(self, subject=None, hemi=None, **kwargs):
        self.aparc.labels, self.aparc.ctab, self.aparc.names = ioFunctions.loadAparc(subject,hemi)

