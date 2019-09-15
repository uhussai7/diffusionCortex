import ioFunctions
import numpy as np



class annot:
    def __init__(self):
        self.labels = []
        self.ctab = []
        self.names= []

class gyrus:
    def __init__(self):
        self.coords = []
        self.faces = []

    def getGyrus(self, surface, gyrusIndex): #filter out a gyrus based on
        if surface.aparc is None:
            raise ValueError("Aparc not loaded")
        inds = np.where(surface.aparc.labels == gyrusIndex)
        self.coords = surface.coords[inds]
        for face in surface.faces:
            if (bool(set(face) & set(inds[0][:]))):
                self.faces.append(face)
        self.coords=np.asarray(self.coords)
        self.faces=np.asarray(self.faces)

class surface:
    from meshes import annot
    def __init__(self):
        self.coords = []
        self.faces = []
        self.aparc = annot() #this will be for lh/rh.aparc.annot

    def getSurf(self,subject=None, hemi=None, surf=None,**kwargs):
        self.coords, self.faces = ioFunctions.loadSurf(subject,hemi,surf)

    def getAparc(self, subject=None, hemi=None, **kwargs):
        self.aparc.labels, self.aparc.ctab, self.aparc.names = ioFunctions.loadAparc(subject,hemi)