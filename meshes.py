import ioFunctions

class surface:
    def __init__(self):
        self.coords = []
        self.faces = []

    def getSurf(self,subject=None, hemi=None, surf=None,**kwargs):
        self.coords, self.faces = ioFunctions.loadSurf(subject,hemi,surf)