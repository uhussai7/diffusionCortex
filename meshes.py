import ioFunctions
import numpy as np
from scipy import interpolate

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

class Submesh:
    def __init__(self):
        self.vertices = []
        self.faces = []

    def getSubmesh(self, surface, gyrusIndex): #filter out a submesh based on a label for vertices
        if surface.aparc is None:
            raise ValueError("Aparc not loaded")
        print("fetching submesh as a new mesh")
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
        self.volume_info = []

    def getSurf(self,subject=None, hemi=None, surf=None,**kwargs):
        coords, faces, self.volume_info  = ioFunctions.loadSurf(subject,hemi,surf)
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


class Projection: #this is to project volume data onto mesh
    def __init__(self):
        self.vertices=[]
        self.scalar=[]
        self.vector=[]
        self.tensor=[]

    def project(self, vol=None, mesh=None, header=None):
        #vol should be nibabel nifti object and mesh is surface we want to project onto
        if vol is None:
            raise ValueError("no volume provided")
        if mesh is None:
            raise ValueError("no mesh to be projected on provided")
        if header is None:
            raise ValueError("no orig.mgz header provided, need this for transforms")

        #we will bring vertex coordinates to voxel space and then interpolate
        #inverse of tkrvox2ras will bring us to voxel space of orig
        #sfrom of orig brings us to world coordinates
        #inverse of sform of vol brings the coordinates to voxel space of vol where we can interpolate on grid

        Torig=np.asmatrix(header.get_vox2ras_tkr())
        Norig=np.asmatrix(header.get_vox2ras())
        sform=np.asmatrix(vol.get_sform())

        xfm=np.matmul(Norig, np.linalg.inv(Torig))
        xfm=np.matmul(np.linalg.inv(sform), xfm)

        shape=vol.shape
        img=vol.get_data()
        print(shape)
        if shape[3] == 3:
            i = np.linspace(0, shape[0] - 1, num=shape[0])
            j = np.linspace(0, shape[1] - 1, num=shape[1])
            k = np.linspace(0, shape[2] - 1, num=shape[2])
            interpolator = []
            #print(i)
            #print(j)
            #print(k)
            #for f in range(shape[3]):
            interpolator = [interpolate.RegularGridInterpolator((i, j, k), img[:, :, :, f]) for f in range(shape[3])]

        for vertex in mesh.vertices:
            self.vertices.append(Vertex(coords=vertex.coords))
            point=np.asarray(vertex.coords)
            point=np.append(point,1)
            point=np.squeeze(np.asarray(np.matmul(xfm, point)))  #this line is so bloated has to be better way

           # if shape[3]==1:

            if shape[3]==3:
                temp1=[0,0,0]
                #print(point[0:3])
                temp=point[0:3]
                #print(temp)
                for j in range(shape[3]):
                    #print(j)
                    temp1[j]=float(interpolator[j](temp))
                self.vector.append(np.asarray(temp1))
