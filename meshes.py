import ioFunctions
import numpy as np
from scipy import interpolate
from mayavi import mlab
import igl

class Face:
    def __init__(self,vertex_ids=None):
        """
        A class for faces of triangulation
        :param vertex_ids: 3 ints for vertex ids
        """
        self.vertex_ids = vertex_ids

class Vertex:
    def __init__(self,coords=None):
        """
        A class for vertices making up a triangulation
        :param coords: 3D coordinates of each vertex
        :param faces: Id's of faces that the vertex belongs to
        """
        self.coords = np.asarray(coords)
        self.faces = []

class Normals:
    def __init__(self):
        self.vectors = []

    def getNormals(self, vertices=None, faces=None):

        self.vectors=igl.per_vertex_normals()



class Annot:
    def __init__(self):
        """
        A class to hold Freesurfer annotations
        """
        self.labels = []
        self.ctab = []
        self.names= []

class Submesh:
    def __init__(self):
        """
        A class to generate submeshes (gyri, sulci, etc.) based on Freesurfer labels
        """
        self.vertices = []
        self.faces = []

    def getSubmesh(self, surface, index): #filter out a submesh based on a label for vertices
        """
        Gets submesh based on freesurfer labels, submesh is a new mesh with new indices for faces, vertices
        :param surface: Mesh to extract submesh from, usually a hemisphere
        :param index: Freesurfer label of region to extract
        :return: Will store new mesh in self.vertices and self.faces of same class
        """
        if surface.aparc is None:
            raise ValueError("Aparc not loaded")
        print("fetching submesh as a new mesh")
        inds = np.where(surface.aparc.labels == index)
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
        """
        Class to store surface meshes
        """
        self.vertices= []
        self.faces = []
        self.aparc = Annot() #this will be for lh/rh.aparc.annot
        self.volume_info = []

    def getSurf(self,subject=None, hemi=None, surf=None,**kwargs):
        """
        Gets Freesurfer mesh using loadSurf
        :param subject: Subject id
        :param hemi: Which hemisphere
        :param surf: Which surface (pial, white, etc.)
        :param kwargs:
        :return: Will store new mesh in self.vertices and self.faces of same class
        """
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
        """
        Gets the parcellation labels from Freesurfer
        :param subject: Subject id
        :param hemi: Which hemisphere
        :param kwargs:
        :return: Will store in self.aparc which is Annot() class
        """
        self.aparc.labels, self.aparc.ctab, self.aparc.names = ioFunctions.loadAparc(subject,hemi)


class Projection: #this is to project volume data onto mesh
    def __init__(self):
        """
        Class for projecting data onto surfaces
        """
        self.vertices=[]
        self.scalar=[]
        self.vector=[]
        self.tensor=[]
        self.points=[]

    def project(self, volume=None, mesh=None, header=None):
        """
        Project data onto surface
        :param volume: Volume class for gridded data
        :param mesh: Mesh of surface to project data on
        :param header: Header from original.mgz to create correct xfm
        :return: Fills out self.scalar, self.vector etc. in same class
        """
        #vol should be nibabel nifti object and mesh is surface we want to project onto
        if volume is None:
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
        sform=np.asmatrix(volume.vol.get_sform())

        xfm=np.matmul(Norig, np.linalg.inv(Torig))
        xfm=np.matmul(np.linalg.inv(sform), xfm)

        shape=volume.vol.shape

        points=[]
        for vertex in mesh.vertices:
            #self.vertices.append(Vertex(coords=vertex.coords))
            #point=np.asarray(vertex.coords)
            point = vertex.coords
            point=np.append(point,1)
            point=np.asarray(xfm @ point) #this line is so bloated has to be better way
            self.points.append(point[0,0:3])

        #TODO have to put something for other stuff too like scalars, etc.
        #if shape[3]==1:
        if shape[3]==3:
            temp=[]
            for j in range(shape[3]):
                temp.append(volume.interpolator[j](self.points))
            self.vector = np.column_stack((temp[0],temp[1],temp[2]))


class Foliation: #this will load a whole brain foliation
    def __init__(self,N=None):
        """
        A class to store a foliation, i.e., a family of surfaces layering the cortex
        :param N: Number of surfaces in the foliation
        """
        if N is None:
            raise ValueError("please call with number of surfaces")
        self.N_s=N
        self.surfaces=[Surface() for surf in range(self.N_s)]

    def getFoliation(self,subject=None, hemi=None):
        """
        Gets the foliation surfaces
        :param subject: Subject id
        :param hemi: Which hemisphere
        :return: Fills out self.surfaces which is an array of dim N_s of the Surface class
        """
        N=1
        for surf in self.surfaces:
            surface="equi"+str(N)+".pial"
            surf.getSurf(subject=subject,hemi=hemi,surf=surface)
            N = N + 1

class FoliationProjection():
    def __init__(self,foliation=None,volume=None, header=None):
        """
        Projects data onto a foliation
        :param foliation: an object of the foliation class
        :param volume: Volume class for gridded data
        :param header: Header from original.mgz to create correct xfm
        """
        self.projection=[Projection() for i in range(foliation.N_s)]
        if volume.interpExists==0:
            volume.makeInterpolator()
        for i in range(foliation.N_s):
            print("projecting on surface"+ str(i))
            self.projection[i].project(volume=volume,mesh=foliation.surfaces[i],header=header)

class Volume():
    def __init__(self):
        """
        Class for storing gridded volume data
        """
        self.vol = []
        self.interpExists = 0
        self.interpolator = []

    def getVolume(self, filename=None):
        """
        Gets volume data
        :param filename: Path of volume file
        :return:
        """
        self.vol=ioFunctions.loadVol(filename=filename)

    def makeInterpolator(self):
        """
        Makes a linear interpolator
        :return: Fills out self. interpolator and sets self.interpExists = 1 after interpolator is calculated
        """
        shape = self.vol.shape
        img = self.vol.get_data()
        #TODO other shapes like scalars most impot
        if shape[3] == 3:
            i = np.linspace(0, shape[0] - 1, num=shape[0])
            j = np.linspace(0, shape[1] - 1, num=shape[1])
            k = np.linspace(0, shape[2] - 1, num=shape[2])
            self.interpolator = [interpolate.RegularGridInterpolator((i, j, k), img[:, :, :, f]) for f in range(shape[3])]
            self.interpExists=1





class Vision:
    def __init__(self):
        mesh=[]
        scalar=[]
        vector=[]
        x=[]
        y=[]
        z=[]
        triangles=[]
        vector_added=[]

    def processMesh(self, mesh=None):
        self.mesh=mesh
        x=[]
        y=[]
        z=[]
        triangles=[]
        for vertex in self.mesh.vertices:
            x.append(vertex.coords[0])
            y.append(vertex.coords[1])
            z.append(vertex.coords[2])
        triangles = np.row_stack([face.vertex_ids for face in self.mesh.faces])
        self.x=x
        self.y=y
        self.z=z
        self.triangles=triangles
        mlab.triangular_mesh(self.x,self.y,self.z,self.triangles)
        #mlab.triangular_mesh(x, y, z, triangles)

    def addVector(self, vector=None): #for now this will take vector fields from Projection class
        tempVector=np.row_stack([thisVector for thisVector in vector])
        self.vector=tempVector
        self.vector_added=1


    def show(self):
        mlab.triangular_mesh(self.x,self.y,self.z,self.triangles)
        if self.vector_added==1:
            mlab.quiver3d(self.x, self.y, self.z, self.vector[:, 0], self.vector[:, 1], self.vector[:, 2])
        mlab.show()


