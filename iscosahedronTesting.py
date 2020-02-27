import numpy as np
import geodesic
from mayavi import mlab
import math
from anti_lib import Vec
import matplotlib.cm as cm
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import diffusion



class isomesh:
    def __init__(self):
        self.faces=[] #this is the full icosahedron
        self.vertices=[] #this is the full icosahedron
        self.chart_grid=[]
        self.phi = (math.sqrt(5) + 1) / 2
        self.rad = math.sqrt(self.phi + 2)
        self.chart_vertices = [] #the ordering of these vertices is different than the ordering of self.vertices
        self.chart_faces=[] #the faces are arranged so that 0:3, 4:7, etc. are each chart
        self.Points=[] #this is temporary
        self.m= 4 #parameter for geodesic mesh
        self.testStrip=np.empty([self.m+1,self.m+1]) #note that this is a half strip
        self.sphereFunction=np.empty([5,2,100])
        self.s_flat=[]

    def get_icomesh(self):
        geodesic.get_icosahedron(self.vertices, self.faces)
        X = 1 / self.rad
        Z = self.phi / self.rad
        self.rotate_y_anagle=math.pi/2 - math.atan(Z/X)
        rot_y=R.from_euler('y',self.rotate_y_anagle)
        #self.vertices=np.asarray(self.vertices)
        #self.faces = np.asarray(self.faces)
        #self.chart_vertices.extend([Vec(X, 0, Z), Vec(-X, 0, Z),
         # Vec(0, Z, X), Vec(Z, X, 0),
         # Vec(0, Z, -X), Vec(X, 0, -Z),
         # Vec(0, -Z, X), Vec(Z, -X, 0),
         # Vec(0, -Z, -X), Vec(-Z, -X, 0),
         # Vec(-X, 0, -Z), Vec(-Z, X, 0)])
        self.chart_vertices.extend([Vec(0.894427,0.000000,0.447214),
            Vec(0.000000,0.000000,1.000000),
            Vec(0.276393,0.850651,0.447214),
            Vec(0.723607,0.525731,-0.447214),
            Vec(-0.276393,0.850651,-0.447214),
            Vec(-0.000000,0.000000,-1.000000),
            Vec(0.276393,-0.850651,0.447214),
            Vec(0.723607,-0.525731,-0.447214),
            Vec(-0.276393,-0.850651,-0.447214),
            Vec(-0.723607,-0.525731,0.447214),
            Vec(-0.894427,0.000000,-0.447214),
            Vec(-0.723607,0.525731,0.447214)])
        self.chart_faces=[[2,0,1],[0,2,3],
          [4,3,2],[3,4,5],
          [11, 2, 1], [2, 11, 4],
          [10, 4, 11], [4, 10, 5],
          [9, 11, 1], [11, 9, 10],
          [8, 10, 9], [10, 8, 5],
          [6, 9, 1], [9, 6, 8],
          [7, 8, 6], [8, 7, 5],
          [0, 6, 1], [6, 0, 7],
          [3, 7, 0], [7, 3, 5]
           ]
        self.vertices=self.chart_vertices
        self.faces=self.chart_faces


        m = self.m
        n = 0
        reps = 1
        repeats = 1 * reps
        freq = repeats * (m * m + m * n + n * n)
        grid = geodesic.make_grid(freq, m, n)
        self.chart_grid=grid
        #points = self.chart_vertices
        Points=[]
        for c in range(0,5):
            tempfaces=self.chart_faces[4*c:4*c+4]
            points = []
            f=0
            for face in tempfaces:
                face_edges=(0,0,0)
                if(f%2==0):
                   face_edges = (1,0,0)
                temp=geodesic.grid_to_points(
                    grid, freq, True,
                    [self.chart_vertices[face[i]] for i in range(3)],face_edges)#,(0,0,0)
                if(f % 2 ==0):
                    points.append(np.flip(temp))
                else:
                    points.append(temp)
                f=f+1
            Points.append(points)
        self.Points=np.asarray(Points)

    def plot_icomesh(self):
        x=[]
        y=[]
        z=[]
        triangles=[]
        for vertex in self.vertices:
            x.append(vertex[0])
            y.append(vertex[1])
            z.append(vertex[2])
        triangles=np.row_stack([face for face in self.faces])
        mlab.triangular_mesh(x,y,z,triangles)
        # mlab.show()

    #def constructStrip(self): #for testing


    def mat2mesh(self,i,j):
        m = self.m
        N = m + 1
        if (j > i):
            out=N * i + j - (i + 1) * (i + 2) / 2 # this is the formula for the upper triangle

        else:
            pos = N * i + j
            neg1 = (i * (i - 1)) / 2
            neg2 = i * (N - i)
            out=pos - neg1 - neg2

        return out

    def makeFlat(self,): #fills up vertices of mesh with colors
        m= self.m
        N= m+1
        height = N+1
        width = 5*(N+1)
        self.s_flat=np.empty([height,width])
        self.s_flat[:]=0
        for c in range(0,5):
            for i in range(0,N):
                for j in range(0,N):
                    I=i
                    J=(N+1)*c + j + 1
                    self.s_flat[I,J]=self.cij2thetaphi(c,i,j)

    def cij2thetaphi(self,c,i,j):
        v=self.mat2mesh(i,j)
        if (j>i):
            t=0
        else:
            t=1
        #self.Points[c][t][v]
        return v#((0*c+t/10+v/100)*100)

   # def makeSquares(self):
    #     m = self.m
    #     N = m + 1
    #     np.empty([])

iso=isomesh()
iso.get_icomesh()
iso.makeFlat()
plt.imshow(iso.s_flat)
plt.colorbar()
iso.plot_icomesh()
c=0
for c in range(0,1):
    for i in range(1,2):
        d=0
        for point in iso.Points[c][i][0:1]:
            x = []
            y = []
            z = []
            x.append(point[0])
            y.append(point[1])
            z.append(point[2])
            col=cm.BuPu(d/12)
            mlab.points3d(x, y, z, scale_factor=0.08,color=col[0:3])#(1/(d*2),1/d,1/(1+d)))
            d=d+1

mlab.show()
dvol=diffusion.diffVolume()
dvol.getVolume("C:\\Users\\uhussain\\Documents\\ShareVM\\Cortex\\101006\\Diffusion\\Diffusion")
dvol.shells()



# for point in dvol.bvecs[1]:
#     x = []
#     y = []
#     z = []
#     print(point)
#     x.append(point[0])
#     y.append(point[1])
#     z.append(point[2])
#     mlab.points3d(x, y, z, scale_factor=0.08, color=col[0:3])  # (1/(d*2),1/d,1/(1+d)))
# mlab.show()


