import numpy as np
import geodesic
from mayavi import mlab
import math
from anti_lib import Vec
import matplotlib.cm as cm

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
    def get_icomesh(self):
        geodesic.get_icosahedron(self.vertices, self.faces)
        X = 1 / self.rad
        Z = self.phi / self.rad
        #self.vertices=np.asarray(self.vertices)
        #self.faces = np.asarray(self.faces)
        self.chart_vertices.extend([Vec(X, 0, Z), Vec(-X, 0, Z),
         Vec(0, Z, X), Vec(Z, X, 0),
         Vec(0, Z, -X), Vec(X, 0, -Z),
         Vec(0, -Z, X), Vec(Z, -X, 0),
         Vec(0, -Z, -X), Vec(-Z, -X, 0),
         Vec(-X, 0, -Z), Vec(-Z, X, 0)])
        self.chart_faces=[[2,0,1],[0,2,3],
          [4,3,2],[5,3,4],
          [0, 1, 6], [1, 6, 7],
          [1, 7, 3], [7, 3, 5],
          [0, 6, 9], [6,8, 9],
          [6, 7, 8], [7, 8, 5],
          [0, 9, 11], [9, 11, 10],
          [9, 10, 8], [8, 5, 10],
          [0, 2, 11], [2, 11, 4],
          [11, 4, 10], [4, 10, 5]
           ]

        m = 9
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
                face_edges = face
                print(face)
                if(f % 2 == 1):
                    face_edges=(0,0,0)
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

iso=isomesh()
iso.get_icomesh()
iso.plot_icomesh()
c=0
d=1
#for i in range(0,4):
i=0
for i in range(0,2):
    d=0
    for point in iso.Points[c][i]:
        x = []
        y = []
        z = []
        x.append(point[0])
        y.append(point[1])
        z.append(point[2])
        col=cm.BuPu(d/6)
        mlab.points3d(x, y, z, scale_factor=0.08,color=col[0:3])#(1/(d*2),1/d,1/(1+d)))
        d=d+1
        print(d)

mlab.show()


#need to make some protoype for triangular matrix,
# face (5) -> triangle pair (2) -> matrix
# there will be overlap in the bottom triangle pair and the one on top
# still have to get ordering of vertices correct
# we want a map from the vertex coordinates to the matrix indices
#       -use triangular matrix for this although should be able to derive formula for this

#make triangular matrix
m=5 #this controls the number of vertices
flat=np.ones([m+1,m+1])
flat_upper=np.triu(flat) #the way things are set up upper here includes the diagnol itself

#for each i,j

m=6
i=5
j=5
N=m+1
if(j>i):
    print(N*i+j+1-(i+1)*(i+2)/2) #this is the formula for the upper triangle
else:
    pos=N*i+j
    neg1=(i*(i-1))/2
    neg2=i*(N-i)
    print(pos-neg1-neg2)
# phi = (math.sqrt(5) + 1) / 2
# rad = math.sqrt(phi+2)
#
#
for c in range(0,5):
    print(c)
    x=[]
    y=[]
    z=[]
    triangles=[]
    for vertex in new_vertices:
        x.append(vertex[0])
        y.append(vertex[1])
        z.append(vertex[2])
    tempfaces=new_faces[4*c:4*c+4]
    triangles=np.row_stack([face for face in tempfaces])
    d=c+1
    mlab.triangular_mesh(x,y,z,triangles,color=(1/(d*2),1/d,1/(1+d)))
    mlab.show()
#
#
# del grid
# del points
# del x,y,z
#
#
# x=[]
# y=[]
# z=[]
# for point in points:
#     x.append(point[0])
#     y.append(point[1])
#     z.append(point[2])
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x,y,z)
#
# iso.plot_icomesh()
# mlab.points3d(x,y,z,scale_factor=0.08)