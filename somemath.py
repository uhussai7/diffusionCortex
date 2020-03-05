from numpy import float64, hypot, zeros
from math import cos, pi, sin, atan2, sqrt
from copy import deepcopy
import numpy as np
import geodesic
from mayavi import mlab
import math
from anti_lib import Vec
import matplotlib.cm as cm
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from dipy.core.sphere import cart2sphere
from dipy.core.sphere import sphere2cart


# Global variables.
EULER_NEXT = [1, 2, 0, 1]    # Used in the matrix_indices() function.
EULER_TRANS_TABLE = {
        'xzx': [0, 1, 1],
        'yxy': [1, 1, 1],
        'zyz': [2, 1, 1],

        'xzy': [0, 1, 0],
        'yxz': [1, 1, 0],
        'zyx': [2, 1, 0],

        'xyx': [0, 0, 1],
        'yzy': [1, 0, 1],
        'zxz': [2, 0, 1],

        'xyz': [0, 0, 0],
        'yzx': [1, 0, 0],
        'zxy': [2, 0, 0]
}
EULER_EPSILON = 1e-5

def wrap_angles(angle, lower, upper, window=2*pi):
    """Convert the given angle to be between the lower and upper values.
    @param angle:   The starting angle.
    @type angle:    float
    @param lower:   The lower bound.
    @type lower:    float
    @param upper:   The upper bound.
    @type upper:    float
    @param window:  The size of the window where symmetry exists (defaults to 2pi).
    @type window:   float
    @return:        The wrapped angle.
    @rtype:         float
    """

    # Check the bounds and window.
    if window - (upper - lower) > 1e-7:
        raise ValueError("The lower and upper bounds [%s, %s] do not match the window size of %s." % (lower, upper, window))

    # Keep wrapping until the angle is within the limits.
    while True:
        # The angle is too big.
        if angle > upper:
            angle = angle - window

        # The angle is too small.
        elif angle < lower:
            angle = angle + window

        # Inside the window, so stop wrapping.
        else:
            break

    # Return the wrapped angle.
    return angle

def axis_angle_to_euler_zyz(axis, angle):
    """Convert the axis-angle notation to zyz Euler angles.
    This first generates a rotation matrix via axis_angle_to_R() and then used this together with R_to_euler_zyz() to obtain the Euler angles.
    @param axis:    The 3D rotation axis.
    @type axis:     numpy array, len 3
    @param angle:   The rotation angle.
    @type angle:    float
    @return:        The alpha, beta, and gamma Euler angles in the zyz convention.
    @rtype:         float, float, float
    """

    # Init.
    R = zeros((3, 3), float64)

    # Get the rotation.
    axis_angle_to_R(axis, angle, R)

    # Return the Euler angles.
    return R_to_euler_zyz(R)

def axis_angle_to_R(axis, angle, R):
    """Generate the rotation matrix from the axis-angle notation.
    Conversion equations
    ====================
    From Wikipedia (U{http://en.wikipedia.org/wiki/Rotation_matrix}), the conversion is given by::
        c = cos(angle); s = sin(angle); C = 1-c
        xs = x*s;   ys = y*s;   zs = z*s
        xC = x*C;   yC = y*C;   zC = z*C
        xyC = x*yC; yzC = y*zC; zxC = z*xC
        [ x*xC+c   xyC-zs   zxC+ys ]
        [ xyC+zs   y*yC+c   yzC-xs ]
        [ zxC-ys   yzC+xs   z*zC+c ]
    @param axis:    The 3D rotation axis.
    @type axis:     numpy array, len 3
    @param angle:   The rotation angle.
    @type angle:    float
    @param R:       The 3x3 rotation matrix to update.
    @type R:        3x3 numpy array
    """

    # Trig factors.
    ca = cos(angle)
    sa = sin(angle)
    C = 1 - ca

    # Depack the axis.
    x, y, z = axis

    # Multiplications (to remove duplicate calculations).
    xs = x*sa
    ys = y*sa
    zs = z*sa
    xC = x*C
    yC = y*C
    zC = z*C
    xyC = x*yC
    yzC = y*zC
    zxC = z*xC

    # Update the rotation matrix.
    R[0, 0] = x*xC + ca
    R[0, 1] = xyC - zs
    R[0, 2] = zxC + ys
    R[1, 0] = xyC + zs
    R[1, 1] = y*yC + ca
    R[1, 2] = yzC - xs
    R[2, 0] = zxC - ys
    R[2, 1] = yzC + xs
    R[2, 2] = z*zC + ca

def R_to_euler_zyz(R):
    """Convert the rotation matrix to the zyz Euler angles.
    @param R:       The 3x3 rotation matrix to extract the Euler angles from.
    @type R:        3D, rank-2 numpy array
    @return:        The alpha, beta, and gamma Euler angles in the zyz convention.
    @rtype:         tuple of float
    """

    # Redirect to R_to_euler()
    return R_to_euler(R, 'zyz')

def R_to_euler(R, notation, axes_rot='static', second_sol=False):
    """Convert the rotation matrix to the given Euler angles.
    This uses the algorithms of Ken Shoemake in "Euler Angle Conversion. Graphics Gems IV. Paul Heckbert (ed.). Academic Press, 1994, ISBN: 0123361567. pp. 222-229." (U{http://www.graphicsgems.org/}).
    The Euler angle notation can be one of:
        - xyx
        - xyz
        - xzx
        - xzy
        - yxy
        - yxz
        - yzx
        - yzy
        - zxy
        - zxz
        - zyx
        - zyz
    @param R:               The 3x3 rotation matrix to extract the Euler angles from.
    @type R:                3D, rank-2 numpy array
    @param notation:        The Euler angle notation to use.
    @type notation:         str
    @keyword axes_rot:      The axes rotation - either 'static', the static axes or 'rotating', the rotating axes.
    @type axes_rot:         str
    @keyword second_sol:    Return the second solution instead (currently unused).
    @type second_sol:       bool
    @return:                The alpha, beta, and gamma Euler angles in the given convention.
    @rtype:                 tuple of float
    """

    # Duplicate R to avoid its modification.
    R = deepcopy(R)

    # Get the Euler angle info.
    i, neg, alt = EULER_TRANS_TABLE[notation]

    # Axis rotations.
    rev = 0
    if axes_rot != 'static':
        rev = 1

    # Find the other indices.
    j, k, h = matrix_indices(i, neg, alt)

    # No axis repetition.
    if alt:
        # Sine of the beta angle.
        sin_beta = sqrt(R[i, j]**2 + R[i, k]**2)

        # Non-zero sin(beta).
        if sin_beta > EULER_EPSILON:
            alpha = atan2( R[i, j],   R[i, k])
            beta  = atan2( sin_beta,  R[i, i])
            gamma = atan2( R[j, i],  -R[k, i])

        # sin(beta) is zero.
        else:
            alpha = atan2(-R[j, k],   R[j, j])
            beta  = atan2( sin_beta,  R[i, i])
            gamma = 0.0

    # Axis repetition.
    else:
        # Cosine of the beta angle.
        cos_beta = sqrt(R[i, i]**2 + R[j, i]**2)

        # Non-zero cos(beta).
        if cos_beta > EULER_EPSILON:
            alpha = atan2( R[k, j],   R[k, k])
            beta  = atan2(-R[k, i],   cos_beta)
            gamma = atan2( R[j, i],   R[i, i])

        # cos(beta) is zero.
        else:
            alpha = atan2(-R[j, k],  R[j, j])
            beta  = atan2(-R[k, i],   cos_beta)
            gamma = 0.0

    # Remapping.
    if neg:
        alpha, beta, gamma = -alpha, -beta, -gamma
    if rev:
        alpha_old = alpha
        alpha = gamma
        gamma = alpha_old

    # Angle wrapping.
    if alt and -pi < beta < 0.0:
        alpha = alpha + pi
        beta = -beta
        gamma = gamma + pi

    alpha = wrap_angles(alpha, 0.0, 2.0*pi)
    beta  = wrap_angles(beta,  0.0, 2.0*pi)
    gamma = wrap_angles(gamma, 0.0, 2.0*pi)

    # Return the Euler angles.
    return alpha, beta, gamma

def matrix_indices(i, neg, alt):
    """Calculate the parameteric indices i, j, k, and h.
    This is one of the algorithms of Ken Shoemake in "Euler Angle Conversion. Graphics Gems IV. Paul Heckbert (ed.). Academic Press, 1994, ISBN: 0123361567. pp. 222-229."  (U{http://www.graphicsgems.org/}).
    The indices (i, j, k) are a permutation of (x, y, z), and the index h corresponds to the row containing the Givens argument a.
    @param i:   The index i.
    @type i:    int
    @param neg: Zero if (i, j, k) is an even permutation of (x, y, z) or one if odd.
    @type neg:  int
    @param alt: Zero if the first and last system axes are the same, or one if they are different.
    @type alt:  int
    @return:    The values of j, k, and h.
    @rtype:     tuple of int
    """

    # Calculate the indices.
    j = EULER_NEXT[i + neg]
    k = EULER_NEXT[i+1 - neg]

    # The Givens rotation row index.
    if alt:
        h = k
    else:
        h = i

    # Return.
    return j, k, h

def euler_to_axis_angle_zyz(alpha, beta, gamma):
    """Convert the zyz Euler angles to axis-angle notation.
    This function first generates a rotation matrix via euler_*_to_R() and then uses R_to_axis_angle() to convert to the axis and angle notation.
    @param alpha:   The alpha Euler angle in rad.
    @type alpha:    float
    @param beta:    The beta Euler angle in rad.
    @type beta:     float
    @param gamma:   The gamma Euler angle in rad.
    @type gamma:    float
    @return:        The 3D rotation axis and angle.
    @rtype:         numpy 3D rank-1 array, float
    """

    # Init.
    R = zeros((3, 3), float64)

    # Get the rotation.
    euler_to_R_zyz(alpha, beta, gamma, R)

    # Return the axis and angle.
    return R_to_axis_angle(R)

def R_to_axis_angle(R):
    """Convert the rotation matrix into the axis-angle notation.
    Conversion equations
    ====================
    From Wikipedia (U{http://en.wikipedia.org/wiki/Rotation_matrix}), the conversion is given by::
        x = Qzy-Qyz
        y = Qxz-Qzx
        z = Qyx-Qxy
        r = hypot(x,hypot(y,z))
        t = Qxx+Qyy+Qzz
        theta = atan2(r,t-1)
    @param R:   The 3x3 rotation matrix to update.
    @type R:    3x3 numpy array
    @return:    The 3D rotation axis and angle.
    @rtype:     numpy 3D rank-1 array, float
    """

    # Axes.
    axis = zeros(3, float64)
    axis[0] = R[2, 1] - R[1, 2]
    axis[1] = R[0, 2] - R[2, 0]
    axis[2] = R[1, 0] - R[0, 1]

    # Angle.
    r = hypot(axis[0], hypot(axis[1], axis[2]))
    t = R[0, 0] + R[1, 1] + R[2, 2]
    theta = atan2(r, t-1)

    # Normalise the axis.
    if r != 0.0:
        axis = axis / r

    # Return the data.
    return axis, theta

def euler_to_R_zyz(alpha, beta, gamma, R):
    """Generate the z-y-z Euler angle convention rotation matrix.
    Rotation matrix
    ===============
    The rotation matrix is defined as the vector of unit vectors::
        R = [mux, muy, muz].
    According to wikipedia (U{http://en.wikipedia.org/wiki/Euler_angles#Table_of_matrices}), the rotation matrix for the zyz convention is::
              | -sa*sg + ca*cb*cg   -ca*sg - sa*cb*cg    sb*cg            |
        R  =  |  sa*cg + ca*cb*sg    ca*cg - sa*cb*sg    sb*sg            |,
              | -ca*sb               sa*sb               cb               |
    where::
        ca = cos(alpha),
        sa = sin(alpha),
        cb = cos(beta),
        sb = sin(beta),
        cg = cos(gamma),
        sg = sin(gamma).
    @param alpha:   The alpha Euler angle in rad for the z-rotation.
    @type alpha:    float
    @param beta:    The beta Euler angle in rad for the y-rotation.
    @type beta:     float
    @param gamma:   The gamma Euler angle in rad for the second z-rotation.
    @type gamma:    float
    @param R:       The 3x3 rotation matrix to update.
    @type R:        3x3 numpy array
    """

    # Trig.
    sin_a = sin(alpha)
    cos_a = cos(alpha)
    sin_b = sin(beta)
    cos_b = cos(beta)
    sin_g = sin(gamma)
    cos_g = cos(gamma)

    # The unit mux vector component of the rotation matrix.
    R[0, 0] = -sin_a * sin_g  +  cos_a * cos_b * cos_g
    R[1, 0] =  sin_a * cos_g  +  cos_a * cos_b * sin_g
    R[2, 0] = -cos_a * sin_b

    # The unit muy vector component of the rotation matrix.
    R[0, 1] = -cos_a * sin_g  -  sin_a * cos_b * cos_g
    R[1, 1] =  cos_a * cos_g  -  sin_a * cos_b * sin_g
    R[2, 1] =  sin_a * sin_b

    # The unit muz vector component of the rotation matrix.
    R[0, 2] =  sin_b * cos_g
    R[1, 2] =  sin_b * sin_g
    R[2, 2] =  cos_b

def EulerZYZ2AxisAngle(alpha, beta, gamma):
    phi=(np.pi + alpha - gamma)/2
    singammaalpha2=np.sin(gamma+alpha)/2
    if(singammaalpha2!=0):
        theta= np.arctan( np.tan(beta/2)/(singammaalpha2) )
    else:
        theta=0
    cbeta2=np.cos(beta/2)
    cbeta2=cbeta2*cbeta2
    cosalphagamma2=np.cos((alpha+gamma)/2)
    cosalphagamma2=cosalphagamma2*cosalphagamma2
    psi = np.arccos( 2*cbeta2*cosalphagamma2-1)

    return theta,phi,psi

def parity_symmetrize(grid):
    grid=np.asarray(grid)
    shp=grid.shape
    N=shp[0]
    for i in range(0,int(N/2)):
        for j in range(0,N):
            it=N-1-i
            jt=(j+int(N/2))%N
            temp=0.5*(grid[i,j]+grid[it,jt])
            grid[i,j]=temp
            grid[it,jt]=temp
    return grid
class sphereSig():
    def __init__(self):
        self.grid=[]
        self.N=[]

    def plot(self):
        N=self.N
        theta = np.linspace(0, np.pi, N)
        phi = np.linspace(0, 2 * np.pi, N)
        theta, phi = np.meshgrid(theta, phi,indexing='ij')

        # The Cartesian coordinates of the unit sphere
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)

        fcolors = self.grid
        fmax, fmin = fcolors.max(), fcolors.min()
        fcolors = (fcolors - fmin)/(fmax - fmin)

        # Set the aspect ratio to 1 so our sphere looks spherical
        fig = plt.figure(figsize=plt.figaspect(1.))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=cm.seismic(fcolors))
        # Turn off the axis planes
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    def gauss(self,theta_o,phi_o,sig,N):
        self.N=N
        theta = np.linspace(0, np.pi, N)
        phi = np.linspace(0, 2 * np.pi, N)
        theta, phi = np.meshgrid(theta, phi,indexing='ij')

        ex_th=(theta-theta_o)*(theta-theta_o)/(2*sig*sig)
        ex_ph=(phi-phi_o)*(phi-phi_o)/(2*sig*sig)

        self.grid=np.exp(-ex_th-ex_ph)
        self.grid=self.grid

    def cap_north(self,N):
        self.N=N
        self.grid=np.empty([N,N])
        self.grid[:,:]=0
        self.grid[0:4,:]=1

    def cap_north_hetro(self,N):
        self.N=N
        self.grid=np.empty([N,N])
        self.grid[:,:]=0
        self.grid[0:4,:]=np.linspace(0,1,num=N)

    def square_x(self,N):
        self.N=N
        self.grid = np.empty([N, N])
        self.grid[:,:] = 0
        mid=int(N/2)
        self.grid[mid-2:mid+2,0:2] = 1
        self.grid[mid-2:mid+2,N-2:N-1] = 1

    # def square_y(self):
    #     N = self.N
    #     self.grid = np.empty([N, N])
    #     self.grid = 0
    #     mid = int(N / 2)
    #     self.grid[mid - 2:mid + 2,] = 1



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
        self.new_s_flat=[]

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

    def makeFlat(self,interpolator): #
        m= self.m
        N= m+1
        height = N+1
        width = 5*(N+1)
        ii = np.linspace(0, N - 1, N)
        jj = np.linspace(0, N - 1, N)
        jjj, iii = np.meshgrid(ii, jj)
        jjj = np.flip(jjj)
        iii = np.transpose(iii)
        jjj = np.transpose(jjj)
        self.s_flat=np.empty([height,width])
        self.s_flat[:]=0
        for c in range(0,5):
            for ii in range(0,N):
                for jj in range(0,N):
                    I=ii
                    J=(N+1)*c + jj + 1
                    i = iii[ii, jj]
                    j = jjj[ii, jj]
                    theta,phi = self.cij2thetaphi(c, i, j)
                    x_t, y_t, z_t = sphere2cart(1,theta,phi)
                    #val1=interpolator(theta,phi)
                    val1=interpolator(x_t,y_t,z_t)
                    x_t, y_t, z_t = sphere2cart(1, np.pi-theta,phi+np.pi)
                    #val2=interpolator(np.pi-theta,phi+np.pi)
                    val2 = interpolator(x_t, y_t, z_t)
                    self.s_flat[I,J]= 0.5*(val1+val2)
        for c in range(0,5): #padding
            ct=c-1
            #print((N+1)*ct+1)
            #print((N+1)*ct+4)
            self.s_flat[1:N,c*(N+1)] = self.s_flat[1,(N+1)*ct+1:(N+1)*c-1]
            ct=(c+2)%N
            self.s_flat[N,(N+1)*c+1:(N+1)*c+N] =np.flip(self.s_flat[1:N,(ct+1)*(N+1)-2])

        stacks=np.empty([N+1,N+1,5])
        self.new_s_flat=np.empty([5*(N+1),N+1])
        for c in range(0,5): #do a re-stack to be consistent with taco
            stacks[:,:,c]=self.s_flat[0:N+1,c*(N+1):c*(N+1)+N+1]
            self.new_s_flat[-(N+1)*(c+1):5*(N+1)-c*(N+1),:]=self.s_flat[0:N+1,c*(N+1):c*(N+1)+N+1]

    def cij2thetaphi(self,c,i,j):
        v=int(self.mat2mesh(i,j))
        if (j>i):
            t=0
        else:
            t=1
        xyz=self.Points[c][t][v]
        r, theta, phi = cart2sphere(xyz[0],xyz[1],xyz[2])
        return theta, phi

