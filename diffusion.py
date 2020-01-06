import ioFunctions
import numpy as np
from scipy import interpolate
import dipy
import s2cnn
from scipy.interpolate import griddata
import torch
import s2conv

class diff2d():
    def __init__(self):
        self.vox=[]
        self.signal=[]

    def loadData(self,dvol=None):
        if dvol is None:
            raise ValueError("please call with a diffVolume object")
        img=dvol.vol.getData()
        shape=img.shape
        img=img.reshape((shape[0]*shape[1]*shape[2],shape[3]),order='F')


class diffVolume():
    def __init__(self):
        """
        Class for storing gridded volume data
        """
        self.vol = []
        self.interpExists = 0
        self.interpolator = []
        self.bvals = []
        self.bvecs = []
        self.bvecs_hemi_cart=[]
        self.bvecs_hemi_sphere=[]
        self.inds= []
        self.gtab = []
        self.img=[]

    def getVolume(self, folder=None):
        """
        Gets volume data
        :param filename: Path of volume file
        :return:
        """
        self.vol, self.gtab =ioFunctions.loadDiffVol(folder=folder)
        self.img = self.vol.get_data()

    def makeInterpolator(self):
        """
        Makes a linear interpolator
        :return: Fills out self. interpolator and sets self.interpExists = 1 after interpolator is calculated
        """
        shape = self.vol.shape
        print(shape)
        img = self.vol.get_data()
        #TODO other shapes like scalars most impot
        if  len(shape) > 3:
            if shape[3] == 3:
                i = np.linspace(0, shape[0] - 1, num=shape[0])
                j = np.linspace(0, shape[1] - 1, num=shape[1])
                k = np.linspace(0, shape[2] - 1, num=shape[2])
                self.interpolator = [interpolate.RegularGridInterpolator((i, j, k), img[:, :, :, f]) for f in range(shape[3])]
                self.interpExists=1
            if shape[3]==1:
                i = np.linspace(0, shape[0] - 1, num=shape[0])
                j = np.linspace(0, shape[1] - 1, num=shape[1])
                k = np.linspace(0, shape[2] - 1, num=shape[2])
                self.interpolator = interpolate.RegularGridInterpolator((i, j, k), img[:, :, :,0])
                self.interpExists = 1
        else:
            i = np.linspace(0, shape[0] - 1, num=shape[0])
            j = np.linspace(0, shape[1] - 1, num=shape[1])
            k = np.linspace(0, shape[2] - 1, num=shape[2])
            self.interpolator = interpolate.RegularGridInterpolator((i, j, k), img[:, :, :])
            self.interpExists = 1

    def shells(self):
        tempbvals=[]
        tempbvals=np.round(self.gtab.bvals,-2)
        inds_sort=np.argsort(tempbvals)
        bvals_sorted=self.gtab.bvals[inds_sort]
        bvecs_sorted=self.gtab.bvecs[inds_sort]
        tempbvals=np.sort(tempbvals)
        gradbvals=np.gradient(tempbvals)
        inds_shell_cuts=np.where(gradbvals!=0)
        shell_cuts=[]
        for i in range(int(len(inds_shell_cuts[0]) / 2)):
            shell_cuts.append(inds_shell_cuts[0][i * 2])
        shell_cuts.insert(0,-1)
        shell_cuts.append(len(bvals_sorted))
        print(shell_cuts)
        print(bvals_sorted.shape)
        temp_bvals=[]
        temp_bvecs=[]
        temp_inds=[]
        for t in range(int(len(shell_cuts)-1)):
            print(shell_cuts[t]+1,shell_cuts[t + 1])
            temp_bvals.append(bvals_sorted[shell_cuts[t]+1:1+shell_cuts[t+1]])
            temp_bvecs.append(bvecs_sorted[shell_cuts[t]+1:1+shell_cuts[t+1]])
            temp_inds.append(inds_sort[shell_cuts[t]+1:1+shell_cuts[t+1]])
        self.bvals=temp_bvals
        self.bvecs=temp_bvecs
        self.inds=temp_inds
        self.inds=np.asarray(self.inds)


        pi=3.14159265
        for bvecs in self.bvecs: #this is shells
            temp_bvec = []
            temp_vec = []
            for bvec in bvecs: #this is each vector in shell
                r, theta, phi=dipy.core.sphere.cart2sphere(bvec[0],bvec[1],bvec[2])
                #if theta > pi/2:
                #    theta= pi- theta
                #    phi=phi+3.14159265
                phi=(phi)%(2*pi)
                x,y,z=dipy.core.sphere.sphere2cart(1,theta,phi)
                temp_vec.append([x,y,z])
                temp_bvec.append([r,theta,phi])
            self.bvecs_hemi_sphere.append(temp_bvec)
            self.bvecs_hemi_cart.append(temp_vec)
        self.bvecs_hemi_cart=np.asarray(self.bvecs_hemi_cart)
        self.bvecs_hemi_sphere=np.asarray(self.bvecs_hemi_sphere)

    def conv(self,p1,p2,N,shellN):
        """
        :param p1: voxel 1 coordinates
        :param p2: voxel 2 coordinates
        :param N:  size of convolution plane
        :param shellN: number of shells including b_0
        :return: SO(3) function
        """
        #put functions on grid
        so3=s2conv.so3()
        so3.makeSo3(N,shellN)
        #so3=np.empty([N,N,N,2,shellN])
        for shell in range(0,shellN):
            s1 = []
            s2 = []
            th = []
            ph = []
            i = 0
            for ind in self.inds[shell]:
                s1.append(self.img[p1[0], p1[1], p1[2], ind])
                s2.append(self.img[p2[0], p2[1], p2[2], ind])
                th.append(self.bvecs_hemi_sphere[shell][i][1])
                ph.append(self.bvecs_hemi_sphere[shell][i][2])
                i = i + 1
            th = np.asarray(th)
            ph = np.asarray(ph)
            s1 = np.asarray(s1)
            s2 = np.asarray(s2)
            b=int(N/2)
            pi=3.14159265359
            theta=np.linspace(0,pi,N)
            phi=np.linspace(0,2*pi,N)
            ss1 = griddata((th, ph), 100000/s1, (theta[None,:], phi[:,None]), method='nearest')
            ss2= griddata((th, ph), 100000/s2, (theta[None,:], phi[:,None]), method='nearest')

            sss1 = np.empty([N, N, 2])
            sss1[:, :, 0] = np.real(ss1)
            sss1[:, :, 1] = np.imag(ss1)

            sss2 = np.empty([N, N, 2])
            sss2[:, :, 0] = np.real(ss2)
            sss2[:, :, 1] = np.imag(ss2)

            so3.signal1[:,:,shell] = np.real(ss1)
            so3.signal2[:,:,shell] = np.real(ss2)


            g1 = torch.tensor(sss1, dtype=torch.float)
            g1ft = s2cnn.soft.s2_fft.s2_fft(g1, b_out=b)
            g1ftn = g1ft.numpy()

            g2 = torch.tensor(sss2, dtype=torch.float)
            g2ft = s2cnn.soft.s2_fft.s2_fft(g2, b_out=b)
            g2ftn = g2ft.numpy()

            # lets try to do a convoluion
            xn1 = np.empty([b * b, 1, 1, 2])
            xn1[:, 0, 0, 0] = g1ftn[:, 0]
            xn1[:, 0, 0, 1] = g1ftn[:, 1]
            x1 = torch.tensor(xn1, dtype=torch.float)

            # lets try to do a convoluion
            xn2 = np.empty([b * b, 1, 1, 2])
            xn2[:, 0, 0, 0] = g2ftn[:, 0]
            xn2[:, 0, 0, 1] = g2ftn[:, 1]
            x2 = torch.tensor(xn2, dtype=torch.float)

            xx = s2cnn.s2_mm(x1, x2)
            xxift = s2cnn.so3_fft.so3_ifft(xx)
            # xxift=s2cnn.so3_fft.SO3_ifft_real.apply(xx)
            xxiftn = xxift.numpy()
            xxiftnsmall = np.empty([N, N, N,2])
            xxiftnsmall = xxiftn[0, 0, :, :, :, :]
            so3.so3[:,:,:,:,shell]=xxiftnsmall #[beta, alpha, gamma, complex] beta=[0, pi], alpha=[0,2pi]  gamma=[0,2pi]
        return so3