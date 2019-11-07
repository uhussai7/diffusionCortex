import ioFunctions
import numpy as np
from scipy import interpolate
import dipy


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
        self.inds= []
        self.gtab = []

    def getVolume(self, folder=None):
        """
        Gets volume data
        :param filename: Path of volume file
        :return:
        """
        self.vol, self.gtab =ioFunctions.loadDiffVol(folder=folder)


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

