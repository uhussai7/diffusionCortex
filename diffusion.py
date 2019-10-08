import ioFunctions
import numpy as np
from scipy import interpolate
import dipy


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
                k = np.linspace(0, shapeimpoer[2] - 1, num=shape[2])
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


