import numpy as np
import nibabel as nib
from dipy.core.geometry import sphere2cart
from dipy.core.geometry import cart2sphere
from somemath import axis_angle_to_euler_zyz
from somemath import euler_to_axis_angle_zyz
from scipy import interpolate
import somemath
import torch
import s2cnn


class so3():
    def __init__(self):
        """
        Class for storing the output of convolutions
        """
        self.so3=[]
        self.shellN=[]
        self.N=[]
        self.euler_max=[]
        self.axisa_max=[]
        self.so3_axa=[]
        self.signal1=[]
        self.signal2=[]
        self.axa_top=[]
        self.max_indices=[]
        self.top_average=[]

    def makeSo3(self,N,shellN):
        """
        Makes so3 grid [beta, alpha, gamma, complex] beta=[0, pi], alpha=[0,2pi]  gamma=[0,2pi]
        :param N: number of voxels
        :param shellN: number of shells
        :return:
        """
        self.so3=np.empty([N,N,N,2,shellN])
        self.shellN=shellN
        self.N=N
        self.signal1 = np.empty([N, N, shellN])
        self.signal2 = np.empty([N, N, shellN])
    # def analyse(self):
    #     for shell in range(1,shellN):
    #         so3_real=
    def makeNii(self,filename,shell):
        """
        Will save nifti file
        :param filename: filename of nifti file
        :param shell: which shell
        :return:
        """
        N=self.N
        temp=np.empty([N,N,N,2])
        temp=self.so3[:,:,:,0,shell]
        affine = np.diag([1, 1, 1, 1])
        name="%s_shell%d.nii.gz" % (filename,shell)
        print(name)
        nii=nib.Nifti1Image(temp,affine)
        dbeta=180/(N-1)
        dalpha=360/(N-1)
        dgamma=360/(N-1)
        sform=[[dbeta,0,0,0],[0,dalpha,0,0],[0,0,dgamma,0],[0,0,0,1]]
        nii.set_sform(sform)
        nib.save(nii,name)

    def top_axa(self):
        """
        Axis-angle representation of points with max overlap (all shells, real part)
        :return: axis-angles
        """
        for shell in range(0,self.shellN):
            temp=self.so3[:,:,:,0,shell]
            max_indices=np.asarray(np.where(temp==1))

            shape=max_indices.shape

            x=np.empty(shape[1])
            y=np.empty(shape[1])
            z=np.empty(shape[1])
            psi=np.empty(shape[1])
            for i in range(0,shape[1]):
                max_index=max_indices[:,i]
                beta=max_index[0]*(np.pi/(self.N-1))
                alpha=max_index[1]*(2*np.pi/(self.N-1))
                gamma=max_index[2]*(2*np.pi/(self.N-1))
                axis_angle=somemath.euler_to_axis_angle_zyz(alpha,beta,gamma)
                x[i]=axis_angle[0][0]
                y[i]=axis_angle[0][1]
                z[i]=axis_angle[0][2]
                psi[i]=axis_angle[1]
            r, theta, phi = cart2sphere(x, y, z)
            axa_top=np.row_stack((theta,phi,psi))
            self.max_indices.append(max_indices)
            self.axa_top.append(axa_top)

        #self.axa_top = np.asarray(self.axa_top)
        avg=self.axa_top[1:,2,0]
        avg=np.asarray(avg)
        avg=avg.sum()
        avg=avg/(self.shellN-1)
        self.top_average=avg

    def makeSo3_axa(self):
        """
        convert the ZYZ Euler angel in to axis-angle representation [theta,phi,psi] and move the convolution intensities
        :return:
        """
        N=self.N
        N3=N*N*N

        beta=np.linspace(0,np.pi,N)
        alpha = np.linspace(0, 2*np.pi, N)
        gamma = np.linspace(0, 2*np.pi, N)

        beta,alpha,gamma=np.meshgrid(beta,alpha,gamma,indexing='ij')

        beta=beta.flatten()
        alpha=alpha.flatten()
        gamma=gamma.flatten()
        scalar=np.empty([self.shellN,N3])

        for shell in range(0, self.shellN):
            scalar[shell,:]=self.so3[:,:,:,0,shell].flatten()

        # theta = []
        # phi=[]
        # psi=[]
        #
        # for i in range(0,N3):
        #     t,p,s = somemath.EulerZYZ2AxisAngle(alpha[i],beta[i],gamma[i])
        #     theta.append(t)
        #     phi.append(p)
        #     psi.append(s)
        #
        # theta=np.asarray(theta)
        # phi=np.asarray(phi)
        # psi=np.asarray(psi)
        #
        # theta=theta+np.pi
        # phi=phi+np.pi

        x=[]
        y=[]
        z=[]
        psi=[]

        for i in range(0,N3):
            axisangle=somemath.euler_to_axis_angle_zyz(alpha[i], beta[i], gamma[i])
            x.append(axisangle[0][0])
            y.append(axisangle[0][1])
            z.append(axisangle[0][2])
            psi.append(axisangle[1])

        x=np.asarray(x)
        y=np.asarray(y)
        z=np.asarray(z)

        r, theta, phi = cart2sphere(x, y, z)
        phi = phi + np.pi

        theta_a = np.linspace(0, np.pi, N)
        phi_a = np.linspace(0, 2 * np.pi, N)
        psi_a = np.linspace(0, np.pi, N)

        theta_a,phi_a,psi_a=np.meshgrid(theta_a,phi_a,psi_a,indexing='ij')

        theta_a=theta_a.flatten()
        phi_a = phi_a.flatten()
        psi_a = psi_a.flatten()

        sig=[]
        self.so3_axa=[]

        for shell in range(0, self.shellN):
            sig=interpolate.griddata((theta,phi,psi),scalar[shell,:],(theta_a,phi_a,psi_a),method='linear')
            sig=np.reshape(sig,(N,N,N))
            self.so3_axa.append(sig)


        # N_arr=np.linspace(0,N-1,num=N)
        # theta=np.linspace(0,np.pi,N)
        # phi = np.linspace(0, 2*np.pi, N)
        # psi = np.linspace(0, np.pi, N)
        #
        # theta_m, phi_m, psi_m = np.meshgrid(theta,phi,psi)
        #
        #
        #
        # theta_l=np.reshape(theta_m,[N3])
        # phi_l = np.reshape(phi_m, [N3])
        # psi_l = np.reshape(psi_m, [N3])
        # #field= np.empty([N3])
        # self.so3_axa=np.empty([N,N,N,2,self.shellN])
        #
        # for shell in range(0,self.shellN):
        #     temp_real = self.so3[:, :, :, 0, shell]
        #     interpolator = interpolate.RegularGridInterpolator((N_arr, N_arr, N_arr), temp_real)
        #     field = np.empty([N3])
        #     for i in range(0,N3):
        #         axis_xyz=np.asarray(sphere2cart(1,theta_l[i],phi_l[i]))
        #         euler_zyz=axis_angle_to_euler_zyz(axis_xyz,psi_l[i])
        #         alpha_i=euler_zyz[0]*(N-1)/(2*np.pi)
        #         beta_i=euler_zyz[1]*(N-1)/(np.pi)
        #         gamma_i=euler_zyz[2]*(N-1)/(2*np.pi)
        #         field[i]=interpolator([beta_i, alpha_i, gamma_i])
        #     #self.so3_axa.append(np.reshape(field,[N,N,N]))
        #     self.so3_axa[:,:,:,0,shell]=np.reshape(field,[N,N,N])


    def makeNii_axa(self,filename,shell):
        """
        Will save nifti file
        :param filename: filename of nifti file
        :param shell: which shell
        :return:
        """
        N=self.N
        temp=np.empty([N,N,N,2])
        temp=self.so3_axa[:,:,:,0,shell]
        affine = np.diag([1, 1, 1, 1])
        name="%s_shell%d.nii.gz" % (filename,shell)
        # print(name)
        nii=nib.Nifti1Image(temp,affine)
        # dbeta=180/(N-1)
        # dalpha=360/(N-1)
        # dgamma=360/(N-1)
        # sform=[[dbeta,0,0,0],[0,dalpha,0,0],[0,0,dgamma,0],[0,0,0,1]]
        # nii.set_sform(sform)
        nib.save(nii,name)


class conv():
    def __init__(self):
        self.s1=somemath.sphereSig()
        self.s2=somemath.sphereSig()
        self.so3 = so3()

    def conv(self):
        N=self.s1.N
        b = int(N / 2)
        pi = 3.14159265359


        self.so3.makeSo3(N,1)

        sss1 = np.empty([N, N, 2])
        sss1[:, :, 0] = np.real(self.s1.grid)
        sss1[:, :, 1] = np.imag(self.s1.grid)

        sss2 = np.empty([N, N, 2])
        sss2[:, :, 0] = np.real(self.s2.grid)
        sss2[:, :, 1] = np.imag(self.s2.grid)

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
        xxiftnsmall = np.empty([N, N, N, 2])
        xxiftnsmall = xxiftn[0, 0, :, :, :, :]
        xxiftnsmall = xxiftnsmall/xxiftnsmall.max()
        self.so3.so3[:, :, :, :,0] = xxiftnsmall  # [beta, alpha, gamma, complex] beta=[0, pi], alpha=[0,2pi]  gamma=[0,2pi]
