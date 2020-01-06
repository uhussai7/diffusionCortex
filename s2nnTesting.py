import torch
import s2cnn
import scipy
import numpy as np
import scipy.special as special
import matplotlib.pyplot as plt
import nibabel as nib
import s2conv


N=64
o=np.pi/2
conv=s2conv.conv()
#conv.s1.gauss(0*np.pi/2,o,0.1,N)
#conv.s2.gauss(np.pi/2,o+np.pi,0.1,N)
conv.s1.square_x(N)
conv.s2.cap_north(N)
conv.conv()
conv.so3.makeNii('test',0)
conv.so3.top_axa()
# conv.so3.makeSo3_axa()
# test=np.asarray(conv.so3.so3_axa)
# test=test[0,:,:,:]
# affine=np.diag([1,1,1,1])
# nii=nib.Nifti1Image(test,affine)
# nib.save(nii,'test.nii.gz')
#
# N=64
# b=int(N/2)
# pi=3.14159265359
# theta=np.linspace(0,pi,N)
# phi=np.linspace(0,2*pi,N)
# ph, th = np.meshgrid(phi,theta)
#
# # delta=np.empty([N,N])
# # delta[:,:]=0
# # for i in range(0,3):
# #     for j in range(0,3):
# #         print(i,j)
# #         delta[i,j]=1
# # plt.imshow(delta[:,:])
# # delta[:,:]=0
# # delta[:,b]=100
# # #delta[N-1,0]=1
#
#
# m=6
# k=5
# # for l in range(m,15):
# #     s=special.sph_harm(m-k,l,ph,th)
# #s=special.sph_harm(0,3,s2_grid_np[:,1],s2_grid_np[:,0])
# #h=plt.contourf(ph,th,s)
# #plt.show()
#
# s=np.empty([N,N])
# s[:,:]=0
# for l in range(-1,40):
#     for m in range(-l,l+1):
#         harm=special.sph_harm(m,l,ph,th)
#         s=s+harm
# #s[0,:]=100
# #s[N-1,:]=100
# #s[b,:]=100
# #s[b+1,:]=100
#
# s[:,0]=100
# s[:,b]=100
# s[0,:]=100
# s[N-1,:]=100
#
# ss=np.empty([N,N,2])
# ss[:,:,0]=np.real(s)
# ss[:,:,1]=np.imag(s)
# plt.imshow(ss[:,:,0])
#
# ss=torch.tensor(ss,dtype=torch.float)
# ssft=s2cnn.soft.s2_fft.s2_fft(ss,b_out=int(N/2.0))
# ssftn=ssft.numpy()
# print("l:",(np.argmax(ssftn[:,0])-m+k)/(l+1),l)
# print("m:",np.argmax(ssftn[:, 0])-l*(l+1), m-k)
#
# # for l in range(0,6):
# #     for m in range(-l,l+1):
# #         print(l,m,l*(l+1),l*m,l*(l+1)+m)
#
#
#
# import matplotlib.pyplot as plt
# plt.plot(ssftn[:,0])
# plt.ylabel('some numbers')
# plt.show()
#
#
# #lets try to do a convoluion
# xn=np.empty([b*b,1,1,2])
# xn[:,0,0,0]=ssftn[:,0]
# xn[:,0,0,1]=ssftn[:,1]
# x = torch.tensor(xn, dtype=torch.float)
#
# xx=s2cnn.s2_mm(x,x)
# xxift=s2cnn.so3_fft.so3_ifft(xx)
# #xxift=s2cnn.so3_fft.SO3_ifft_real.apply(xx)
# xxiftn=xxift.numpy()
# xxiftnsmall=np.empty([N,N,N])
# xxiftnsmall=xxiftn[0,0,:,:,:,0]
# affine=np.diag([1,1,1,1])
# nii=nib.Nifti1Image(xxiftnsmall,affine)
# nib.save(nii,'blah.nii.gz')
#
#
# #how to take gfft of this signal
# #
# # s2cnn.s2_ft
# #
# # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # b_in, b_out = 6, 6  # bandwidth
# # x = torch.randn(2 * b_in, 2 * b_in, 2 * b_in, dtype=torch.float, device=device)
#
#
#
#
# # pylint: disable=C,R,E1101,E1102,W0621
# '''
# Compare so3_ft with so3_fft
# '''
# # import torch
# #
# # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# #
# # b_in, b_out = 6, 6  # bandwidth
# # # random input data to be Fourier Transform
# # x = torch.randn(2 * b_in, 2 * b_in, 2 * b_in, dtype=torch.float, device=device)  # [beta, alpha, gamma]
# #
# #
# # # Fast version
# # from s2cnn.soft.so3_fft import so3_rfft
# #
# # y1 = so3_rfft(x, b_out=b_out)
# #
# #
# # # Equivalent version but using the naive version
# # from s2cnn import so3_rft, so3_soft_grid
# # import lie_learn.spaces.S3 as S3
# #
# # # so3_ft computes a non weighted Fourier transform
# # weights = torch.tensor(S3.quadrature_weights(b_in), dtype=torch.float, device=device)
# # x = torch.einsum("bac,b->bac", (x, weights))
# #
# # y2 = so3_rft(x.view(-1), b_out, so3_soft_grid(b_in))
# #
# #
# # # Compare values
# # assert (y1 - y2).abs().max().item() < 1e-4 * y1.abs().mean().item()
# #
#
#
# # do convolution with s2 function
