import diffusion
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import torch
import s2cnn
import nibabel as nib
import os
import scipy.special as special
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
import somemath
import anti_lib
import anti_lib_progs
import geodesic
from dipy.core.sphere import cart2sphere
from dipy.core.sphere import sphere2cart

from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import SmoothSphereBivariateSpline
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from numpy import save
from numpy import load

#fresh testing
dvol=diffusion.diffVolume()
dvol.getVolume("C:\\Users\\uhussain\\Documents\\ShareVM\\Cortex\\101006\\Diffusion\\Diffusion")
dvol.shells()
dti=diffusion.dti()
dti.load("C:\\Users\\uhussain\\Documents\\ShareVM\\Cortex\\101006\\Diffusion\Diffusion\\dti")
mask=dvol.mask.get_data()
i,j,k=np.where(mask ==1)

#Choose 10000 voxels for training and 1000 for testing?
Ntrain=25000
Ntest=2500
N=Ntrain+Ntest
all_inds=random.sample(range(N),N)
train_inds=all_inds[0:Ntrain]
test_inds=all_inds[Ntrain:N]

X_train=np.empty([Ntrain,6,30,3])
X_test=np.empty([Ntest,6,30,3])
Y_train=np.empty([Ntrain,10])  #V1x3, V2X3, FA, Lx3
Y_test=np.empty([Ntest,10])
t=0
for ind in train_inds:
    p = [i[ind], j[ind], k[ind]]
    for shell in range(1,4):
        X_train[t,:,:,shell-1]=dvol.makeFlatHemisphere(p,shell)
    t=t+1
t = 0
for ind in test_inds:
    p = [i[ind], j[ind], k[ind]]
    for shell in range(1, 4):
        X_test[t, :, :, shell - 1] = dvol.makeFlatHemisphere(p, shell)
    t = t + 1
t=0
for ind in train_inds:
    #p = [i[ind], j[ind], k[ind]]
    Y_train[t, 0] = dti.FA[i[ind], j[ind], k[ind]]
    Y_train[t, 1] = dti.L1[i[ind], j[ind], k[ind]]
    Y_train[t, 2] = dti.L2[i[ind], j[ind], k[ind]]
    Y_train[t, 3] = dti.L3[i[ind], j[ind], k[ind]]
    Y_train[t, 4] = dti.V1[i[ind], j[ind], k[ind],0]
    Y_train[t, 5] = dti.V1[i[ind], j[ind], k[ind],1]
    Y_train[t, 6] = dti.V1[i[ind], j[ind], k[ind],2]
    Y_train[t, 7] = dti.V2[i[ind], j[ind], k[ind],0]
    Y_train[t, 8] = dti.V2[i[ind], j[ind], k[ind],1]
    Y_train[t, 9] = dti.V2[i[ind], j[ind], k[ind],2]
    t=t+1
t=0
for ind in test_inds:
    #p = [i[ind], j[ind], k[ind]]
    Y_test[t, 0] = dti.FA[i[ind], j[ind], k[ind]]
    Y_test[t, 1] = dti.L1[i[ind], j[ind], k[ind]]
    Y_test[t, 2] = dti.L2[i[ind], j[ind], k[ind]]
    Y_test[t, 3] = dti.L3[i[ind], j[ind], k[ind]]
    Y_test[t, 4] = dti.V1[i[ind], j[ind], k[ind],0]
    Y_test[t, 5] = dti.V1[i[ind], j[ind], k[ind],1]
    Y_test[t, 6] = dti.V1[i[ind], j[ind], k[ind],2]
    Y_test[t, 7] = dti.V2[i[ind], j[ind], k[ind],0]
    Y_test[t, 8] = dti.V2[i[ind], j[ind], k[ind],1]
    Y_test[t, 9] = dti.V2[i[ind], j[ind], k[ind],2]
    t=t+1

save('X_train.npy',X_train)
save('Y_train.npy',Y_train)
save('X_test.npy',X_test)
save('Y_test.npy',Y_test)


X_train=load('X_train.npy')
Y_train=load('Y_train.npy')
X_test=load('X_test.npy')
Y_test=load('Y_test.npy')




Y_train_hold=Y_train
# #
Y_train = Y_train_hold[:,0:4]
Y_train[:,0]=10*Y_train[:,0]
Y_train[:,1:4]=1000*Y_train[:,1:4]
# #Y_train=np.empty([Ntrain,6])
# t=0
# for one_train in Y_train_hold:
#     x=  one_train[4]
#     y = one_train[5]
#     z = one_train[6]
#     throw, th1, ph1=cart2sphere(x,y,z)
#     Y_train[t,4]=th1
#     Y_train[t, 5] = ph1
#     t=t+1
#
# #cnn stuff
#
model = Sequential() #model
model.add(Conv2D(64,kernel_size=3,activation='relu', input_shape= (6,30,3)))
#model.add(Conv2D(1,kernel_size=3,activation='relu'))
#model.add(Conv2D(8,kernel_size=3,activation='relu'))
model.add(Conv2D(32,kernel_size=3,activation='relu'))
#model.add(Conv2D(2,kernel_size=3,activation='relu'))
#model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(4,activation='relu'))
#model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.compile(optimizer='adam', loss='mean_squared_logarithmic_error', metrics=['accuracy'])
model.fit(X_train,Y_train, epochs=50)
# #
pred_test=model.predict(X_test)
test=np.copy(Y_test[:,0:4])
test[:,0]=10*test[:,0]
test[:,1:4]=1000*test[:,1:4]
# #
percent_diff=pred_test-test
percent_diff=percent_diff/test
for kk in range(0,Ntest):
      for tt in range(0,4):
          percent_diff[kk,tt]=100*(pred_test[kk,tt]-test[kk,tt])/test[kk,tt]

# dvol=diffusion.diffVolume()
# #dvol.getVolume("K:\\Datasets\\HCP_diffusion\\101006\\Diffusion\\Diffusion")
# dvol.getVolume("C:\\Users\\uhussain\\Documents\\ShareVM\\Cortex\\101006\\Diffusion\\Diffusion")
# #dvol.getVolume("C:\\Users\\uhussain\\Documents\\ShareVM\\Phantom") #phantom
# # #dvol.getVolume("K:\\Datasets\\HCP_diffusion\\101006\\Diffusion\\Diffusion")
# dvol.shells()
# #test=dvol.makeFlatHemisphere([91,100,97],3)
# test=dvol.makeFlatHemisphere([47,33,4],3)
# plt.imshow(test)
# plt.show()
# dti=diffusion.dti()
# dti.load("C:\\Users\\uhussain\\Documents\\ShareVM\\Cortex\\101006\\Diffusion\Diffusion\\dti")
# # dvol.plotSignal([91,100,97],3)
# # test1=dvol.makeFlatHemisphere([47,33,1],3)

##TEst the hemisphere flatening
# iso=somemath.isomesh()
# iso.get_icomesh()
#
# theta=[]
# phi=[]
# s=[]
# x=[]
# y=[]
# z=[]
# for c in range(0,1):
#     for i in range(0,2):
#         d=0
#         for point in iso.Points[c][i]:
#             r, theta_t, phi_t = cart2sphere(point[0],point[1],point[2])
#             x.append(point[0])
#             y.append(point[1])
#             z.append(point[2])
#             theta.append(theta_t)
#             phi.append(phi_t)
#             s.append(c + i/10 + d/100)
#             theta.append(np.pi-theta_t)
#             phi.append(phi_t+np.pi)
#             x_t,y_t,z_t=sphere2cart(1,np.pi-theta_t,phi_t+np.pi)
#             x.append(x_t)
#             y.append(y_t)
#             z.append(z_t)
#             s.append(c + i / 10 + d / 100)
#             d=d+1
#
#
# th = np.asarray(theta)
# ph = np.asarray(phi)
# s = np.asarray(s)
#
# thph=np.column_stack((th,ph))
# xyz=np.column_stack((x,y,z))
# interpolator=LinearNDInterpolator(thph,s)
# #interpolator = LinearNDInterpolator(thph, s)
# #interpolator=SmoothSphereBivariateSpline(th,ph,s)
# interpolator=NearestNDInterpolator(xyz,s)
#
# iso.makeFlat(interpolator)
#
# plt.imshow(iso.s_flat)
# plt.show()
# #so3=dvol.conv([47,33,3],[47,33,4],32,4,tN=5) #orhtogonal white matter
# # #so3=dvol.conv([103,73,97],[100,73,96],32,4) #cortex superficial deep
# so3.makeNii('test',3)
# real_list=so3.so3[:,:,:,0,3]
#
# p1=[73,89,74]
# p2=[88,89,90]
# for shell in range(0, 4):
#     s1 = []
#     s2 = []
#     th = []
#     ph = []
#     i = 0
#     for ind in dvol.inds[shell]:
#         s1.append(dvol.img[p1[0], p1[1], p1[2], ind])
#         s2.append(dvol.img[p2[0], p2[1], p2[2], ind])
#         th.append(dvol.bvecs_hemi_sphere[shell][i][1])
#         ph.append(dvol.bvecs_hemi_sphere[shell][i][2])
#         i = i + 1
#     th = np.asarray(th)
#     ph = np.asarray(ph)
#     s1 = np.asarray(s1)
#     s2 = np.asarray(s2)

#plot the spherical signals using somemath.sphereSig() class
# sig=[somemath.sphereSig(),somemath.sphereSig()]
# sig[0].grid=so3.signal1[:,:,3]
# sig[1].grid=so3.signal2[:,:,3]
# sig[0].N=32
# sig[1].N=32
# sig[0].plot()
# sig[1].plot()

# #convolution testing
# # conv=s2conv.conv()
# # conv.s1.gauss(0,0,0.25,64)
# # conv.s2.gauss(-np.pi/2,0,0.25,64)
# # conv.conv()
# # conv.so3.makeNii('gauss',0)
# # conv.s1.plot()
# # conv.s2.plot()
# # # so3=dvol.conv([86,80,83],[86,80,83],32,4)
# # # #incase we want to update
# # from imp import reload
# # reload(s2conv)
# # s=s2conv.so3()
# # s.makeSo3(so3.N,so3.shellN)
# # s.so3=so3.so3
# # s.makeSo3_axa()
# # s.makeNii_axa("diff",1)




# #extract shell data at two voxels and make 2d images out of it using spherical harmonics
# img=dvol.vol.get_data()
# for shell in range(0,4):
#     print(shell)
#     s1=[]
#     s2=[]
#     th=[]
#     ph=[]
#     i=0
#     for ind in dvol.inds[shell]:
#         s1.append(img[97,54,84,ind])
#         s2.append(img[99, 54,96, ind])
#         th.append(dvol.bvecs_hemi_sphere[shell][i][1])
#         ph.append(dvol.bvecs_hemi_sphere[shell][i][2])
#         i=i+1
#     th=np.asarray(th)
#     ph=np.asarray(ph)
#     s1=np.asarray(s1)
#     s2=np.asarray(s2)
#     #make a grid
#     N=64
#     b=int(N/2)
#     pi=3.14159265359
#     theta=np.linspace(0,pi,N)
#     phi=np.linspace(0,2*pi,N)
#     ss1 = griddata((th, ph), 10000/s1, (theta[None,:], phi[:,None]), method='nearest')
#     ss2= griddata((th, ph), 10000/s2, (theta[None,:], phi[:,None]), method='nearest')
#
#
#     sss1=np.empty([N,N,2])
#     sss1[:,:,0]=np.real(ss1)
#     sss1[:,:,1]=np.imag(ss1)
#
#     sss2=np.empty([N,N,2])
#     sss2[:,:,0]=np.real(ss2)
#     sss2[:,:,1]=np.imag(ss2)
#
#     g1=torch.tensor(sss1,dtype=torch.float)
#     g1ft=s2cnn.soft.s2_fft.s2_fft(g1,b_out=b)
#     g1ftn=g1ft.numpy()
#
#     g2=torch.tensor(sss2,dtype=torch.float)
#     g2ft=s2cnn.soft.s2_fft.s2_fft(g2,b_out=b)
#     g2ftn=g2ft.numpy()
#
#     #lets try to do a convoluion
#     xn1=np.empty([b*b,1,1,2])
#     xn1[:,0,0,0]=g1ftn[:,0]
#     xn1[:,0,0,1]=g1ftn[:,1]
#     x1 = torch.tensor(xn1, dtype=torch.float)
#
#     #lets try to do a convoluion
#     xn2=np.empty([b*b,1,1,2])
#     xn2[:,0,0,0]=g2ftn[:,0]
#     xn2[:,0,0,1]=g2ftn[:,1]
#     x2 = torch.tensor(xn2, dtype=torch.float)
#
#     xx=s2cnn.s2_mm(x1,x2)
#     xxift=s2cnn.so3_fft.so3_ifft(xx)
#     #xxift=s2cnn.so3_fft.SO3_ifft_real.apply(xx)
#     xxiftn=xxift.numpy()
#     xxiftnsmall=np.empty([N,N,N])
#     xxiftnsmall=xxiftn[0,0,:,:,:,0]
#     affine=np.diag([1,1,1,1])
#     nii=nib.Nifti1Image(xxiftnsmall,affine)
#     buffer="%d.nii.gz" % shell
#     dir_path = os.path.dirname(os.path.realpath(__file__))
#     example_filename = os.path.join(dir_path)
#     nib.save(nii,example_filename) #beta alpha gamma
#
# count=0
# sig_g1=np.empty([N,N])
# sig_g1[:,:]=0
# sig_g2=sig_g1
# ph, th = np.meshgrid(phi,theta)
# flm=np.complex()
# for l in range(0,b):
#     for m in range(-l,l+1):
#         k=l * (l + 1) + m
#         fr=g1ftn[k,0]
#         fi= g1ftn[k, 1]
#         flm=complex(fr,fi)
#         sig_g1=sig_g1+flm*special.sph_harm(m,l,ph,th)
#         fr = g1ftn[k, 0]
#         fi = g1ftn[k, 1]
#         flm = complex(fr, fi)
#         sig_g1 = sig_g1 + flm * special.sph_harm(m, l, ph, th)
#         fr = g2ftn[k, 0]
#         fi = g2ftn[k, 1]
#         flm = complex(fr, fi)
#         sig_g2 = sig_g2 + flm * special.sph_harm(m, l, ph, th)
#
# plt.subplot(2,1,1)
# plt.imshow(np.real(sig_g1))
# plt.subplot(2,1,2)
# plt.imshow(np.real(sig_g2))


# x=[]
# y=[]
# z=[]
# for vec in dvol.bvecs_hemi_cart[1]:
#     x.append(vec[0])
#     y.append(vec[1])
#     z.append(vec[2])
#
# fig = go.Figure(data=[go.Scatter3d(x=xt, y=yt,mode='markers')])
# fig=px.scatter(xt,yt)
# fig.show()
#
#
# plt.scatter(xt, yt, alpha=0.5)
# plt.title('Scatter plot pythonspot.com')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()