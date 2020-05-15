import numpy as np
from tensorflow.keras import backend as K
from functools import reduce
from tensorflow import gather

def project(kernel):
    """
    Projects onto kernel basis. Fancy way of saying extract coefficients
    :param kernel: this will have shape 3 x 3 x (N_shells x N_filters) but has to be numpy
    :return: weights this will have shape 6 x (N_shells x N_filters) will be numpy
    """
    N=kernel.shape[-1]
    weights = np.empty([9,N])
    for i in range(0,N):
        weights[0,i] = kernel[1, 2, i]
        weights[1,i] = kernel[0, 2, i]
        weights[2,i] = kernel[0, 1, i]
        weights[3,i] = kernel[1, 0, i]
        weights[4,i] = kernel[2, 0, i]
        weights[5,i] = kernel[2, 1, i]
        weights[6, i] = kernel[0, 0, i]
        weights[7, i] = kernel[1, 1, i]
        weights[8, i] = kernel[2, 2, i]

    return weights


def unproject(weights):
    """
    Takes coefficients and puts it back in kernel form
    :param weights: has shape 6 x (N_shells x N_filters) but has to be numpy
    :return: kernel 3 x 3 x (N_shells x N_filters) but has to be numpy
    """

    N=weights.shape[-1]
    kernel = np.empty([3,3,N])
    for i in range(0,N):
        kernel[1, 2,i] = weights[0,i]
        kernel[0, 2,i] = weights[1,i]
        kernel[0, 1,i] = weights[2,i]
        kernel[1, 0,i] = weights[3,i]
        kernel[2, 0,i] = weights[4,i]
        kernel[2, 1,i] = weights[5,i]
        kernel[0, 0,i] = weights[6, i]
        kernel[1, 1,i] = weights[7, i]
        kernel[2, 2,i] = weights[8, i]
    return kernel

def rotate(kernel, angle):
    """
    Counter clockwise rotation
    :param kernel: tensor for kernel
    :param angle: angle is represented as an integer
    :return: rotated tensor kernel
    """
    if angle is None:
        angle = 1
    if int(angle) is False:
        raise ValueError("angles need to be ints")
    else:
        angle = angle % 6

    shape=kernel.shape
    Ns=kernel.shape[2:]
    N=reduce(lambda x,y:x*y,Ns)
    #N1 = kernel.shape[-2]
    #N2 = kernel.shape[-1]
    #N = N1 * N2
    kernel = K.reshape(kernel,[3,3,N])
    weights = project(kernel.numpy())
    for i in range(0,N):
        weights[0:6,i] = np.roll(weights[0:6,i], angle)
    kernel=K.variable(unproject(weights))
    return K.reshape(kernel,shape)


def reflect(kernel, axis):
    """
    Reflects along an axis.
    :param kernel: tensor for kernel
    :param axis: axis of reflection, represented as integer 0 is x-axis, 1 is middle of first edge, 2 is first vertex etc..
    :return:
    """
    if axis is None:
        axis = 0
    if int(axis) is False:
        raise ValueError("axis need to be ints")
    else:
        axis = axis % 6

    # first reflect on x-axis and then rotate
    shape = kernel.shape
    Ns = kernel.shape[2:]
    N = reduce(lambda x, y: x * y, Ns)
    #N1 = kernel.shape[-2]
    #N2 = kernel.shape[-1]
    #N = N1 * N2
    kernel=K.reshape(kernel, [3, 3, N])
    weights = project(kernel.numpy())
    for i in range(0,N):
        temp_weights = np.copy(weights[1:3,i])
        weights[1:3,i] = np.roll(weights[4:6,i], 1)
        weights[4:6,i] = np.roll(temp_weights, 1)
    kernel = K.variable(unproject(weights))
    return rotate(K.reshape(kernel, shape),axis)

    #



def expand_scalar(kernel):
    kernels = []
    for angle in range(0,6):
        kernels.append(rotate(kernel,angle))
    for axis in range (0,6):
        kernels.append(reflect(kernel,axis))
    return K.concatenate(kernels)

def rotate_deep(kernel,phi):
    #kernel has size [3,3,12,in,out] where the 12 is for rotations/reflections
    #first 6 indices are for rotations and other six are for reflections
    #we need to shift these indices i.e. theta --> theta - phi


    idx_rot=np.arange(6)
    idx_rot = (idx_rot - phi)%6
    idx_ref = idx_rot + 6
    idx = np.concatenate((idx_rot,idx_ref))
    kernel=kernel.numpy()
    kernel=kernel[:,:,idx,:,:]
    return rotate(K.variable(kernel),phi)

def reflect_deep(kernel, phi):
    # kernel has size [3,3,12,in,out] where the 12 is for rotations/reflections
    # first 6 indices are for rotations and other six are for reflections

    idx_rot = np.arange(6)
    idx_rot = (phi-idx_rot) % 6
    idx_ref = idx_rot + 6
    idx = np.concatenate((idx_ref, idx_rot))
    kernel = kernel[:, :, idx, :, :]
    return reflect(kernel, phi)

    #since this is a deep layer and tensor kernel should have
    #shape [3,3,12,in,out]
    #we need a phi that will run in range(0,12) 0 to 5 for rot rest refl

    #rotate_deep(kernel,phi)
    #reflect_deep(kernel,phi)



#def gpad(input,deep):



def conv2d(input,kernel,deep):
    """
    Takes group convolutions where the group is the dihedral group of order 12
    :param input: scalar or regular input, note that scalar layers need to be padded before hand
    :param kernel: the kernel
    :param deep: whether it is a deep layer or the first layer
    :return: the result of the convolution
    """
#     #have to differentiate between scalar or regular (thats tricky)
#     # if input has size [batch,x,y,shells] it is scalar

    #this is for scalar
    #expand the kernels
    #assume that kernel has shape [3,3,in,out] and out is the number of channels
    #in = 3
    #we have to dump the froup expansion into the channel dimension
    kernels=[]
    for angle in range(0,6):
        kernels.append(rotate(kernel,angle))
    for axis in range(0,6):
        kernels.append(reflect(kernel,axis))
    kernels=K.concatenate(kernels,-1)
    return kernels





#     # if input has size [batch, x,y,shells*12] it is regular
#     #kernel will have size [x,y,shells,filters]
