from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.utils import conv_utils
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras import constraints
from tensorflow.keras import activations


# class group_d6():
#     def __int__(self):
#
#     def rotate(self,kernel,angle):
#         #keep angle an integer because the rotation angle is that same
#
#
#     def reflect(self,kernel,axis):
#         #keep axis and integer, starting from 0 clockwise
#
#     def conv2d(self,input,kernel):
#         #have to differentiate between scalar or regular (thats tricky)
#         # if input has size [batch,x,y,shells] it is scalar
#         # if input has size [batch, x,y,shells*12] it is regular
#         #kernel will have size [x,y,shells,filters]



class groupConvScalar(Layer):

    def __init__(self, filters,
                 #kernel_size,
                 #strides=(1, 1),
                 #padding='valid',
                 #data_format=None,
                 #dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 deep=None,
                 **kwargs):
        if deep is None:
            raise ValueError('Please specify whether this is a zeroth layer (deep=0)'
                             'or a deep layer (deep=1). Found `None`.')
        super(groupConvScalar, self).__init__(**kwargs)
        kernel_size = 3
        rank = 2
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.deep = deep


    def build(self, input_shape):
        """
        :param input_shape: The number of channels go last
        :return:
        """

        #need to differentiate between first layer and deep layer
        channel_axis=-1 #channels always go last
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        if self.deep==0:
            input_dim=input_shape[channel_axis]
        if self.deep==1 and isinstance(input_shape[channel_axis]%12):
            input_dim=input_shape[channel_axis]/12
        else:
            raise ValueError('The number of channels in deep layer'
                             'not divisible by order of group'
                             'something is wrong')

        kernel_shape=self.kernel_size+(input_dim,self.filters)
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        trainable=True)
        else:
            self.bias = None

        super(groupConvScalar, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):

        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)