import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

class BasicBlock(tf.keras.Model):
	def __init__(self, input_shape, out_channel, downsample=None, stride=1):
		super(BasicBlock, self).__init__()
		self.downsample = downsample
		if self.downsample is not None:
			self.residual_layer = layers.Conv2D(filters=out_channel, 
												kernel_size=1,
												strides=stride)
		conv_block = [layers.BatchNormalization(input_shape=input_shape), 
					  layers.ReLU(),
					  layers.Conv2D(filters=out_channel, 
					  				kernel_size=3,
					  				strides=stride),
					  layers.BatchNormalization(),
					  layers.ReLU(),
					  layers.Conv2D(filters=out_channel, 
					  				kernel_size=3,
					  				strides=1),
					  layers.BatchNormalization()]
		self.conv_block = models.Sequential(layers=conv_block)


	def call(self, inputs, training=False):
		print(inputs.shape)
		if self.downsample is not None: 
			residual = self.residual_layer(inputs)
		else:
			residual = inputs
		return residual + self.conv_block(inputs)



class UpBasicBlock(tf.keras.Model):
	def __init__():
		super(UpBasicBlock, self).__init__(in_shape, out_channels, stride=2)
		self.residual_layer = UpSampleConvLayer(in_shape, out_channels, kernel_size=1, stride=1, upsample=stride)
		conv_block = [layers.BatchNormalization(input_shape=in_shape), 
					  layers.ReLU(),
					  UpSampleConvLayer(in_shape, out_channels, kernel_size=3, stride=1, upsample=stride),
					  layers.ReLU(),
					  layers.Conv2D(filters=out_channel, 
						  				kernel_size=3,
						  				strides=1)]
		self.conv_block = models.Sequential(layers=conv_block)


	def call(self, inputs, training=False):
		return self.residual_layer(inputs) + self.conv_block(inputs)


class UpSampleConvLayer(tf.keras.Model):
	def __init__(self, in_shape, out_channels, kernel_size, stride, upsample=None):
		super(UpSampleConvLayer, self).__init__()
		self.upsample = upsample
		self.conv_in_shape = in_shape
		if upsample:
			self.upsample_layer = layers.UpSampling2D(size=upsample, interpolation=bilinear)
		# TODO: Relection Padding? 
		self.conv2d = layers.Conv2D(filters=out_channels, kernel_size=kernel_size, strides=stride, input_shape=self.conv_in_shape)

	def call(self, inputs, training=False):
		if self.upsample:
			x = self.upsample_layer(inputs)
		out = self.conv2d(x)
		return out


class Bottleneck(tf.keras.Model):
	""" Pre-activation residual block
	Identity Mapping in Deep Residual Networks
	ref https://arxiv.org/abs/1603.05027
	"""

	def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=layers.BatchNormalization):
		super(Bottleneck, self).__init__()
		self.expansion = 4
		self.downsample = downsample
		if self.downsample is not None:
			self.residual_layer = layers.Conv2D(filters=planes * self.expansion,
											kernel_size=1, stride=stride)
		conv_block = [norm_layer(),
					   layers.ReLU(),
					   layers.Conv2D(filters=planes, kernel_size=1, stride=1),
					   norm_layer(),
					   layers.ReLU(),
					   layers.Conv2D(filters=planes, kernel_size=3, strides=stride),
					   norm_layer(),
					   layers.ReLU(),
					   layers.Conv2D(planes, planes * self.expansion, kernel_size=1, stride=1)]
		self.conv_block = models.Sequential(layers=conv_block)

	def call(self, x):
		if self.downsample is not None:
			residual = self.residual_layer(x)
		else:
			residual = x
		return residual + self.conv_block(x)


class UpBottleneck(tf.keras.Model):
	""" Up-sample residual block (from MSG-Net paper)
    Enables passing identity all the way through the generator
    ref https://arxiv.org/abs/1703.06953
    """

	def __init__(self, inplanes, planes, stride=2, norm_layer=layers.BatchNormalization):
		super(UpBottleneck, self).__init__()
		self.expansion = 4
		self.residual_layer = UpSampleConvLayer(inplanes, planes * self.expansion,
												kernel_size=1, stride=1, upsample=stride)

		conv_block = [norm_layer(),
					  layers.ReLU(),
					  layers.Conv2D(filters=planes, kernel_size=1, stride=1),
					  norm_layer(),
					  layers.ReLU(),
					  UpSampleConvLayer(filters=planes, kernel_size=3, stride=1, upsample=stride),
					  norm_layer(),
					  layers.ReLU(),
					  layers.Conv2D(filters=planes * self.expansion, kernel_size=1, stride=1)]
		self.conv_block = models.Sequential(layers=conv_block)

	def call(self, x):
		return self.residual_layer(x) + self.conv_block(x)



class Inspiration(tf.keras.Model):
	""" Inspiration Layer (from MSG-Net paper)
    tuning the featuremap with target Gram Matrix
    ref https://arxiv.org/abs/1703.06953
    """
    def __init__(self, C, B=1):
        super(Inspiration, self).__init__()
        # B is equal to 1 or input mini_batch
        self.weight = tf.random.uniform([1,C,C], name="weight")
        # non-parameter buffer
        self.G = tf.get_variable("gram_matrix", [B,C,C])
        self.C = C

    def setTarget(self, target):
        self.G = target

    def call(self, X):
        # input X is a 3D feature map
        self.P = tf.matmul(tf.tile(self.weight, [tf.shape(self.G)[0], 1, 1]), self.G)  ########## Not fully sure pytorch code is equivalent to this line, as the matrix multiplication behavior of pytorch is not fully understanded for matrices with shapes BxCxC and BxCxC
        return tf.reshape(tf.matmul(tf.tile(tf.transpose(self.P, perm=[0,2,1]), [tf.shape(X)[0], self.C, self.C]), tf.reshape(X, [tf.shape(X)[0], tf.shape(X)[1], -1])), X.shape)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'N x ' + str(self.C) + ')'