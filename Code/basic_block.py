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




