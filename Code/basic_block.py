import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np


class BasicBlock(tf.keras.layers.Layer):
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
        #print(inputs.shape)
        if self.downsample is not None:
            residual = self.residual_layer(inputs)
        else:
            residual = inputs
        return residual + self.conv_block(inputs)


class UpBasicBlock(tf.keras.layers.Layer):
    def __init__(in_shape, out_channels, stride=2):
        super(UpBasicBlock, self).__init__()
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


class UpSampleConvLayer(tf.keras.layers.Layer):
    def __init__(self, inplane, out_channels, kernel_size, stride, upsample=None):
        super(UpSampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = layers.UpSampling2D(size=upsample, interpolation="bilinear")
        # TODO: Relection Padding?
        self.conv2d = layers.Conv2D(filters=out_channels, kernel_size=kernel_size, strides=stride, padding='same')

    def call(self, inputs, training=False):
        if self.upsample:
            x = self.upsample_layer(inputs)
        out = self.conv2d(x)
        return out


class Bottleneck(tf.keras.layers.Layer):
    """ Pre-activation residual block
    Identity Mapping in Deep Residual Networks
    ref https://arxiv.org/abs/1603.05027
    """

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=layers.BatchNormalization):
        super(Bottleneck, self).__init__()
        self.expansion = 4
        self.downsample = downsample
        if self.downsample is not None:
            self.residual_layer = layers.Conv2D(planes * self.expansion, kernel_size=1, strides=stride, padding='same') #Originally kernel_size = 1
        conv_block = [norm_layer(),
                      layers.ReLU(),
                      layers.Conv2D(filters=planes, kernel_size=1, strides=1),
                      norm_layer(),
                      layers.ReLU(),
                      layers.Conv2D(filters=planes, kernel_size=3, strides=stride, padding='same'),
                      norm_layer(),
                      layers.ReLU(),
                      layers.Conv2D(planes * self.expansion, kernel_size=1, strides=1)]
        self.conv_block = models.Sequential(layers=conv_block)

    def call(self, x):
        if self.downsample is not None:
            residual = self.residual_layer(x)
        else:
            residual = x
        #print(x.shape, residual.shape, self.conv_block(x).shape)
        return residual + self.conv_block(x)


class UpBottleneck(tf.keras.layers.Layer):
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
                      layers.Conv2D(filters=planes, kernel_size=1, strides=1),
                      norm_layer(),
                      layers.ReLU(),
                      UpSampleConvLayer(inplane=planes, out_channels=planes, kernel_size=3, stride=1, upsample=stride),
                      norm_layer(),
                      layers.ReLU(),
                      layers.Conv2D(filters=planes * self.expansion, kernel_size=1, strides=1)]
        self.conv_block = models.Sequential(layers=conv_block)

    def call(self, x):
        #print(x.shape, self.residual_layer(x).shape, self.conv_block(x).shape)
        return self.residual_layer(x) + self.conv_block(x)


class Inspiration(tf.keras.layers.Layer):
    """ Inspiration Layer (from MSG-Net paper)
    tuning the featuremap with target Gram Matrix
    ref https://arxiv.org/abs/1703.06953
    """

    def __init__(self, C, B=1):
        super(Inspiration, self).__init__()

        # B is equal to 1 or input mini_batch
        # self.weight = tf.random.uniform([1, C, C], name="weight")
        init_weight = np.random.uniform(size=[1, C, C])
        self.weight = tf.Variable(init_weight, name="weight", dtype=tf.float32)


        # random_uniform_init = tf.keras.initializers.lecun_uniform(seed=None)
        # self.weight = tf.get_variable("inspiration weight", [], initializer=random_uniform_init)
        # non-parameter buffer
        init_gram = np.random.uniform(size=[B, C, C])
        self.G = tf.Variable(init_gram, "gram_matrix", dtype=tf.float32)
        self.C = C

    def set_target(self, target):
        self.G = target
        ### Bottom is original implementation
        # self.G = tf.expand_dims(target, 0)

    def call(self, X):
        X_in = tf.transpose(X, perm=[0, 3, 1, 2]) #Original paper implementation based on X with shape (b, ch, h, w), whereas our implementation has (b, h, w, ch). To avoid problem in computation, reshape to make it consistent with paper
        # print("weight: {}, self.G, X: {}".format(self.weight.shape, self.G.shape, X_in.shape))

        self.P = tf.matmul(tf.tile(self.weight, [self.G.shape[0], 1, 1]), self.G)  ########## Not fully sure pytorch code is equivalent to this line, as the matrix multiplication behavior of pytorch is not fully understanded for matrices with shapes BxCxC and BxCxC
        
        b, ch, h, w = X_in.shape
        # print("self.P: {}".format(self.P.shape))
        X_out = tf.reshape(tf.matmul(tf.tile(tf.transpose(self.P, perm=[0, 2, 1]), [b, 1, 1]),
                      tf.reshape(X_in, [b, ch, -1])), [b, ch, h, w])

        # tf.reshape(X_in, [X_in.shape[0], X.shape[1], -1])   # This reshapes X_in into (b, ch, h*w)
        return tf.transpose(X_out, perm=[0, 2, 3, 1])

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'N x ' + str(self.C) + ')'


class GramMatrix(tf.keras.layers.Layer):
    def __init__(self):
        super(GramMatrix, self).__init__()

    def call(self, y):
        b, h, w, ch = y.shape
        features = tf.reshape(y, [b, ch, w * h])
        features_t = tf.transpose(features, perm=[0, 2, 1])
        gram = tf.matmul(features, features_t) / (ch * h * w)
        return gram
    #### Bottom is the original implementation
    # def call(self, y):
    #     channels = int(y.shape[-1])
    #     a = tf.reshape(y, [-1, channels])
    #     n = tf.shape(a)[0]
    #     gram = tf.matmul(a, a, transpose_a=True)
    #     return gram / tf.cast(n, tf.float32)


class Net(tf.keras.Model):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=layers.BatchNormalization, n_blocks=6):
        super(Net, self).__init__()
        self.gram = GramMatrix()
        block = Bottleneck
        upblock = UpBottleneck
        expansion = 4
        model1 = [#layers.Conv2D(filters=64, kernel_size=7, strides=1),
                  # norm_layer(64),
                  #layers.ReLU(),
                  block(64, 32, 2, 1, norm_layer),
                  block(32 * expansion, ngf, 2, 1, norm_layer)]
        self.model1 = models.Sequential(layers=model1)

        model = []
        self.ins = Inspiration(ngf * expansion)
        model += [self.model1]
        model += [self.ins]

        for i in range(n_blocks):
            model += [block(ngf * expansion, ngf, 1, None, norm_layer)]
            
        model += [upblock(ngf * expansion, 32, 2, norm_layer),
                  upblock(32 * expansion, 16, 2, norm_layer),
                  norm_layer(),
                  layers.ReLU(),
                  layers.Conv2D(output_nc, kernel_size=7, strides=1, padding='same')]
        self.model = models.Sequential(layers=model)

    def set_target(self, Xs):
        F = self.model1(Xs)
        #print("downsample:", F.shape)
        G = self.gram(F)
        self.ins.set_target(G)

    def call(self, input):
        res = self.model(input)
        #print("upsample:", res.shape)
        return res



def Vgg(trainable=False):
    needed_layers = ['block1_conv2',
                     'block2_conv2',
                     'block3_conv3',
                     'block4_conv3']
    vgg = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet')
    # print(vgg.trainable)
    # vgg.trainable = trainable
    # print(vgg.trainable)
    for layer in vgg.layers:
      layer.trainable = False
      # print(layer, layer.trainable)
    outputs = [vgg.get_layer(name).output for name in needed_layers]
    return tf.keras.models.Model(vgg.input, outputs)
