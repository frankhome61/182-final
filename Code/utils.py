import os

import numpy as np
import tensorflow as tf



def subtract_imagenet_mean_batch(batch):
    """Subtract ImageNet mean pixel-wise from a BGR image."""
    ### Credit https://github.com/KaimingHe/deep-residual-networks/issues/5
    ###        https://forums.fast.ai/t/how-is-vgg16-mean-calculated/4577/14
    ### 	   https://www.tensorflow.org/api_docs/python/tf/reverse
    b, h, w, _ = batch.shape
    # out = tf.reverse(batch, [-1]) * 255 # swap channel from RGB to BGR
    out = batch * 255 # we don't "swap channel from RGB to BGR" anymore, because prerocess_batch function below already did so

    VGG_MEAN = tf.constant([103.939, 116.779, 123.68])
    for i in range(3):
    	VGG_MEAN = tf.expand_dims(VGG_MEAN, 0)
    return out - tf.tile(VGG_MEAN, [b, h, w, 1])

    ### Previous implementation
    # b = batch.numpy()
    # VGG_MEAN = [103.939, 116.779, 123.68]
    # out = np.copy(b) * 255
    # # print(out)
    # out = out[:, :, :, [2,1,0]] # swap channel from RGB to BGR
    # out[:, :, :, 0] -= VGG_MEAN[0]
    # out[:, :, :, 1] -= VGG_MEAN[1]
    # out[:, :, :, 2] -= VGG_MEAN[2]
    # # print(out.shape)
    # return tf.convert_to_tensor(out)

def gram_matrix(y):
    b, h, w, ch = y.shape
    features = tf.reshape(y, [b, ch, w * h])
    features_t = tf.transpose(features, perm=[0, 2, 1])
    gram = tf.matmul(features, features_t) / (ch * h * w)
    return gram
#### Bottom is the original implementation
# def gram_matrix(y):
#     print(y.shape)
#     channels = int(y.shape[-1])
#     a = tf.reshape(y, [-1, channels])
#     n = tf.shape(a)[0]
#     gram = tf.matmul(a, a, transpose_a=True)
#     print(gram.shape)
#     return gram / tf.cast(n, tf.float32)


def preprocess_batch(batch):
	""" Prepocess images so that RGB are flipped into BGR """
	# return tf.reverse(batch, [-1]) # swap channel from RGB to BGR
	return batch


	### PyTorch implementation. For reference
    # batch = batch.transpose(0, 1)    #Original input is probably (b, ch, h, w). This transpose change it into (ch, b, h, w)
    # (r, g, b) = torch.chunk(batch, 3)
    # batch = torch.cat((b, g, r))
    # batch = batch.transpose(0, 1)    #Change back to (b, ch, h, w)
    # return batch