import tensorflow as tf
import numpy as np
# import pandas as pd
import cv2
import os
import tqdm
import heapq
import datetime
import glob
import random,time

from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *
from tensorflow.keras import backend as K
from tensorflow.keras.utils import *
from tensorflow.keras.regularizers import * 

# from sklearn.preprocessing import LabelBinarizer
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# import einops

import pathlib
import itertools
# import skvideo.io

from operator import itemgetter 

import random






def activation_by_name(inputs, activation="relu", name=None):
    """Typical Activation layer added hard_swish and prelu."""
    if activation is None:
        return inputs

    layer_name = name and activation and name + activation
    activation_lower = activation.lower()
    if activation_lower == "hard_swish":
        return keras.layers.Activation(activation=hard_swish, name=layer_name)(inputs)
    elif activation_lower == "mish":
        return keras.layers.Activation(activation=mish, name=layer_name)(inputs)
    elif activation_lower == "phish":
        return keras.layers.Activation(activation=phish, name=layer_name)(inputs)
    elif activation_lower == "prelu":
        shared_axes = list(range(1, len(inputs.shape)))
        shared_axes.pop(-1 if K.image_data_format() == "channels_last" else 0)
        # print(f"{shared_axes = }")
        return keras.layers.PReLU(shared_axes=shared_axes, alpha_initializer=tf.initializers.Constant(0.25), name=layer_name)(inputs)
    elif activation_lower.startswith("gelu/app"):
        # gelu/approximate
        return tf.nn.gelu(inputs, approximate=True, name=layer_name)
    elif activation_lower.startswith("leaky_relu/"):
        # leaky_relu with alpha parameter
        alpha = float(activation_lower.split("/")[-1])
        return keras.layers.LeakyReLU(alpha=alpha, name=layer_name)(inputs)
    elif activation_lower == ("hard_sigmoid_torch"):
        return keras.layers.Activation(activation=hard_sigmoid_torch, name=layer_name)(inputs)
    elif activation_lower == ("squaredrelu") or activation_lower == ("squared_relu"):
        return (tf.nn.relu(inputs) ** 2)  # Squared ReLU: https://arxiv.org/abs/2109.08668
 
    else:
        return keras.layers.Activation(activation=activation, name=layer_name)(inputs)
    
    
    
class ChannelAffine(keras.layers.Layer):
    def __init__(self, use_bias=True, weight_init_value=1, axis=-1, **kwargs):
        super(ChannelAffine, self).__init__(**kwargs)
        self.use_bias, self.weight_init_value, self.axis = use_bias, weight_init_value, axis
        self.ww_init = keras.initializers.Constant(weight_init_value) if weight_init_value != 1 else "ones"
        self.bb_init = "zeros"
        self.supports_masking = False

    def build(self, input_shape):
        if self.axis == -1 or self.axis == len(input_shape) - 1:
            ww_shape = (input_shape[-1],)
        else:
            ww_shape = [1] * len(input_shape)
            axis = self.axis if isinstance(self.axis, (list, tuple)) else [self.axis]
            for ii in axis:
                ww_shape[ii] = input_shape[ii]
            ww_shape = ww_shape[1:]  # Exclude batch dimension

        self.ww = self.add_weight(name="weight", shape=ww_shape, initializer=self.ww_init, trainable=True)
        if self.use_bias:
            self.bb = self.add_weight(name="bias", shape=ww_shape, initializer=self.bb_init, trainable=True)
        super(ChannelAffine, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return inputs * self.ww + self.bb if self.use_bias else inputs * self.ww

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(ChannelAffine, self).get_config()
        config.update({"use_bias": self.use_bias, "weight_init_value": self.weight_init_value, "axis": self.axis})
        return config
    
    
    
BATCH_NORM_EPSILON = 1e-5
TF_BATCH_NORM_EPSILON = 0.001
LAYER_NORM_EPSILON = 1e-5

def conv2d_no_bias(inputs, filters, kernel_size=1, strides=1, padding="VALID", use_bias=False, groups=1, use_torch_padding=True, name=None, **kwargs):
    """Typical Conv2D with `use_bias` default as `False` and fixed padding"""
    pad = (kernel_size[0] // 2, kernel_size[1] // 2) if isinstance(kernel_size, (list, tuple)) else (kernel_size // 2, kernel_size // 2)
    if use_torch_padding and padding.upper() == "SAME" and max(pad) != 0:
        inputs = keras.layers.ZeroPadding2D(padding=pad, name=name and name + "pad")(inputs)
        padding = "VALID"

    groups = max(1, groups)
    return keras.layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        groups=groups,
        #kernel_initializer=CONV_KERNEL_INITIALIZER,
        name=name and name + "conv",
        **kwargs,
    )(inputs)


def depthwise_conv2d_no_bias(inputs, kernel_size, strides=1, padding="VALID", use_bias=False, use_torch_padding=True, name=None, **kwargs):
    """Typical DepthwiseConv2D with `use_bias` default as `False` and fixed padding"""
    pad = (kernel_size[0] // 2, kernel_size[1] // 2) if isinstance(kernel_size, (list, tuple)) else (kernel_size // 2, kernel_size // 2)
    if use_torch_padding and padding.upper() == "SAME" and max(pad) != 0:
        inputs = keras.layers.ZeroPadding2D(padding=pad, name=name and name + "dw_pad")(inputs)
        padding = "VALID"
    return keras.layers.DepthwiseConv2D(
        kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        #kernel_initializer=CONV_KERNEL_INITIALIZER,
        name=name and name + "dw_conv",
        **kwargs,
    )(inputs)



def drop_block(inputs, drop_rate=0, name=None):
    """Stochastic Depth block by Dropout, arxiv: https://arxiv.org/abs/1603.09382"""
    if drop_rate > 0:
        noise_shape = [None] + [1] * (len(inputs.shape) - 1)  # [None, 1, 1, 1]
        return keras.layers.Dropout(drop_rate, noise_shape=noise_shape, name=name and name + "drop")(inputs)
    else:
        return inputs
    
    
    
def layer_norm(inputs, zero_gamma=False, epsilon=LAYER_NORM_EPSILON, center=True, name=None):
    """Typical LayerNormalization with epsilon=1e-5"""
    norm_axis = -1 if K.image_data_format() == "channels_last" else 1
    gamma_init = tf.zeros_initializer() if zero_gamma else tf.ones_initializer()
    return keras.layers.LayerNormalization(axis=norm_axis, epsilon=epsilon, gamma_initializer=gamma_init, 
                                           center=center, name=name and name + "ln")(inputs)  


class HeadInitializer(tf.initializers.Initializer):
    def __init__(self, stddev=0.02, scale=0.001, **kwargs):
        super().__init__(**kwargs)
        self.stddev, self.scale = stddev, scale

    def __call__(self, shape, dtype="float32"):
        return tf.initializers.TruncatedNormal(stddev=self.stddev)(shape, dtype=dtype) * self.scale

    def get_config(self):
        base_config = super().get_config()
        base_config.update({"stddev": self.stddev, "scale": self.scale})
        return base_config

    
def add_pre_post_process(model, rescale_mode="tf", input_shape=None, post_process=None):
    input_shape = model.input_shape[1:-1] if input_shape is None else input_shape
    model.preprocess_input = PreprocessInput(input_shape, rescale_mode=rescale_mode)
    model.decode_predictions = imagenet_decode_predictions if post_process is None else post_process
    model.rescale_mode = rescale_mode
  


def global_response_normalize(inputs, axis="auto", name=None):
    axis = (-1 if backend.image_data_format() == "channels_last" else 1) if axis == "auto" else axis
    num_dims = len(inputs.shape)
    axis = (num_dims + axis) if axis < 0 else axis
    if backend.backend() == "torch":
        nn = functional.norm(inputs, axis=[ii for ii in range(1, num_dims) if ii != axis], keepdims=True)
    else:
        # An ugly work around for `tf.norm` run into `loss=nan`
        # nn = functional.norm(inputs, axis=[ii for ii in range(1, num_dims) if ii != axis], keepdims=True)
        norm_scale = functional.cast(functional.shape(inputs)[1] * functional.shape(inputs)[2], inputs.dtype) ** 0.5
        # nn = functional.reduce_sum(functional.square(inputs / norm_scale), axis=[ii for ii in range(1, num_dims) if ii != axis], keepdims=True)
        nn = functional.reduce_mean(functional.square(inputs), axis=[ii for ii in range(1, num_dims) if ii != axis], keepdims=True)
        nn = functional.sqrt(nn) * norm_scale
    nn = nn / (functional.reduce_mean(nn, axis=axis, keepdims=True) + 1e-6)
    nn = ChannelAffine(use_bias=True, weight_init_value=0, axis=axis, name=name and name + "gamma")(inputs * nn)
    return nn + inputs


def add_with_layer_scale_and_drop_block(short, deep, layer_scale=0, residual_scale=0, drop_rate=0, name=""):
    """Just simplify calling, perform `out = short + drop_block(layer_scale(deep))`"""
    short = ChannelAffine(use_bias=False, weight_init_value=residual_scale, name=name + "res_gamma")(short) if residual_scale > 0 else short
    deep = ChannelAffine(use_bias=False, weight_init_value=layer_scale, name=name + "gamma")(deep) if layer_scale > 0 else deep
    deep = drop_block(deep, drop_rate=drop_rate, name=name)
    # print(f">>>> {short.shape = }, {deep.shape = }")
    return keras.layers.Add(name=name + "output")([short, deep])


def block(inputs, output_channel, layer_scale_init_value=1e-6, use_grn=False, drop_rate=0, activation="gelu", name=""):
    nn = depthwise_conv2d_no_bias(inputs, kernel_size=7, padding="SAME", use_bias=True, name=name)
    nn = layer_norm(nn, epsilon=LAYER_NORM_EPSILON, name=name)
    nn = keras.layers.Dense(4 * output_channel, name=name + "up_dense")(nn)
    nn = activation_by_name(nn, activation, name=name)
    if use_grn:
        nn = global_response_normalize(nn, name=name + "grn_")
    nn = keras.layers.Dense(output_channel, name=name + "down_dense")(nn)
    return add_with_layer_scale_and_drop_block(inputs, nn, layer_scale=layer_scale_init_value, drop_rate=drop_rate, name=name)



def ConvNeXt(
    num_blocks=[3, 3, 9, 3],
    out_channels=[96, 192, 384, 768],
    stem_width=-1,
    layer_scale_init_value=1e-6,  # 1e-6 for v1, 0 for v2
    use_grn=False,  # False for v1, True for v2
    head_init_scale=1.0,
    layer_norm_epsilon=1e-6,  # 1e-5 for ConvNeXtXXlarge, 1e-6 for others
    output_num_filters=-1,  # If apply additional dense + activation before output dense, <0 for not using
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="gelu",
    drop_connect_rate=0.1,
    classifier_activation="softmax",
    dropout=0,
    pretrained=None,
    model_name="convnext",
    kwargs=None,
):
    inputs = keras.layers.Input(input_shape)

    """ Stem """
    stem_width = stem_width if stem_width > 0 else out_channels[0]
    nn = conv2d_no_bias(inputs, stem_width, kernel_size=4, strides=4, padding="VALID", use_bias=True, name="stem_")
    nn = layer_norm(nn, epsilon=LAYER_NORM_EPSILON, name="stem_")

    """ Blocks """
    total_blocks = sum(num_blocks)
    global_block_id = 0
    for stack_id, (num_block, out_channel) in enumerate(zip(num_blocks, out_channels)):
        stack_name = "stack{}_".format(stack_id + 1)
        if stack_id > 0:
            nn = layer_norm(nn, epsilon=LAYER_NORM_EPSILON, name=stack_name + "downsample_")
            nn = conv2d_no_bias(nn, out_channel, kernel_size=2, strides=2, use_bias=True, name=stack_name + "downsample_")
        for block_id in range(num_block):
            block_name = stack_name + "block{}_".format(block_id + 1)
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks
            nn = block(nn, out_channel, layer_scale_init_value, use_grn, block_drop_rate, activation, name=block_name)
            global_block_id += 1

    """  Output head """
    if num_classes > 0:
        nn = keras.layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        if dropout > 0:
            nn = keras.layers.Dropout(dropout, name="head_drop")(nn)
        nn = layer_norm(nn, epsilon=LAYER_NORM_EPSILON, name="head_")
        if output_num_filters > 0:
            nn = keras.layers.Dense(output_num_filters, use_bias=True, name="head_pre_dense")(nn)
            nn = activation_by_name(nn, activation=activation, name="head_pre_")
            
        head_init = HeadInitializer(scale=head_init_scale)
        nn = keras.layers.Dense(
            num_classes, dtype="float32", activation=classifier_activation, kernel_initializer=head_init, bias_initializer=head_init, name="predictions"
        )(nn)

    model = keras.models.Model(inputs, nn, name=model_name)

    return model

# def ConvNeXt(
#     num_blocks=[3, 3, 9, 3],
#     out_channels=[96, 192, 384, 768],
#     stem_width=-1,
#     layer_scale_init_value=1e-6,  # 1e-6 for v1, 0 for v2
#     use_grn=False,  # False for v1, True for v2
#     head_init_scale=1.0,
#     layer_norm_epsilon=1e-6,  # 1e-5 for ConvNeXtXXlarge, 1e-6 for others
#     output_num_filters=-1,  # If apply additional dense + activation before output dense, <0 for not using
#     input_shape=(224, 224, 3),
#     num_classes=1000,
#     activation="gelu",
#     drop_connect_rate=0.1,
#     classifier_activation="softmax",
#     dropout=0,
#     pretrained=None,
#     model_name="convnext",
#     kwargs=None,
# ):
#     # Regard input_shape as force using original shape if len(input_shape) == 4,
#     # else assume channel dimension is the one with min value in input_shape, and put it first or last regarding image_data_format
     
#     inputs = Input(input_shape)

#     """ Stem """
#     stem_width = stem_width if stem_width > 0 else out_channels[0]
#     nn = conv2d_no_bias(inputs, stem_width, kernel_size=4, strides=4, padding="valid", use_bias=True, name="stem_")
#     nn = layer_norm(nn, epsilon=layer_norm_epsilon, name="stem_")

#     """ Blocks """
#     total_blocks = sum(num_blocks)
#     global_block_id = 0
#     for stack_id, (num_block, out_channel) in enumerate(zip(num_blocks, out_channels)):
#         stack_name = "stack{}_".format(stack_id + 1)
#         if stack_id > 0:
#             nn = layer_norm(nn, epsilon=layer_norm_epsilon, name=stack_name + "downsample_")
#             nn = conv2d_no_bias(nn, out_channel, kernel_size=2, strides=2, use_bias=True, name=stack_name + "downsample_")
#         for block_id in range(num_block):
#             block_name = stack_name + "block{}_".format(block_id + 1)
#             block_drop_rate = drop_connect_rate * global_block_id / total_blocks
#             nn = block(nn, out_channel, layer_scale_init_value, use_grn, layer_norm_epsilon, block_drop_rate, activation, name=block_name)
#             global_block_id += 1

#     """  Output head """
#     if num_classes > 0:
#         nn = layers.GlobalAveragePooling2D(name="avg_pool")(nn)
#         if dropout > 0:
#             nn = Dropout(dropout, name="head_drop")(nn)
#         nn = layer_norm(nn, epsilon=layer_norm_epsilon, name="head_")
#         if output_num_filters > 0:
#             nn = Dense(output_num_filters, use_bias=True, name="head_pre_dense")(nn)
#             nn = activation_by_name(nn, activation=activation, name="head_pre_")
#         head_init = HeadInitializer(scale=head_init_scale)
#         nn = Dense(num_classes, dtype="float32", activation=classifier_activation, kernel_initializer=head_init, name="predictions")(nn)

#     model = models.Model(inputs, nn, name=model_name)
 
#     return model


def ConvNeXtTiny(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", **kwargs):
    num_blocks = [3, 3, 9, 3]
    out_channels = [96, 192, 384, 768]
    return ConvNeXt(**locals(), model_name="convnext_tiny", **kwargs)

CONV_KERNEL_INITIALIZER = keras.initializers.VarianceScaling(scale=2.0, mode="fan_out", 
                                                             distribution="truncated_normal")



def ConvNeXtSmall(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax",   **kwargs):
    num_blocks = [3, 3, 27, 3]
    out_channels = [96, 192, 384, 768]
    return ConvNeXt(**locals(), model_name="convnext_small", **kwargs)

 


def ConvNeXtV2(
    num_blocks=[3, 3, 9, 3],
    out_channels=[96, 192, 384, 768],
    stem_width=-1,
    layer_scale_init_value=0,  # 1e-6 for v1, 0 for v2
    use_grn=True,  # False for v1, True for v2
    head_init_scale=1.0,
    layer_norm_epsilon=1e-6,
    output_num_filters=-1,
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="gelu",
    drop_connect_rate=0.1,
    classifier_activation="softmax",
    dropout=0,
    pretrained=None,
    model_name="convnext_v2",
    kwargs=None,
):
    return ConvNeXt(**locals())

 
    
    

def ConvNeXtV2Base(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", **kwargs):
    num_blocks = [3, 3, 27, 3]
    out_channels = [128, 256, 512, 1024]
    return ConvNeXtV2(**locals(), model_name="convnext_v2_base", **kwargs)


 

 
