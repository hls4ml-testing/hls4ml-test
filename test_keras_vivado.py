import pytest
import hls4ml
import tensorflow as tf
import numpy as np
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input, Dense, Activation, Conv1D, Conv2D, \
                                    Reshape, ELU, LeakyReLU, ThresholdedReLU, \
                                    PReLU, BatchNormalization, Add, Subtract, \
                                    Multiply, Average, Maximum, Minimum, Concatenate, \
                                    MaxPooling1D, MaxPooling2D, AveragePooling1D, \
                                    AveragePooling2D
import math
import os
from tensorflow.keras import backend as K

from utils import vivado_hls

error=0.1

# ALMOST DONE
# TODO Consider BinaryDense ve TernaryDense layers
def test_dense():
    model = tf.keras.models.Sequential()
    model.add(Dense(2,
              input_shape=(1,),
              name='Dense',
              use_bias=True,
              kernel_initializer= tf.keras.initializers.RandomUniform(minval=1, maxval=10),
              bias_initializer='zeros',
              kernel_regularizer=None,
              bias_regularizer=None,
              activity_regularizer=None,
              kernel_constraint=None,
              bias_constraint=None))
    model.add(Activation(activation='elu', name='Activation'))
    model.compile(optimizer='adam', loss='mse')

    vivado_hls(model,1,10,error)

# DONE
keras_activation_functions = [LeakyReLU, ELU]
@pytest.mark.parametrize("activation_functions", keras_activation_functions)
def test_activation_leakyrelu_elu(activation_functions):
    model = tf.keras.models.Sequential()
    model.add(Dense(64,
              input_shape=(1,),
              name='Dense',
              kernel_initializer='lecun_uniform',
              kernel_regularizer=None))
    model.add(activation_functions(alpha=1.0))

    model.compile(optimizer='adam', loss='mse')

    vivado_hls(model,1,10,error)



# DONE
keras_activation_functions = [PReLU]
@pytest.mark.parametrize("activation_functions", keras_activation_functions)
def test_activation_prelu(activation_functions):
    model = tf.keras.models.Sequential()
    model.add(Dense(64,
              input_shape=(1,),
              name='Dense',
              kernel_initializer='lecun_uniform',
              kernel_regularizer=None))
    model.add(activation_functions(alpha_initializer="zeros",))

    model.compile(optimizer='adam', loss='mse')

    vivado_hls(model,1,10,error)



keras_activation_functions = [ThresholdedReLU]
@pytest.mark.parametrize("activation_functions", keras_activation_functions)
def test_activation_thresholdedrelu(activation_functions):
    model = tf.keras.models.Sequential()
    model.add(Dense(64,
              input_shape=(1,),
              name='Dense',
              kernel_initializer='lecun_uniform',
              kernel_regularizer=None))
    model.add(activation_functions(theta=1.0))

    model.compile(optimizer='adam', loss='mse')

    vivado_hls(model,1,10,error)


# DONE
keras_activation_functions = [Activation]
@pytest.mark.parametrize("activation_functions", keras_activation_functions)
def test_activation(activation_functions):
    model = tf.keras.models.Sequential()
    model.add(Dense(64,
              input_shape=(1,),
              name='Dense',
              kernel_initializer='lecun_uniform',
              kernel_regularizer=None))
    model.add(Activation(activation='relu', name='Activation'))

    model.compile(optimizer='adam', loss='mse')

    vivado_hls(model,1,10,error)
