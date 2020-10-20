import os

import pytest
import string
import numpy as np
from qkeras import *
from tensorflow.keras.layers import Input

import hls4ml
from utils import vivado_hls


# xc7k70tl-fbv676-2L
# xc7a12ti-csg325-1L
# xc7a12tl-csg325-2L
# xa7a35t-csg325-1I
# xc7z020-clg484-2
# xa7z010-clg225-1I
# xc7s50-fgga484-2
# xa7s6-cpga196-1I

error=15

# ternary_tanh
qactivation_list = ['quantized_relu', 'quantized_tanh', 'binary_tanh', 'quantized_bits']
qactivation_stochastic_kernel = ['stochastic_ternary', 'stochastic_binary']
qactivation_stochastic_bias = ['ternary', 'binary']
quantized_bit_list = ['2', '3', '4', '5', '6', '7', '8']
quantized_integer_list = ['0', '1']

@pytest.mark.parametrize('activation_int', quantized_integer_list)
@pytest.mark.parametrize('activation_bit', quantized_bit_list)
def test_dense(activation_bit, activation_int):
    x = x_in = Input(10)
    x = QDense(
        16,
        kernel_quantizer='quantized_bits(' + activation_bit + ',' + activation_int + ',1)',
        bias_quantizer='quantized_bits(' + activation_bit + ')',
        name='Qdense',
    )(x)
    x = QActivation('quantized_relu')(x)
    model = Model(inputs=x_in, outputs=x)

    # w=np.random.randn(10,16)
    # b=np.random.randn(16)
    # model.set_weights([w,b])

    vivado_hls(model,10,10,error)

@pytest.mark.parametrize('activation_kernel', qactivation_stochastic_kernel)
@pytest.mark.parametrize('activation_bias', qactivation_stochastic_bias)
def test_dense_stochastic(activation_kernel, activation_bias):
    test_input=np.random.randn(1,10)

    x = x_in = Input(10)
    x = QDense(10, kernel_quantizer=activation_kernel, bias_quantizer=activation_bias, name='Qdense',)(x)
    x = QActivation('quantized_relu')(x)

    model = Model(inputs=x_in, outputs=x)

    # w=np.random.randn(10,10)
    # b=np.random.randn(10)
    # model.set_weights([w,b])

    vivado_hls(model,10,10,error)

@pytest.mark.parametrize('activation', qactivation_list)
def test_activation(activation):
    test_input=np.random.randn(1,10)

    x = x_in = Input(10)
    x = QDense(10, kernel_quantizer='quantized_bits(3,0,1)', bias_quantizer='quantized_bits(3)', name='Qdense',)(x)
    x = QActivation(activation)(x)

    model = Model(inputs=x_in, outputs=x)
    vivado_hls(model,10,10,error)

