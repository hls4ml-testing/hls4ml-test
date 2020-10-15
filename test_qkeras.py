import pytest
import random
import string
import numpy as np
import hls4ml
from qkeras import *
from tensorflow.keras.layers import Input

from utils import *

# ternary_tanh
qactivation_list = ['quantized_relu', 'quantized_tanh', 'binary_tanh', 'quantized_bits']
qactivation_stochastic_kernel = ['stochastic_ternary', 'stochastic_binary']
qactivation_stochastic_bias = ['ternary', 'binary']
quantized_bit_list = ['2', '3', '4', '5', '6', '7', '8']
quantized_integer_list = ['0', '1']


# Acceptable accuracy error
percent_error = 65

@pytest.mark.parametrize('activation_int', quantized_integer_list)
@pytest.mark.parametrize('activation_bit', quantized_bit_list)
def test_dense(activation_bit, activation_int):
    test_input = np.random.randn(1,10)
    x = x_in = Input(10)
    x = QDense(
        16,
        kernel_quantizer='quantized_bits(' + activation_bit + ',' + activation_int + ',1)',
        bias_quantizer='quantized_bits(' + activation_bit + ')',
        name='Qdense',
    )(x)
    x = QActivation('quantized_relu')(x)

    model = Model(inputs=x_in, outputs=x)
    hls_model = hls4ml.converters.convert_from_keras_model(model,project_name=generate_project_name())
    hls_model.compile()

    model_pred=model.predict(test_input)
    hls_pred=hls_model.predict(test_input)

    assert(_output_shape(model_pred,hls_pred))
    assert(_accuracy(model_pred,hls_pred,percent_error))
    assert(_layer_number(model,hls_model))
    assert(_alpha(model,hls_model))


@pytest.mark.parametrize('activation_kernel', qactivation_stochastic_kernel)
@pytest.mark.parametrize('activation_bias', qactivation_stochastic_bias)
def test_dense_stochastic(activation_kernel, activation_bias):
    test_input=np.random.randn(1,10)

    x = x_in = Input(10)
    x = QDense(10, kernel_quantizer=activation_kernel, bias_quantizer=activation_bias, name='Qdense',)(x)
    x = QActivation('quantized_relu')(x)

    model = Model(inputs=x_in, outputs=x)
    hls_model = hls4ml.converters.convert_from_keras_model(model,project_name=generate_project_name())
    hls_model.compile()

    model_pred=model.predict(test_input)
    hls_pred=hls_model.predict(test_input)

    assert(_output_shape(model_pred,hls_pred))
    assert(_accuracy(model_pred,hls_pred,percent_error))
    assert(_layer_number(model,hls_model))
    assert(_alpha(model,hls_model))


@pytest.mark.parametrize('activation', qactivation_list)
def test_activation(activation):
    test_input=np.random.randn(1,10)

    x = x_in = Input(10)
    x = QDense(10, kernel_quantizer='quantized_bits(3,0,1)', bias_quantizer='quantized_bits(3)', name='Qdense',)(x)
    x = QActivation(activation)(x)

    model = Model(inputs=x_in, outputs=x)
    hls_model = hls4ml.converters.convert_from_keras_model(model,project_name=generate_project_name())
    hls_model.compile()

    model_pred=model.predict(test_input)
    hls_pred=hls_model.predict(test_input)

    assert(_output_shape(model_pred,hls_pred))
    assert(_accuracy(model_pred,hls_pred,percent_error))

    if activation == 'quantized_bits':
        assert len(model.layers) == len(hls_model.get_layers())
    else:
        assert len(model.layers) + 1 == len(hls_model.get_layers())
        assert (
            list(hls_model.get_layers())[-1].attributes['activation'] == activation.split('_')[1]
            if 'quantized' in activation
            else activation
        )



@pytest.mark.parametrize('activation_kernel', qactivation_stochastic_kernel)
@pytest.mark.parametrize('activation_bias', qactivation_stochastic_bias)
def test_conv2d_stochastic(activation_kernel, activation_bias):
    x = x_in = Input((28, 28, 1))
    x = QConv2D(
        18, (3, 3), kernel_quantizer=activation_kernel, bias_quantizer=activation_bias
    )(x)
    x = QActivation('quantized_relu')(x)

    model = Model(inputs=x_in, outputs=x)
    hls_model = hls4ml.converters.convert_from_keras_model(model,project_name=generate_project_name())
    hls_model.compile()

    assert(_layer_number(model,hls_model))
    assert(_alpha(model,hls_model))


@pytest.mark.parametrize('activation_kernel', qactivation_stochastic_kernel)
@pytest.mark.parametrize('activation_bias', qactivation_stochastic_bias)
def test_conv1d_stochastic(activation_kernel, activation_bias):
    x = x_in = Input((28, 1))
    x = QConv1D(10, 3, kernel_quantizer=activation_kernel, bias_quantizer=activation_bias)(x)

    model = Model(inputs=x_in, outputs=x)
    hls_model = hls4ml.converters.convert_from_keras_model(model)
    hls_model.compile()

    assert(_layer_number(model,hls_model))
    assert(_alpha(model,hls_model))

