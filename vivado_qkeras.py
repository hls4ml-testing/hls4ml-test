import pytest
import random
import string
import numpy as np
import hls4ml
from qkeras import *
from tensorflow.keras.layers import Input
import os


error=20

def vivado_hls(model,input_shape,number,error):
    input_data=np.random.randn(number,input_shape)

    hls_model = hls4ml.converters.convert_from_keras_model(model,output_dir='temp',hls_config={ "default_precision":'ap_fixed<18,6>'})
    hls_model.compile()
    hls_model.write()


    np.savetxt('./temp/tb_data/tb_input_features.dat',input_data)

    prediction=model.predict(input_data)
    np.savetxt('./temp/tb_data/tb_output_predictions.dat',prediction)

    os.system('cd temp && vivado_hls -f build_prj.tcl "csim=1 synth=0 cosim=0 validation=0"')

    hls_pred=t=np.loadtxt('./temp/tb_data/csim_results.log')

    print(np.linalg.norm(prediction - hls_pred))
    assert (np.linalg.norm(prediction - hls_pred) < error)

    os.system("rm -rf temp")
    os.system("rm temp.tar.gz")
    os.system('rm -rf my-hls-test')
    os.system("rm my-hls-test.tar.gz")


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