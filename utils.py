import os

import numpy as np
import hls4ml

from qkeras import *

__all__=['_accuracy','_output_shape','_layer_number','_alpha','cleanup']


# NOTE: FPGA LIST
# xc7k70tl-fbv676-2L
# xc7a12ti-csg325-1L
# xc7a12tl-csg325-2L
# xa7a35t-csg325-1I
# xc7z020-clg484-2
# xa7z010-clg225-1I
# xc7s50-fgga484-2
# xa7s6-cpga196-1I

def _accuracy(prediction1,prediction2,error):
    return (np.linalg.norm(prediction1 - prediction2)<error)

def _output_shape(prediction1,prediction2):
    return len(prediction1[0])==len(prediction2)

def _layer_number(model,hls_model):
    return len(model.layers) + 1 == len(hls_model.get_layers())

def _alpha(model,hls_model):
    return (list(hls_model.get_layers())[1].attributes['class_name'] == model.layers[1].__class__.__name__) & (list(hls_model.get_layers())[2].attributes['class_name'] == 'Alpha')

def cleanup():
    os.system('rm -rf my-hls-test')
    os.system('rm my-hls-test.tar.gz')

def vivado_hls(model,input_shape,number,error):
    input_data=np.random.randn(number,input_shape)

    hls_model = hls4ml.converters.convert_from_keras_model(model,output_dir='temp',fpga_part='xc7s50-fgga484-2',hls_config={ "default_precision":'ap_fixed<18,6>'})
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