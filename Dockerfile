FROM tensorflow/tensorflow:2.3.1

WORKDIR /code

RUN apt update
RUN apt install -y graphviz
RUN apt install -y python-pydot python-pydot-ng
RUN apt install -y git

RUN git clone https://github.com/google/qkeras
RUN cd qkeras && git checkout b055c684721a10cfb806aaf0f6a37757ae5406d8
RUN cd qkeras && pip3 install .
RUN rm -rf qkeras

RUN git clone https://github.com/hls-fpga-machine-learning/hls4ml.git
RUN cd hls4ml && pip3 install .
RUN rm -rf hls4ml

RUN pip3 install pyparsing torch torchvision pytest jupyterlab matplotlib
