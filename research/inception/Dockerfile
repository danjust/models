from tensorflow/tensorflow:latest

WORKDIR /model
ADD . /model

RUN export PYTHONPATH='/model':$PYTHONPATH

ENTRYPOINT ["/bin/bash"]
