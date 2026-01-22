FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    ca-certificates curl tar \
    python2 python2-dev python2-minimal \
  && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/pip/2.7/get-pip.py -o /tmp/get-pip.py \
 && python2 /tmp/get-pip.py "pip<21" "setuptools<45" "wheel" \
 && rm -f /tmp/get-pip.py

 RUN apt-get update && apt-get install -y \
    libjpeg-dev zlib1g-dev \
  && rm -rf /var/lib/apt/lists/*


WORKDIR /naoqi
COPY pynaoqi-python2.7-2.1.2.17-linux64.tar /naoqi/
RUN tar -xvf pynaoqi-python2.7-2.1.2.17-linux64.tar && rm pynaoqi-python2.7-2.1.2.17-linux64.tar

ENV NAOQI_SDK=/naoqi/pynaoqi-python2.7-2.1.2.17-linux64
ENV PYTHONPATH=${NAOQI_SDK}:${PYTHONPATH}
ENV LD_LIBRARY_PATH=${NAOQI_SDK}:${LD_LIBRARY_PATH}

WORKDIR /workspace