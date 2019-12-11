FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04

RUN apt-get update && apt-get -y install cmake libboost-all-dev && apt-get clean
RUN mkdir /muzero
RUN mkdir -p /usr/local/lib64
COPY libtorch /usr/local/lib64/libtorch

COPY CMakeLists.txt /muzero
COPY main.cpp /muzero
COPY random.cpp /muzero
COPY random.h /muzero
COPY example.txt /muzero
COPY abseil-cpp /muzero/abseil-cpp

RUN mkdir /muzero/build
RUN cd /muzero/build && cmake -DCMAKE_BUILD_TYPE=Release ../ && make