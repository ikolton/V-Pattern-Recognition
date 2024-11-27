#!/bin/bash
set -e

echo "WARNING: do not run this script directly. It is intended to be run inside a docker container."

while [ "$resp" != 'y' ]; do
  echo "Do you wish to proceed with the installation? (y/n)"
  read -r resp
  if [ "$resp" == 'n' ]; then
    exit 0
  fi
done

# Install torch2trt
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd ./torch2trt
python3 setup.py install
cd ..

# Install TensorRT
RUN dpkg -i ./nv-tensorrt.deb && \
    rm ./nv-tensorrt.deb

# Install NanoOwl
git clone https://github.com/NVIDIA-AI-IOT/nanoowl
cd ./nanoowl
python3 setup.py develop --user

mkdir -p data && \
  python3 -m nanoowl.build_image_encoder_engine data/owl_image_encoder_patch32.engine
