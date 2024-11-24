# Use an official NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    tensorrt \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /Seal
COPY . /Seal

# Install torch2trt
RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt
WORKDIR torch2trt
RUN python setup.py install
WORKDIR /Seal

# Install TensorRT
RUN sudo dpkg -i nv-tensorrt.deb

# Update and install Python, pip, and other necessary dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip3 install --no-cache-dir --upgrade pip

# Copy requirements.txt into the container
COPY requirements.txt /Seal/

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install NanoOwl
RUN git clone https://github.com/NVIDIA-AI-IOT/nanoowl
WORKDIR nanoowl
RUN python3 setup.py develop --user

RUN mkdir -p data && \
    python3 -m nanoowl.build_image_encoder_engine data/owl_image_encoder_patch32.engine
WORKDIR /Seal
