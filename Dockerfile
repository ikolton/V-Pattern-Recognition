# Use an official NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG TENSOR_RT_INSTALLATION_FILE="./nv-tensorrt.deb"
ARG SSH_ROOT_PASSWORD="root"

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

# Update and install Python, pip, and other necessary dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir --upgrade pip

# Set working directory
WORKDIR /Seal
COPY . /Seal

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install torch2trt
RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt
WORKDIR torch2trt
RUN python3 setup.py install
WORKDIR /Seal

COPY ${TENSOR_RT_INSTALLATION_FILE} ./nv-tensorrt.deb
RUN dpkg -i ./nv-tensorrt.deb && \
    rm ./nv-tensorrt.deb

# Install NanoOwl
RUN git clone https://github.com/NVIDIA-AI-IOT/nanoowl
WORKDIR nanoowl
RUN python3 setup.py develop --user

RUN mkdir -p data && \
    python3 -m nanoowl.build_image_encoder_engine data/owl_image_encoder_patch32.engine
WORKDIR /Seal

# ================================ SSH Configuration ================================

# Update and install SSH server
RUN apt-get update && apt-get install -y openssh-server

# Create SSH configuration directory if it doesn't exist
RUN mkdir -p /var/run/sshd

# Allow root login (not recommended for production; use a user with proper SSH keys instead)
RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config

# Set root password (replace 'rootpassword' with a secure password or configure SSH keys)
RUN echo "root:${SSH_ROOT_PASSWORD}" | chpasswd

# Expose the SSH port
EXPOSE 22

# Start the SSH service
CMD ["/usr/sbin/sshd", "-D"]
