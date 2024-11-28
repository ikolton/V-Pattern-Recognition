# V-Pattern-Recognition

*Oryginalny paper* : https://arxiv.org/pdf/2312.14135
*github* : https://github.com/penghao-wu/vstar

## Docker file for NanoOWL
Before you build a docker image, you have to include the TensorRT installation file in the root directory of the project. 
You can specify the name of the file by setting the `TENSOR_RT_INSTALLATION_FILE` argument in the `docker build` command.
nv-tensorrt.deb is the name of the file in command below.

<B> IMPORTANT: </B> The dockerfile currently assumes that the original name of TensorRT installation file is: <B>nv-tensorrt-local-repo-ubuntu2204-10.6.0-cuda-11.8</B>,
if it is different, then modify the `RUN` command in the Dockerfile.

To build the docker image, run the following command:
```bash
sudo docker build --build-arg TENSOR_RT_INSTALLATION_FILE=nv-tensorrt.deb -t my-image .
```

To run the docker image, run the following command:
```bash
sudo docker run --rm --gpus all -it my-image /bin/bash
```

When you enter container, run following command:
```bash
sh install.sh
```