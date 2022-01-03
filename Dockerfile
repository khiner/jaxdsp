# Adapted from https://aws.amazon.com/blogs/machine-learning/training-and-deploying-deep-learning-models-using-jax-with-amazon-sagemaker/
# Using a CUDA base image, even though only this project only requires a CPU, so a GPU will be available if the host supports it.
# TODO ENV PIP_NO_CACHE_DIR=1 instead of `--no-cache-dir`
#   https://stackoverflow.com/questions/45594707/what-is-pips-no-cache-dir-good-for#comment106648551_57490475
#   "in docker for python 3.6.10 or higher"
FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu18.04

# ffmpeg-4 update needed to resolve https://stackoverflow.com/a/66235550/780425
RUN apt update && apt install software-properties-common -y
RUN add-apt-repository ppa:jonathonf/ffmpeg-4
RUN apt upgrade -y
RUN apt install -y python3-pip
RUN ln -sf /usr/bin/python3 /usr/bin/python && ln -sf /usr/bin/pip3 /usr/bin/pip
RUN pip --no-cache-dir install --upgrade pip setuptools_rust
RUN ln -s /usr/lib/cuda /usr/local/cuda-11.1 # Needed for some libs? (This is copy-pasta and I haven't tested with a GPU-enabled host yet.)
RUN pip --no-cache-dir install --upgrade jax==0.2.6 jaxlib==0.1.57+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html

# aiortc requirements: https://github.com/aiortc/aiortc#linux
RUN apt install libavdevice-dev libavfilter-dev libopus-dev libvpx-dev pkg-config libsrtp2-dev -y

RUN apt install git -y # Needed for a package dependency specified with a git URL (`jax_spectral`)

COPY setup.py README.md /workspace/
COPY ./server /workspace/server
COPY ./jaxdsp /workspace/jaxdsp
WORKDIR /workspace/server

RUN pip install -r requirements.txt

EXPOSE 8080
CMD python ./server.py
