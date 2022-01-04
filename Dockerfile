# Adapted from https://aws.amazon.com/blogs/machine-learning/training-and-deploying-deep-learning-models-using-jax-with-amazon-sagemaker/
# Using a CUDA base image, even though only this project only requires a CPU, so a GPU will be available if the host supports it.
FROM nvidia/cuda:11.4.2-cudnn8-devel-ubuntu20.04

ENV PIP_NO_CACHE_DIR=1
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections # Avoid hanging on any install prompts
RUN apt update -y && apt upgrade -y
RUN apt install -y python3-pip
RUN ln -sf /usr/bin/python3 /usr/bin/python && ln -sf /usr/bin/pip3 /usr/bin/pip
RUN pip install --upgrade pip setuptools_rust
RUN pip install pyOpenSSL --upgrade
RUN ln -s /usr/lib/cuda /usr/local/cuda-11.1 # Needed for some libs? (This is copy-pasta and I haven't tested with a GPU-enabled host yet.)
# Using an older version of jax/jaxlib due to:
#   ```
#   RuntimeError: This version of jaxlib was built using AVX instructions, which your CPU and/or operating system do not support.
#   You may be able work around this issue by building jaxlib from source.
#   ```
RUN pip install --upgrade jax==0.2.6 jaxlib==0.1.57+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html

# aiortc requirements: https://github.com/aiortc/aiortc#linux
RUN apt install libavdevice-dev libavfilter-dev libopus-dev libvpx-dev pkg-config libsrtp2-dev -y
RUN apt install git -y # Needed for a package dependency specified with a git URL (`jax_spectral`)

COPY setup.py README.md /workspace/
COPY ./server /workspace/server
COPY ./jaxdsp /workspace/jaxdsp
WORKDIR /workspace/server

RUN pip install -r requirements.txt

EXPOSE 8080
CMD ["server.py"]
ENTRYPOINT ["python"]
