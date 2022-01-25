# Using a CUDA image as a base, even though only this project only requires a CPU, so a GPU will be available if the host supports it.
FROM nvidia/cuda:11.4.2-cudnn8-devel-ubuntu20.04

ENV PIP_NO_CACHE_DIR=1
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections # Avoid hanging on any install prompts
RUN apt update -y && apt upgrade -y
RUN apt install -y python3-pip git # git needed for a package dependency specified with a git URL (`jax_spectral`)
RUN pip install --upgrade pip setuptools_rust
RUN ln -s /usr/lib/cuda /usr/local/cuda-11.1 # Needed for some libs? (This is copy-pasta and I haven't tested with a GPU-enabled host yet.)

# aiortc requirements: https://github.com/aiortc/aiortc#linux
RUN apt install libavdevice-dev libavfilter-dev libopus-dev libvpx-dev pkg-config libsrtp2-dev -y

# Using an older version of jax/jaxlib due to:
#   `RuntimeError: This version of jaxlib was built using AVX instructions, which your CPU and/or operating system do not support.`
RUN pip install --upgrade jax==0.2.6 jaxlib==0.1.57 -f https://storage.googleapis.com/jax-releases/jax_releases.html

COPY setup.py README.md /workspace/
COPY ./server/requirements.txt /workspace/server/
COPY ./jaxdsp /workspace/jaxdsp
WORKDIR /workspace/server
RUN pip install -r requirements.txt

# Copy the server script last to avoid slow docker rebuilds every time it changes
COPY ./server/server.py /workspace/server/

# Make `print` work
ENV PYTHONUNBUFFERED=1

# 8080: HTTP (REST API), 8765: WebSocket (signaling and monitoring)
EXPOSE 8080 8765
CMD ["server.py"]
ENTRYPOINT ["python3"]
