FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=nonintercative

# upgrade env
RUN apt update && apt upgrade -y

# aiortc requirements: https://github.com/aiortc/aiortc#linux
RUN apt install python3-pip libavdevice-dev libavfilter-dev libopus-dev libvpx-dev pkg-config -y

EXPOSE 8080
COPY setup.py README.md server jaxdsp /workspace/
COPY ./server /workspace/server
COPY ./jaxdsp /workspace/jaxdsp
WORKDIR /workspace/server

RUN pip3 install -r requirements.txt

CMD python3 ./server.py
