FROM nvidia/cuda:11.4.3-devel-ubuntu20.04
RUN apt-get update && \
    apt-get install -y python3 python3-pip wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install matplotlib numpy pandas
