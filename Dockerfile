FROM nvcr.io/nvidia/l4t-cuda:10.2.460-runtime
ENV DEBIAN_FRONTEND = noninteractive
RUN apt-get update
RUN apt install git-all -y

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    python3-matplotlib \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
    
    
RUN python3 -m pip install --upgrade pip \
    && pip3 list
    
WORKDIR /workspace    
RUN git clone https://github.com/zzhawk/yet-another-vectornet.git /workspace/vectornet
RUN cd /workspace/vectornet && \
    git checkout orin && \
    git pull && \
    pip3 install -r requirements.txt
