FROM dustynv/pytorch:2.1-r36.2.0
ENV DEBIAN_FRONTEND = noninteractive


RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    vim
                   
RUN pip install opencv-python

