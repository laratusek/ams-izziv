FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    nano \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    numpy \
    scipy \
    scikit-image \
    matplotlib \
    pandas \
    tqdm \
    nibabel \
    SimpleITK \
    opencv-python \
    pyyaml

CMD ["bash"]