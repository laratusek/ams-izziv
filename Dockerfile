# Uporabimo zahtevano verzijo PyTorch in CUDA
#FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

# Nastavitev delovnega imenika
WORKDIR /workdir

# Namestitev sistemskih odvisnosti za OpenCV in urejanje datotek
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Nadgradnja pip in namestitev osnovnih paketov
RUN pip install --upgrade pip

# Namestitev specifičnih verzij za STUNet
RUN pip install numpy \
    nibabel \
    scipy \
    timm==0.6.12 \
    torchinfo \
    SimpleITK \
    batchgenerators==0.25 \
    nnunet-customized  # STUNet pogosto zahteva svojo verzijo nnU-Net

# 1. Kloniranje repozitorija iz slike
RUN git clone https://github.com/uni-medical/STU-Net.git /workdir/STU-Net

# 2. Premik v mapo nnUNet-2.2 in namestitev v "editable" načinu

WORKDIR /workdir/STU-Net/nnUNet-2.2

RUN pip install -e .

# 3. Nastavitev okoljskih spremenljivk za nnU-Net v2
# Pozor: v2 uporablja nekoliko drugačna imena spremenljivk kot v1!
ENV nnUNet_raw="/workdir/nnUNet_raw"
ENV nnUNet_preprocessed="/workdir/nnUNet_preprocessed"
ENV nnUNet_results="/workdir/nnUNet_results"

# Vrnitev v korensko mapo repozitorija za lažji zagon skript
WORKDIR /workdir

