# Use a cuda enabled base image
FROM nvcr.io/nvidia/pytorch:24.02-py3
# Set the working directory in the container to /gan-interpolator
WORKDIR /gan-interpolator

# Add the current directory contents into the container at /gan-interpolator
ADD . /gan-interpolator

# Set environment variables for tzdata
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Chicago

RUN apt-get update && apt-get install -y \
    libjpeg-dev \
    libpng-dev \
    python3-opencv

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -ya

# Add conda to path
ENV PATH /opt/conda/bin:$PATH

# Update Conda
RUN conda update -n base -c defaults conda

# Use conda to install scikit-image, matplotlib, scipy, and numpy
RUN conda install -c conda-forge \
    opencv \
    scikit-image \
    matplotlib \
    scipy \
    numpy \
    cupy \
    torchmetrics \
    kornia \
    Pillow \
    tqdm
    
RUN pip3 install torch torchvision

# Make port 80 available to the world outside this container
EXPOSE 80