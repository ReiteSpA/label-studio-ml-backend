FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel
ARG DEBIAN_FRONTEND=noninteractive
ARG TEST_ENV

WORKDIR /app

RUN conda update conda -y

RUN --mount=type=cache,target="/var/cache/apt",sharing=locked \
    --mount=type=cache,target="/var/lib/apt/lists",sharing=locked \
    apt-get -y update \
    && apt-get install -y git \
    && apt-get install -y wget \
    && apt-get install -y g++ freeglut3-dev build-essential libx11-dev \
    libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev libfreeimage-dev ninja-build \
    && apt-get -y install ffmpeg libsm6 libxext6 libffi-dev python3-dev python3-pip gcc

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_CACHE_DIR=/.cache \
    PORT=9090 \
    WORKERS=2 \
    THREADS=4 \
    CUDA_HOME=/usr/local/cuda \
    GROUNDINGDINO_REPO_PATH=/GroundingDINO
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX;8.9;9.0"

# Verify CUDA_HOME is set correctly
RUN echo "CUDA_HOME is set to: $CUDA_HOME" && \
    ls -la $CUDA_HOME && \
    nvcc --version

# Install PyTorch first with the correct version
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install --no-cache-dir torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# Install other requirements (excluding PyTorch if it's in requirements.txt)
COPY requirements.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip3 install --no-cache-dir -r requirements.txt

# Install GroundingDINO (this will use the already installed PyTorch)
RUN cd / && git clone https://github.com/IDEA-Research/GroundingDINO.git
WORKDIR /GroundingDINO
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip3 install -e . --no-build-isolation

# Download GroundingDINO weights
RUN mkdir -p weights
WORKDIR /GroundingDINO/weights
RUN wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# Download SAM weights
WORKDIR /app
RUN wget -q https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
RUN wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Install test requirements if needed
COPY requirements-test.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    if [ "$TEST_ENV" = "true" ]; then \
      pip3 install --no-cache-dir -r requirements-test.txt; \
    fi

COPY . ./

CMD ["/app/start.sh"]