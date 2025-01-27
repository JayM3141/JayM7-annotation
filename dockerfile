# Base image
FROM python:3.10

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    ffmpeg \
    libsm6 \
    libxext6 \
    cmake \
    rsync \
    libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/* && \
    git lfs install

# Upgrade pip and install Hugging Face dependencies
RUN pip install --no-cache-dir pip -U && \
    pip install --no-cache-dir \
        datasets \
        "huggingface-hub>=0.19" \
        "hf-transfer>=0.1.4" \
        "protobuf<4" \
        "click<8.1" \
        "pydantic~=1.0"

# Install Node.js (if required for your app)
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Install PyTorch and related libraries
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==1.12.0+cpu \
    torchvision==0.13.0+cpu \
    torchaudio==0.12.0+cpu

# Copy application code and requirements
WORKDIR /home/user/app
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port and set default command
EXPOSE 7860
CMD ["python", "app.py"]  # Replace with the main entry point of your app
