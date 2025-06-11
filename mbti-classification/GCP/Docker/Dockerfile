# This image caused torch_xla / TPU runtime crashes, as no unnecessary TPU environment variables are
# defined in Training.py:
# FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-cu124.2-4.py310

# This one is lightweight, official for T4 GPUs only (no unnecessary TPU support at all):
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

# set working directory inside the container
WORKDIR /mbti

# copy the necessary local script
COPY Training.py .

# Install required packages including optional XetHub acceleration and disable cache
RUN pip install --no-cache-dir \
    pandas==2.2.3 \
    numpy==1.26.4 \
    psutil \
    transformers==4.49.0 \
    hf_xet \
    google-cloud-storage \
    tqdm==4.67.1

# Set the default command to run your training script
ENTRYPOINT ["python", "Training.py"]