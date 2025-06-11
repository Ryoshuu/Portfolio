#!/bin/bash
set -e
set -o pipefail

# define global variables for the pipeline
export PROJECT_ID="mbti-444713"
export BUCKET_NAME="$PROJECT_ID-bucket"
export BUCKET_PATH="gs://$BUCKET_NAME"
export REGION="europe-west3"
export REPO_NAME="mbti-repo"
export IMAGE_NAME="mbti-training"
export IMAGE_URI="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:latest"

# get the directory of the current script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# get the parent directory for subscripts
export PARENT_DIR="$SCRIPT_DIR/.."

# move to the parent directory
cd "$SCRIPT_DIR"

# Log the path but show only the relevant path part
echo "Running script from $(pwd | sed "s|.*/Documents|~/Documents|")"

# make sure all subscripts are executable
chmod +x "$SCRIPT_DIR"/*.sh


# run the subscripts

#upload data
echo "=== Step 1: Data Upload to GCS ==="
./upload_data.sh || { echo "[ERROR] Data upload failed!"; exit 1; }
echo "[OK] Data upload succeeded!"

# build and push docker image
echo "=== Step 2: Building and Pushing Docker Image ==="
./build_and_push_image.sh || { echo "[ERROR] pushing docker image failed!"; exit 1; }
echo "[OK] Pushing Docker Image succeeded!"

# submit the training job and execute it
export JOB_NAME="mbti-training-job-$(date +%Y%m%d%H%M%S)"
echo "=== Step 3: Submitting Vertex AI training job and execute it ==="
./submit_training_job.sh || { echo "[ERROR] Training job failed or not finished!"; exit 1; }
echo "[OK] Training succeeded"

# download the model checkpoint
export LOCAL_CHECKPOINT_DIR="$PARENT_DIR/Checkpoints"
export LOCAL_CHECKPOINT_PATH="$LOCAL_CHECKPOINT_DIR/checkpoint-$(date +%Y%m%d-%H%M%S).pth"
echo "=== Step 4: Downloading Model Checkpoint ==="
./download_checkpoint.sh || { echo "[ERROR] downloading model checkpoint failed!"; exit 1; }
echo "[OK] Downloading model checkpoint succeeded"

# clean up the cloud project
echo "=== Step 5: Cleaning up the Cloud Project ==="
./cleanup.sh || { echo "[ERROR] cleaning up the project failed!"; exit 1; }
echo "[OK] Cleaning up the project succeeded"

echo "[OK] Pipeline finished successfully"