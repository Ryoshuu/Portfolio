#!/bin/bash
set -e
set -o pipefail

# Define checkpoint paths
CHECKPOINT_GCS_PATH="$BUCKET_PATH/model-output/checkpoint.pth"

# Make sure the local directory exists ($LOCAL_CHECKPOINT_DIR defined in pipeline.sh)
mkdir -p "$LOCAL_CHECKPOINT_DIR"

# download the model checkpoint ($LOCAL_CHECKPOINT_PATH defined in pipeline.sh)
# suppress the unnecessary warning about multiprocessing as it is not relevant in this case
echo "Downloading model checkpoint from GCS..."
gsutil cp "$CHECKPOINT_GCS_PATH" "$LOCAL_CHECKPOINT_PATH" 2>&1 | grep -v "If you experience problems with multiprocessing"

# hide the full path here for privacy reasons
echo "[OK] Model checkpoint downloaded to ${LOCAL_CHECKPOINT_DIR/$HOME/~}"