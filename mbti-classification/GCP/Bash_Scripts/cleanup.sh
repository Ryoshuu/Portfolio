#!/bin/bash
set -e
set -o pipefail

# check if the checkpoint file really exists in the local directory and only then continue with the clean-up
if [ ! -f "$LOCAL_CHECKPOINT_PATH" ]; then
  echo "[ERROR] No checkpoint found at $LOCAL_CHECKPOINT_PATH. Aborting clean-up."
  exit 1
fi

# delete GCS bucket
echo "Cleaning up entire GCS bucket contents..."
gcloud storage rm -r "$BUCKET_PATH"
echo "[OK] $BUCKET_NAME contents deleted."
# Check if the checkpoint exists locally before proceeding

# delete the repository
echo "Cleaning up Artifact Registry..."
gcloud artifacts repositories delete $REPO_NAME --location=$REGION --quiet || echo "[WARN] Artifact repo may not exist."
echo "[OK] Artifact Registry cleaned up."