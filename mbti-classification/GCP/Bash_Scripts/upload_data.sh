#!/bin/bash
set -e
set -o pipefail

# set datasets' local dir
DATA_DIR=$PARENT_DIR"/Datasets"

# create bucket
gcloud storage buckets create "$BUCKET_PATH" --location="$REGION" --quiet || echo "Bucket may already exist, skipping..."
# Frankfurt, Germany

# upload training and validation set to the bucket
# mask absolute path for privacy issues
gsutil cp "$DATA_DIR/train.csv" "$BUCKET_PATH/data/" 2>&1 | sed "s|$HOME|~|"
gsutil cp "$DATA_DIR/validation.csv" "$BUCKET_PATH/data/" 2>&1 | sed "s|$HOME|~|"
