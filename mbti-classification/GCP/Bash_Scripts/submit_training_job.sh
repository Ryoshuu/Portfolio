#!/bin/bash
set -e
set -o pipefail

echo "Submitting Vertex AI training job: $JOB_NAME"

#define parameters for the job
#with one GPU asthis works much faster for BERT
PARAMS="machine-type=n1-standard-4,\
replica-count=1,\
accelerator-type=NVIDIA_TESLA_T4,\
accelerator-count=1,\
container-image-uri=$IMAGE_URI"

# create a custom training job and log only successful submission
gcloud ai custom-jobs create \
  --region="$REGION" \
  --display-name="$JOB_NAME" \
  --worker-pool-spec="$PARAMS" 2>&1 | grep -E "Using endpoint|is submitted successfully"

# get the full resource name for streaming the logs (and do not log the stderr)
FULL_JOB_NAME=$(gcloud ai custom-jobs list \
  --region="$REGION" \
  --filter="displayName=$JOB_NAME" \
  --sort-by=createTime \
  --limit=1 \
  --format="value(name)" 2>/dev/null)

# Wait 5 seconds to avoid racing logs
sleep 5

# stream the logs directly and the pipeline does not move on to the next step before training is finished then
echo "Streaming logs until training job completes..."
gcloud ai custom-jobs stream-logs "$FULL_JOB_NAME" --region="$REGION" &
STREAM_PID=$!

# Poll for job state every 10 seconds as long as the job has not finished
while true; do
  JOB_STATE=$(gcloud ai custom-jobs describe "$FULL_JOB_NAME" --region="$REGION" --format='value(state)' 2>/dev/null)
  # check if the job has finished somehow, if yes kill the stream and wait till it shutdowns cleanly
  if [[ "$JOB_STATE" == "JOB_STATE_SUCCEEDED" || \
      "$JOB_STATE" == "JOB_STATE_FAILED"    || \
      "$JOB_STATE" == "JOB_STATE_CANCELLED" ]]; then
    echo "[INFO] Job has finished with state: $JOB_STATE"
    kill "$STREAM_PID" 2>/dev/null
    # Wait for the streaming logs process to exit.
    # Since we manually kill the process after the job finishes, it exits with a non-zero status — which would normally
    # break the script due to `set -e`.
    # To avoid this, we append `|| true` so the script continues gracefully.
    wait "$STREAM_PID" 2>/dev/null || true  # returns the exit state of the process id that is not 0 when killed
    break
  fi
  # wait for 10 seconds if the job has not finished before checking the job state again
  sleep 10
done

# echo "[DEBUG] Final job state was: $JOB_STATE"
