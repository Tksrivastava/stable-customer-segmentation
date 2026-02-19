#!/usr/bin/env bash

set -e

cd /stable-customer-segmentation

echo "=================================================="
echo " Retail Clustering Feature Pipeline"
echo " Started at: $(date)"
echo "=================================================="

PIPELINE=${1:-all}

run_feature_pipeline() {
    echo "[INFO] Running feature preparation pipeline..."
    python -u feature-preparation-pipeline.py
}

run_autoencoder_training() {
    echo "[INFO] Running autoencoder training pipeline..."
    python -u autoencoder-training-pipeline.py
}

run_clustering_training() {
    echo "[INFO] Running clustering training pipeline..."
    python -u clustering-training-pipeline.py
}

run_inference() {
    echo "[INFO] Running clustering inference pipeline..."
    python -u clustering-inference-pipeline.py
}

case "$PIPELINE" in
    feature)
        run_feature_pipeline
        ;;
    train)
        run_autoencoder_training
        run_clustering_training
        ;;
    infer)
        run_inference
        ;;
    all)
        run_feature_pipeline
        run_autoencoder_training
        run_clustering_training
        run_inference
        ;;
    *)
        echo "[ERROR] Unknown pipeline: $PIPELINE"
        exit 1
        ;;
esac

echo "=================================================="
echo " Pipeline completed successfully at $(date)"
echo "=================================================="