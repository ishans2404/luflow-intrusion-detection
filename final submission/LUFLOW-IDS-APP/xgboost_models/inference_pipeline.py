
import joblib
import numpy as np
import pandas as pd
import time
import os


def load_model_assets(model_dir):
    """Load and return all serialized artefacts required for inference."""

    model = joblib.load(os.path.join(model_dir, 'optimized_xgboost_luflow.pkl'))
    encoder = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
    feature_names = joblib.load(os.path.join(model_dir, 'feature_names.pkl'))
    metadata = joblib.load(os.path.join(model_dir, 'model_metadata.pkl'))
    return {
        'model': model,
        'encoder': encoder,
        'feature_names': feature_names,
        'metadata': metadata,
        'class_names': encoder.classes_,
    }


def run_inference_with_assets(data, assets):
    """Execute inference using pre-loaded artefacts."""

    model = assets['model']
    encoder = assets['encoder']
    feature_names = assets['feature_names']

    data_features = data[feature_names]

    start_time = time.time()
    predictions = model.predict(data_features)
    probabilities = model.predict_proba(data_features)
    inference_time = time.time() - start_time

    predicted_labels = encoder.inverse_transform(predictions)

    return {
        'predictions': predictions,
        'predicted_labels': predicted_labels,
        'probabilities': probabilities,
        'class_names': encoder.classes_,
        'inference_time_seconds': inference_time,
        'samples_processed': len(data),
        'avg_time_per_sample_ms': (inference_time / len(data)) * 1000,
        'model_metadata': assets['metadata'],
    }

def inference_pipeline(data, model_dir):
    """Complete inference pipeline for network intrusion detection"""
    
    # Load model components
    model = joblib.load(os.path.join(model_dir, 'optimized_xgboost_luflow.pkl'))
    encoder = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
    feature_names = joblib.load(os.path.join(model_dir, 'feature_names.pkl'))
    metadata = joblib.load(os.path.join(model_dir, 'model_metadata.pkl'))
    
    # Ensure correct feature order
    data_features = data[feature_names]
    
    # Make predictions
    start_time = time.time()
    predictions = model.predict(data_features)
    probabilities = model.predict_proba(data_features)
    inference_time = time.time() - start_time
    
    # Convert predictions to labels
    predicted_labels = encoder.inverse_transform(predictions)
    
    return {
        'predictions': predictions,
        'predicted_labels': predicted_labels,
        'probabilities': probabilities,
        'class_names': encoder.classes_,
        'inference_time_seconds': inference_time,
        'samples_processed': len(data),
        'avg_time_per_sample_ms': (inference_time / len(data)) * 1000,
        'model_metadata': metadata
    }
