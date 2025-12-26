"""
ESP32-Compatible TFLite Converter
Converts Keras model to TFLite format compatible with ESP32 Arduino code
Key: Uses FLOAT32 inputs/outputs with INT8 internal quantization
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os
import pandas as pd

# Configuration
KERAS_MODEL_PATH = "best_detection_model.h5"
OUTPUT_DIR = "tflite_models"
DATASET_ROOT = "LISA Traffic Light Dataset"
ANNOTATIONS_ROOT = os.path.join(DATASET_ROOT, "Annotations", "Annotations")

IMG_HEIGHT = 240
IMG_WIDTH = 320
NUM_CALIBRATION_SAMPLES = 250

def load_calibration_data(num_samples=NUM_CALIBRATION_SAMPLES):
    """Load sample images for INT8 calibration"""
    print(f"\n[Loading {num_samples} calibration images...]")
    
    calibration_images = []
    
    for folder in ['dayTrain', 'nightTrain']:
        folder_path = os.path.join(ANNOTATIONS_ROOT, folder)
        if not os.path.exists(folder_path):
            continue
        
        clips = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
        
        for clip in clips[:5]:
            clip_path = os.path.join(folder_path, clip)
            csv_path = os.path.join(clip_path, "frameAnnotationsBULB.csv")
            
            if not os.path.exists(csv_path):
                continue
            
            df = pd.read_csv(csv_path, delimiter=';')
            filenames = df['Filename'].str.split('/').str[-1].str.split('\\').str[-1].unique()
            
            img_dir = os.path.join(DATASET_ROOT, folder, folder, clip, "frames")
            
            for filename in filenames[:20]:
                img_path = os.path.join(img_dir, filename)
                
                if not os.path.exists(img_path):
                    continue
                
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                calibration_images.append(img)
                
                if len(calibration_images) >= num_samples:
                    break
            
            if len(calibration_images) >= num_samples:
                break
        
        if len(calibration_images) >= num_samples:
            break
    
    print(f"✅ Loaded {len(calibration_images)} calibration images")
    return calibration_images


def convert_to_int8_hybrid(model, calibration_images, output_path):
    """
    Convert to INT8 with FLOAT32 inputs/outputs (Hybrid Quantization)
    This matches the training preprocessing (normalized [0,1] inputs)
    and works with the ESP32 Arduino code
    """
    print("\n" + "="*80)
    print("Converting to INT8 Hybrid (FLOAT32 I/O, INT8 internals)")
    print("="*80)
    print("✅ Input: FLOAT32 [0,1] (matches training preprocessing)")
    print("✅ Output: FLOAT32 logits (Arduino applies sigmoid)")
    print("✅ Internal ops: INT8 quantized (for speed)")
    
    def representative_dataset():
        """Generate representative samples for calibration"""
        for img in calibration_images:
            # Match training preprocessing: normalize to [0, 1]
            img_normalized = np.expand_dims(img / 255.0, axis=0).astype(np.float32)
            yield [img_normalized]
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    
    # CRITICAL: Use FLOAT32 for inputs/outputs, INT8 for internal ops
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS  # Fallback for incompatible ops
    ]
    
    # KEY: Don't set inference_input_type or inference_output_type
    # This keeps them as FLOAT32 while internal ops are INT8
    
    print(f"Using {len(calibration_images)} calibration samples...")
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    size_kb = len(tflite_model) / 1024
    print(f"✅ INT8 Hybrid model saved: {output_path}")
    print(f"   Size: {size_kb:.2f} KB ({size_kb/1024:.2f} MB)")
    
    return output_path


def main():
    print("="*80)
    print("ESP32-COMPATIBLE TFLITE CONVERTER")
    print("="*80)
    
    if not os.path.exists(KERAS_MODEL_PATH):
        print(f"❌ Keras model not found: {KERAS_MODEL_PATH}")
        return
    
    print(f"\n[Loading Keras model: {KERAS_MODEL_PATH}]")
    model = keras.models.load_model(KERAS_MODEL_PATH, compile=False)
    print("✅ Keras model loaded successfully")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load calibration data
    calibration_images = load_calibration_data(NUM_CALIBRATION_SAMPLES)
    
    # Convert to INT8 Hybrid (RECOMMENDED FOR ESP32)
    output_path = os.path.join(OUTPUT_DIR, "traffic_light_detector_int8_hybrid.tflite")
    convert_to_int8_hybrid(model, calibration_images, output_path)
    
    print("\n" + "="*80)
    print("✅ CONVERSION COMPLETE!")
    print("="*80)
    print(f"\n🎯 Model: {output_path}")
    print(f"\n📝 Arduino Preprocessing (FLOAT32 input):")
    print(f"   float* input_data = input->data.f;")
    print(f"   for (int i = 0; i < input_size; i++) {{")
    print(f"       input_data[i] = resized_image_buffer[i] / 255.0f;")
    print(f"   }}")
    print(f"\n🔧 Flash to ESP32:")
    print(f"   esptool.py --chip esp32s3 --port COM9 write_flash 0x310000 {output_path}")


if __name__ == "__main__":
    main()
