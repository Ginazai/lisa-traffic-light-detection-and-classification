"""
Standalone TFLite Multi-Format Converter
Converts a single Keras model to multiple TFLite formats:
- INT8 (quantized)
- FLOAT16 (half precision)
- FLOAT32 (full precision)
- Dynamic Range Quantization
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os
import random
import pandas as pd
from pathlib import Path

# Configuration
KERAS_MODEL_PATH = "best_detection_model.h5"
OUTPUT_DIR = "tflite_models"
DATASET_ROOT = "LISA Traffic Light Dataset"
ANNOTATIONS_ROOT = os.path.join(DATASET_ROOT, "Annotations", "Annotations")

IMG_HEIGHT = 240
IMG_WIDTH = 320
NUM_CALIBRATION_SAMPLES = 250  # For INT8 quantization

def load_calibration_data(num_samples=NUM_CALIBRATION_SAMPLES):
    """Load sample images for INT8 calibration"""
    print(f"\n[Loading {num_samples} calibration images...]")
    
    calibration_images = []
    
    # Load from training data
    for folder in ['dayTrain', 'nightTrain']:
        folder_path = os.path.join(ANNOTATIONS_ROOT, folder)
        if not os.path.exists(folder_path):
            continue
        
        clips = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
        
        for clip in clips[:5]:  # Limit clips to speed up
            clip_path = os.path.join(folder_path, clip)
            csv_path = os.path.join(clip_path, "frameAnnotationsBULB.csv")
            
            if not os.path.exists(csv_path):
                continue
            
            df = pd.read_csv(csv_path, delimiter=';')
            filenames = df['Filename'].str.split('/').str[-1].str.split('\\').str[-1].unique()
            
            img_dir = os.path.join(DATASET_ROOT, folder, folder, clip, "frames")
            
            for filename in filenames[:20]:  # Limit images per clip
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


def convert_to_float32(model, output_path):
    """Convert to FLOAT32 TFLite (full precision, no optimization)"""
    print("\n" + "="*80)
    print("Converting to FLOAT32 TFLite (Full Precision)")
    print("="*80)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # No optimizations - keep full precision
    
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    size_kb = len(tflite_model) / 1024
    print(f"✅ FLOAT32 model saved: {output_path}")
    print(f"   Size: {size_kb:.2f} KB ({size_kb/1024:.2f} MB)")
    
    return output_path


def convert_to_float16(model, output_path):
    """Convert to FLOAT16 TFLite (half precision)"""
    print("\n" + "="*80)
    print("Converting to FLOAT16 TFLite (Half Precision)")
    print("="*80)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    size_kb = len(tflite_model) / 1024
    print(f"✅ FLOAT16 model saved: {output_path}")
    print(f"   Size: {size_kb:.2f} KB ({size_kb/1024:.2f} MB)")
    
    return output_path


def convert_to_dynamic_range_quant(model, output_path):
    """Convert to Dynamic Range Quantization (weights INT8, activations FLOAT32)"""
    print("\n" + "="*80)
    print("Converting to Dynamic Range Quantization")
    print("="*80)
    print("(Weights: INT8, Activations: FLOAT32)")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # No representative dataset needed - only weights are quantized
    
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    size_kb = len(tflite_model) / 1024
    print(f"✅ Dynamic Range model saved: {output_path}")
    print(f"   Size: {size_kb:.2f} KB ({size_kb/1024:.2f} MB)")
    
    return output_path


def convert_to_int8(model, calibration_images, output_path):
    """Convert to INT8 TFLite (full integer quantization)"""
    print("\n" + "="*80)
    print("Converting to INT8 TFLite (Full Integer Quantization)")
    print("="*80)
    
    def representative_dataset():
        """Generate representative samples for calibration"""
        for img in calibration_images:
            img_normalized = np.expand_dims(img / 255.0, axis=0).astype(np.float32)
            yield [img_normalized]
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    
    # Full integer quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    print(f"Using {len(calibration_images)} calibration samples...")
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    size_kb = len(tflite_model) / 1024
    print(f"✅ INT8 model saved: {output_path}")
    print(f"   Size: {size_kb:.2f} KB ({size_kb/1024:.2f} MB)")
    
    return output_path


def test_inference_speed(model_path, model_type, num_runs=50):
    """Test inference speed of a TFLite model"""
    import time
    
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Create dummy input
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    
    if input_dtype == np.uint8:
        # INT8 model
        dummy_input = np.random.randint(0, 256, size=input_shape, dtype=np.uint8)
    else:
        # FLOAT32/FLOAT16 model
        dummy_input = np.random.rand(*input_shape).astype(np.float32)
    
    # Warm-up
    for _ in range(5):
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    print(f"\n⚡ {model_type} Inference Speed ({num_runs} runs):")
    print(f"   Average: {avg_time:.2f} ms")
    print(f"   Std Dev: {std_time:.2f} ms")
    print(f"   Min: {min_time:.2f} ms")
    print(f"   Max: {max_time:.2f} ms")
    print(f"   FPS: {1000/avg_time:.1f}")


def compare_model_outputs(model_paths, test_image_path=None):
    """Compare outputs from different model formats"""
    print("\n" + "="*80)
    print("COMPARING MODEL OUTPUTS")
    print("="*80)
    
    # Load or create test image
    if test_image_path and os.path.exists(test_image_path):
        img = cv2.imread(test_image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        print(f"Using test image: {test_image_path}")
    else:
        img = np.random.randint(0, 256, (IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
        print("Using random test image")
    
    results = {}
    
    for name, path in model_paths.items():
        if not os.path.exists(path):
            print(f"⚠️ Model not found: {path}")
            continue
        
        interpreter = tf.lite.Interpreter(model_path=path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        input_dtype = input_details[0]['dtype']
        
        # Prepare input
        if input_dtype == np.uint8:
            input_scale, input_zero_point = input_details[0]['quantization']
            input_data = (img.astype(np.float32) / input_scale + input_zero_point).astype(np.uint8)
            input_data = np.expand_dims(input_data, axis=0)
        else:
            input_data = np.expand_dims(img / 255.0, axis=0).astype(np.float32)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        # Dequantize if needed
        if output_details[0]['dtype'] == np.uint8:
            output_scale, output_zero_point = output_details[0]['quantization']
            output = (output.astype(np.float32) - output_zero_point) * output_scale
        
        results[name] = output[0]
        
        print(f"\n{name}:")
        print(f"  Output shape: {output[0].shape}")
        print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
        print(f"  Output mean: {output.mean():.4f}")
    
    # Compare outputs
    if len(results) >= 2:
        print("\n📊 Output Differences (MSE):")
        model_names = list(results.keys())
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                name1, name2 = model_names[i], model_names[j]
                mse = np.mean((results[name1] - results[name2])**2)
                print(f"  {name1} vs {name2}: {mse:.6f}")


def main():
    print("="*80)
    print("MULTI-FORMAT TFLITE CONVERTER")
    print("="*80)
    
    # Check if Keras model exists
    if not os.path.exists(KERAS_MODEL_PATH):
        print(f"❌ Keras model not found: {KERAS_MODEL_PATH}")
        print("Please train a model first!")
        return
    
    # Load Keras model
    print(f"\n[Loading Keras model: {KERAS_MODEL_PATH}]")
    model = keras.models.load_model(KERAS_MODEL_PATH, compile=False)
    print("✅ Keras model loaded successfully")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"✅ Output directory: {OUTPUT_DIR}")
    
    # Load calibration data for INT8
    calibration_images = load_calibration_data(NUM_CALIBRATION_SAMPLES)
    
    # Convert to all formats
    model_paths = {}
    
    # 1. FLOAT32 (baseline)
    float32_path = os.path.join(OUTPUT_DIR, "model_float32.tflite")
    convert_to_float32(model, float32_path)
    model_paths['FLOAT32'] = float32_path
    
    # 2. FLOAT16
    float16_path = os.path.join(OUTPUT_DIR, "model_float16.tflite")
    convert_to_float16(model, float16_path)
    model_paths['FLOAT16'] = float16_path
    
    # 3. Dynamic Range Quantization
    dynamic_path = os.path.join(OUTPUT_DIR, "model_dynamic_range.tflite")
    convert_to_dynamic_range_quant(model, dynamic_path)
    model_paths['DYNAMIC'] = dynamic_path
    
    # 4. INT8
    int8_path = os.path.join(OUTPUT_DIR, "model_int8.tflite")
    convert_to_int8(model, calibration_images, int8_path)
    model_paths['INT8'] = int8_path
    
    # Summary
    print("\n" + "="*80)
    print("CONVERSION SUMMARY")
    print("="*80)
    
    for name, path in model_paths.items():
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024
            print(f"✅ {name:12s}: {size:8.2f} KB ({path})")
    
    # Calculate size reductions
    if os.path.exists(float32_path):
        float32_size = os.path.getsize(float32_path)
        print(f"\n📊 Size Reductions (vs FLOAT32):")
        
        for name, path in model_paths.items():
            if name != 'FLOAT32' and os.path.exists(path):
                size = os.path.getsize(path)
                reduction = (1 - size/float32_size) * 100
                print(f"  {name:12s}: {reduction:5.1f}% smaller")
    
    # Benchmark inference speed
    print("\n" + "="*80)
    print("INFERENCE SPEED BENCHMARK")
    print("="*80)
    
    for name, path in model_paths.items():
        if os.path.exists(path):
            test_inference_speed(path, name, num_runs=50)
    
    # Compare outputs
    compare_model_outputs(model_paths)
    
    print("\n" + "="*80)
    print("✅ CONVERSION COMPLETE!")
    print("="*80)
    print(f"\nAll models saved to: {OUTPUT_DIR}/")
    print("\nRecommendations:")
    print("  • FLOAT32: Best accuracy, largest size, slowest")
    print("  • FLOAT16: Good accuracy, ~50% size reduction, faster")
    print("  • DYNAMIC: Good accuracy, ~75% size reduction, much faster")
    print("  • INT8: Slight accuracy loss, ~75% size reduction, fastest")
    print("\nFor embedded devices (ESP32-S3, etc.): Use INT8")
    print("For mobile devices: Use FLOAT16 or DYNAMIC")
    print("For desktop/server: Use FLOAT32 or FLOAT16")


if __name__ == "__main__":
    main()
