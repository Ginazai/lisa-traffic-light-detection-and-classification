"""
Standalone script to test EXISTING Keras and TFLite models
Just run this after training - no retraining needed
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os
import random
import pandas as pd

# Paths - CHANGE THESE IF NEEDED
KERAS_MODEL = "best_detection_model.h5"
TFLITE_MODEL = "tflite_models/traffic_light_detector_int8_hybrid.tflite"
DATASET_ROOT = "LISA Traffic Light Dataset"
ANNOTATIONS_ROOT = os.path.join(DATASET_ROOT, "Annotations", "Annotations")

# Config (from training)
IMG_HEIGHT = 240
IMG_WIDTH = 320
GRID_SIZE = 16
NUM_ANCHORS = 4
CONF_THRESHOLD = 0.5  # Lowered for testing
CLASS_NAMES = ['go', 'goLeft', 'stop', 'stopLeft', 'warning', 'warningLeft']

# Anchors (computed during training - using defaults for now)
ANCHORS = np.array([
    [0.003929, 0.005383],
    [0.007824, 0.011130],
    [0.012528, 0.017553],
    [0.019091, 0.026907],
])

def load_test_images(num_samples=30):
    """Load test images with annotations"""
    test_data = []
    
    for seq in ['daySequence1', 'daySequence2', 'nightSequence1', 'nightSequence2']:
        seq_path = os.path.join(ANNOTATIONS_ROOT, seq)
        if not os.path.exists(seq_path):
            continue
            
        csv_path = os.path.join(seq_path, "frameAnnotationsBULB.csv")
        if not os.path.exists(csv_path):
            continue
        
        df = pd.read_csv(csv_path, delimiter=';')
        
        grouped = {}
        for _, row in df.iterrows():
            filename = str(row['Filename']).split('/')[-1].split('\\')[-1]
            if filename not in grouped:
                grouped[filename] = []
            
            grouped[filename].append({
                'x1': int(row['Upper left corner X']),
                'y1': int(row['Upper left corner Y']),
                'x2': int(row['Lower right corner X']),
                'y2': int(row['Lower right corner Y']),
                'class': str(row['Annotation tag']).strip()
            })
        
        img_dir = os.path.join(DATASET_ROOT, seq, seq, "frames")
        
        for filename, boxes in grouped.items():
            img_path = os.path.join(img_dir, filename)
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    orig_h, orig_w = img.shape[:2]
                    test_data.append({
                        'img_path': img_path,
                        'filename': filename,
                        'boxes': boxes,
                        'orig_w': orig_w,
                        'orig_h': orig_h
                    })
    
    return random.sample(test_data, min(num_samples, len(test_data)))

def decode_predictions(predictions):
    """Decode YOLO predictions"""
    boxes = []
    
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            for a in range(NUM_ANCHORS):
                obj_conf = 1 / (1 + np.exp(-np.clip(predictions[i,j,a,0], -10, 10)))
                
                if obj_conf < CONF_THRESHOLD:
                    continue
                
                cell_x = 1 / (1 + np.exp(-np.clip(predictions[i,j,a,1], -10, 10)))
                cell_y = 1 / (1 + np.exp(-np.clip(predictions[i,j,a,2], -10, 10)))
                tw = np.clip(predictions[i,j,a,3], -10, 10)
                th = np.clip(predictions[i,j,a,4], -10, 10)
                
                box_w = ANCHORS[a][0] * np.exp(tw)
                box_h = ANCHORS[a][1] * np.exp(th)
                
                x_center = (j + cell_x) / GRID_SIZE
                y_center = (i + cell_y) / GRID_SIZE
                
                x1 = max(0, min(IMG_WIDTH, int((x_center - box_w/2) * IMG_WIDTH)))
                y1 = max(0, min(IMG_HEIGHT, int((y_center - box_h/2) * IMG_HEIGHT)))
                x2 = max(0, min(IMG_WIDTH, int((x_center + box_w/2) * IMG_WIDTH)))
                y2 = max(0, min(IMG_HEIGHT, int((y_center + box_h/2) * IMG_HEIGHT)))
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                class_probs = 1 / (1 + np.exp(-np.clip(predictions[i,j,a,5:], -10, 10)))
                class_id = np.argmax(class_probs)
                
                confidence = obj_conf * class_probs[class_id]
                boxes.append((x1, y1, x2, y2, confidence, int(class_id)))
    
    return boxes

def nms(boxes, iou_thresh=0.4):
    """Non-maximum suppression"""
    if not boxes:
        return []
    
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    keep = []
    
    while boxes:
        keep.append(boxes[0])
        boxes = [b for b in boxes[1:] if compute_iou(keep[-1], b) < iou_thresh]
    
    return keep

def compute_iou(box1, box2):
    """IoU calculation"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2-x1) * max(0, y2-y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    
    return inter / (area1 + area2 - inter) if (area1 + area2 - inter) > 0 else 0

def test_keras_model(test_data):
    """Test Keras model"""
    print("\n" + "="*80)
    print("TESTING KERAS MODEL")
    print("="*80)
    
    if not os.path.exists(KERAS_MODEL):
        print(f"❌ Keras model not found: {KERAS_MODEL}")
        return None
    
    model = keras.models.load_model(KERAS_MODEL, compile=False)
    print(f"✅ Loaded Keras model from {KERAS_MODEL}\n")
    
    results = []
    
    for idx, item in enumerate(test_data):
        img = cv2.imread(item['img_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        
        preds = model.predict(np.expand_dims(img/255.0, axis=0), verbose=0)[0]
        boxes = decode_predictions(preds)
        boxes = nms(boxes)
        
        detections = {}
        for b in boxes:
            cls = CLASS_NAMES[b[5]]
            detections[cls] = detections.get(cls, 0) + 1
        
        # More visual output
        det_str = ', '.join([f"{k}:{v}" for k,v in detections.items()]) or "❌ NONE"
        match_icon = "✅" if len(boxes) > 0 else "⚠️"
        print(f"{match_icon} {idx+1:2d}. {item['filename']:30s} │ GT:{len(item['boxes']):2d} │ Pred:{len(boxes):2d} │ {det_str}")
        
        item['img_path'] = item['img_path']  # Store for visualization
        results.append({
            'filename': item['filename'],
            'boxes': boxes,
            'gt_boxes': item['boxes'],
            'orig_w': item['orig_w'],
            'orig_h': item['orig_h'],
            'img_path': item['img_path']
        })
    
    return results

def test_tflite_model(test_data):
    """Test TFLite model"""
    print("\n" + "="*80)
    print("TESTING TFLITE MODEL")
    print("="*80)
    
    if not os.path.exists(TFLITE_MODEL):
        print(f"❌ TFLite model not found: {TFLITE_MODEL}")
        return None
    
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    is_quantized = input_details[0]['dtype'] == np.uint8
    
    if is_quantized:
        input_scale, input_zero_point = input_details[0]['quantization']
        output_scale, output_zero_point = output_details[0]['quantization']
        print(f"✅ Loaded INT8 TFLite model from {TFLITE_MODEL}\n")
    else:
        print(f"✅ Loaded FLOAT32 TFLite model from {TFLITE_MODEL}\n")
    
    results = []
    
    for idx, item in enumerate(test_data):
        img = cv2.imread(item['img_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        
        if is_quantized:
            input_data = (img.astype(np.float32) / input_scale + input_zero_point).astype(np.uint8)
            input_data = np.expand_dims(input_data, axis=0)
        else:
            input_data = np.expand_dims(img/255.0, axis=0).astype(np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])[0]
        
        if is_quantized:
            preds = (preds.astype(np.float32) - output_zero_point) * output_scale
        
        boxes = decode_predictions(preds)
        boxes = nms(boxes)
        
        detections = {}
        for b in boxes:
            cls = CLASS_NAMES[b[5]]
            detections[cls] = detections.get(cls, 0) + 1
        
        # More visual output
        det_str = ', '.join([f"{k}:{v}" for k,v in detections.items()]) or "❌ NONE"
        match_icon = "✅" if len(boxes) > 0 else "⚠️"
        print(f"{match_icon} {idx+1:2d}. {item['filename']:30s} │ GT:{len(item['boxes']):2d} │ Pred:{len(boxes):2d} │ {det_str}")
        
        results.append({
            'filename': item['filename'],
            'boxes': boxes,
            'gt_boxes': item['boxes'],
            'orig_w': item['orig_w'],
            'orig_h': item['orig_h'],
            'img_path': item['img_path']
        })
    
    return results

def calculate_simple_metrics(results):
    """Calculate basic metrics"""
    if not results:
        return
    
    total_gt = sum(len(r['gt_boxes']) for r in results)
    total_pred = sum(len(r['boxes']) for r in results)
    
    # Count matches (simple IoU > 0.25 check)
    matches = 0
    for r in results:
        for pred in r['boxes']:
            pred_box = pred[:4]
            for gt in r['gt_boxes']:
                gt_x1 = int(gt['x1'] * IMG_WIDTH / r['orig_w'])
                gt_y1 = int(gt['y1'] * IMG_HEIGHT / r['orig_h'])
                gt_x2 = int(gt['x2'] * IMG_WIDTH / r['orig_w'])
                gt_y2 = int(gt['y2'] * IMG_HEIGHT / r['orig_h'])
                gt_box = (gt_x1, gt_y1, gt_x2, gt_y2, 1.0, 0)
                if compute_iou(pred_box, gt_box) > 0.5:
                    matches += 1
                    break
    
    recall = matches / total_gt if total_gt > 0 else 0
    precision = matches / total_pred if total_pred > 0 else 0
    
    print(f"\nMETRICS:")
    print(f"  Total GT boxes: {total_gt}")
    print(f"  Total predictions: {total_pred}")
    print(f"  Matches (IoU>0.5): {matches}")
    print(f"  Recall: {recall:.1%}")
    print(f"  Precision: {precision:.1%}")
    print(f"  Avg GT per image: {total_gt/len(results):.1f}")
    print(f"  Avg predictions per image: {total_pred/len(results):.1f}")

def visualize_comparison(keras_results, tflite_results, num_samples=3):
    """Create side-by-side comparison visualization"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from datetime import datetime
    
    if not keras_results or not tflite_results:
        return
    
    # Select same samples for both
    samples = random.sample(range(len(keras_results)), min(num_samples, len(keras_results)))
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(16, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx, sample_idx in enumerate(samples):
        keras_res = keras_results[sample_idx]
        tflite_res = tflite_results[sample_idx]
        
        # Load image
        img = cv2.imread(keras_res['boxes'][0] if False else tflite_res['boxes'][0] if False else None)
        # Reload from path
        for test_item in [keras_res, tflite_res]:
            if 'img_path' in test_item:
                img_path = test_item['img_path']
                break
        else:
            # Find from filename
            img_path = None
            for seq in ['daySequence1', 'daySequence2', 'nightSequence1', 'nightSequence2']:
                test_path = os.path.join(DATASET_ROOT, seq, seq, "frames", keras_res['filename'])
                if os.path.exists(test_path):
                    img_path = test_path
                    break
        
        if img_path and os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        else:
            continue
        
        # Plot Keras results
        ax = axes[idx, 0]
        ax.imshow(img)
        ax.set_title(f"KERAS - {keras_res['filename']}\nGT:{len(keras_res['gt_boxes'])} | Pred:{len(keras_res['boxes'])}", fontsize=9)
        
        # Draw GT boxes (green)
        for gt in keras_res['gt_boxes']:
            x1 = int(gt['x1'] * IMG_WIDTH / keras_res['orig_w'])
            y1 = int(gt['y1'] * IMG_HEIGHT / keras_res['orig_h'])
            x2 = int(gt['x2'] * IMG_WIDTH / keras_res['orig_w'])
            y2 = int(gt['y2'] * IMG_HEIGHT / keras_res['orig_h'])
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)
        
        # Draw predictions (red)
        for box in keras_res['boxes']:
            x1, y1, x2, y2, conf, cls_id = box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            label = f"{CLASS_NAMES[cls_id]}:{conf:.2f}"
            ax.text(x1, y1-2, label, color='red', fontsize=7, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.axis('off')
        
        # Plot TFLite results
        ax = axes[idx, 1]
        ax.imshow(img)
        ax.set_title(f"TFLITE - {tflite_res['filename']}\nGT:{len(tflite_res['gt_boxes'])} | Pred:{len(tflite_res['boxes'])}", fontsize=9)
        
        # Draw GT boxes (green)
        for gt in tflite_res['gt_boxes']:
            x1 = int(gt['x1'] * IMG_WIDTH / tflite_res['orig_w'])
            y1 = int(gt['y1'] * IMG_HEIGHT / tflite_res['orig_h'])
            x2 = int(gt['x2'] * IMG_WIDTH / tflite_res['orig_w'])
            y2 = int(gt['y2'] * IMG_HEIGHT / tflite_res['orig_h'])
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)
        
        # Draw predictions (red)
        for box in tflite_res['boxes']:
            x1, y1, x2, y2, conf, cls_id = box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            label = f"{CLASS_NAMES[cls_id]}:{conf:.2f}"
            ax.text(x1, y1-2, label, color='red', fontsize=7, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.axis('off')
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f'comparison-{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved: {filename}")
    plt.close()

def main():
    print("="*80)
    print("STANDALONE MODEL COMPARISON TEST")
    print("="*80)
    
    # Load test data
    print("\n[Loading test images...]")
    test_data = load_test_images(num_samples=30)
    print(f"Loaded {len(test_data)} test images")
    
    # Test Keras model
    keras_results = test_keras_model(test_data)
    if keras_results:
        calculate_simple_metrics(keras_results)
    
    # Test TFLite model
    tflite_results = test_tflite_model(test_data)
    if tflite_results:
        calculate_simple_metrics(tflite_results)
    
    # Create visualization
    if keras_results and tflite_results:
        print("\n[Creating comparison visualization...]")
        visualize_comparison(keras_results, tflite_results, num_samples=6)
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
