"""
LISA Traffic Light Detection + Classification System
True detection: locates traffic lights in full images AND classifies them
TensorFlow 2.19+ with TFLite conversion and OV2640 color calibration
Uses SSD-style architecture optimized for edge devices
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import cv2
import os
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime

# Configuration
# could test 352 x 288 for better accuracy
IMG_HEIGHT = 240
IMG_WIDTH = 320
GRID_SIZE = 16
CELL_HEIGHT = IMG_HEIGHT // GRID_SIZE
CELL_WIDTH = IMG_WIDTH // GRID_SIZE
NUM_ANCHORS = 4
BATCH_SIZE = 8
EPOCHS = 60
LEARNING_RATE = 0.0002
IOU_THRESHOLD_EVAL = 0.25 
IOU_THRESHOLD_NMS = 0.2   # aggresive NMS 
CONF_THRESHOLD = 0.5 # 0.75 reduced for testing
MAX_SAMPLES = 36775 # Total Lisa Traffic Light Dataset samples (36775 total)

# Dataset paths
DATASET_ROOT = "LISA Traffic Light Dataset"
ANNOTATIONS_ROOT = os.path.join(DATASET_ROOT, "Annotations", "Annotations")

# Global anchor boxes (will be computed dynamically)
ANCHORS = None

def apply_ov2640_color_calibration(image, add_noise=True):
    """Apply color calibration to match OV2640 camera characteristics."""
    image = image.astype(np.float32) / 255.0

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = hsv[:, :, 1] * 0.9  # Reduce saturation
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 1)  # Increase brightness
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # Warmer tone
    color_matrix = np.array([
        [1.05, 0.00, 0.00],
        [0.00, 0.98, 0.00],
        [0.00, 0.00, 0.92]
    ])
    image = np.dot(image, color_matrix.T)
    image = np.clip(image, 0, 1)

    # Add sensor noise only when requested
    if add_noise:
        noise = np.random.normal(0, 0.01, image.shape)
        image = np.clip(image + noise, 0, 1)

    return (image * 255).astype(np.uint8)

class DetectionPreview(keras.callbacks.Callback):
    def __init__(self, val_data, loader, num_samples=3):
        super().__init__()
        self.val_data = val_data
        self.loader = loader
        self.num_samples = num_samples

    def on_epoch_end(self, epoch, logs=None):
        sample_data = random.sample(self.val_data, min(self.num_samples, len(self.val_data)))
        results = []
        for item in sample_data:
            img = self.loader.preprocess_image(item['img_path'], item['apply_calibration'])
            if img is None:
                continue
            preds = self.model.predict(np.expand_dims(img/255.0, axis=0), verbose=0)[0]
            boxes = decode_predictions(preds)
            boxes = non_max_suppression(boxes)

            results.append({
                'filename': item['filename'],
                'ground_truth_count': len(item['boxes']),
                'predicted_count': len(boxes),
                'detections': {self.loader.class_names[b[5]]: 1 for b in boxes},
                'boxes': boxes,
                'image': img,
                'gt_boxes': item['boxes'],
                'orig_w': item['orig_w'], 
                'orig_h': item['orig_h']   
            })

        visualize_detections(results, num_display=self.num_samples)
        print(f"Epoch {epoch+1}: detection preview saved")

class MAPCallback(keras.callbacks.Callback):
    def __init__(self, val_data, loader, class_names, iou_threshold=IOU_THRESHOLD_EVAL , num_samples=200):
        super().__init__()
        self.val_data = val_data
        self.loader = loader
        self.class_names = class_names
        self.iou_threshold = iou_threshold
        self.num_samples = num_samples

    def on_epoch_end(self, epoch, logs=None):
        sample_data = random.sample(self.val_data, min(self.num_samples, len(self.val_data)))
        
        results = []
        for item in sample_data:
            img = self.loader.preprocess_image(item['img_path'], item['apply_calibration'])
            if img is None:
                continue
            preds = self.model.predict(np.expand_dims(img/255.0, axis=0), verbose=0)[0]
            boxes = decode_predictions(preds)
            boxes = non_max_suppression(boxes)
            results.append({
                'filename': item['filename'],
                'boxes': boxes,
                'gt_boxes': item['boxes'],
                'orig_w': item['orig_w'],
                'orig_h': item['orig_h']
            })
        
        aps, map_score = evaluate_map(results, self.class_names, iou_threshold=self.iou_threshold)
        
        logs = logs or {}
        logs['val_mAP'] = map_score
        print(f"\nEpoch {epoch+1}: val_mAP={map_score:.3f}")

class LISADetectionDataLoader:
    """Load LISA dataset for detection + classification"""
    
    def __init__(self, dataset_root, annotations_root):
        self.dataset_root = dataset_root
        self.annotations_root = annotations_root
        self.class_names = ['go', 'goLeft', 'stop', 'stopLeft', 'warning', 'warningLeft']

        self.class_mapping = {
            # Go variants
            'go': 'go',
            'goLeft': 'goLeft',
            'go_forward': 'go',  # go_forward → go
            'go_traffic_light': 'go',
            'goLeft_traffic_light': 'goLeft',
            'go_forward_traffic_light': 'go',
            
            # Stop variants
            'stop': 'stop',
            'stopLeft': 'stopLeft',
            'stop_traffic_light': 'stop',
            'stopLeft_traffic_light': 'stopLeft',
            
            # Warning variants
            'warning': 'warning',
            'warningLeft': 'warningLeft',
            'warning_traffic_light': 'warning',
            'warningLeft_traffic_light': 'warningLeft',
        }

        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)
        self.anchors = None  # Will be computed dynamically
        
    def normalize_class_name(self, raw_class):
        """
        Normaliza nombre de clase y mapea a clase base
        """
        # Limpiar espacios y guiones
        clean = str(raw_class).strip().lower().replace(' ', '').replace('-', '')
        
        # Buscar en mapeo
        for original, mapped in self.class_mapping.items():
            if clean == original.lower().replace('_', ''):
                return mapped
        
        # Si no se encuentra, intentar match parcial
        if 'goleft' in clean or 'leftgo' in clean:
            return 'goLeft'
        elif 'go' in clean or 'green' in clean:
            return 'go'
        elif 'stopleft' in clean or 'leftstop' in clean:
            return 'stopLeft'
        elif 'stop' in clean or 'red' in clean:
            return 'stop'
        elif 'warningleft' in clean or 'leftwarning' in clean:
            return 'warningLeft'
        elif 'warning' in clean or 'yellow' in clean:
            return 'warning'
        
        # Si no se pudo mapear, retornar None (será filtrado)
        return None
        
    def parse_annotations(self, csv_path):
        """Parse annotation CSV file"""
        df = pd.read_csv(csv_path, delimiter=';')
        
        grouped = {}
        for _, row in df.iterrows():
            filename = row['Filename']
            
            # Fix filename path
            if '/' in filename:
                filename = filename.split('/')[-1]
            if '\\' in filename:
                filename = filename.split('\\')[-1]
            
            if filename not in grouped:
                grouped[filename] = []
            
            # Normalize class/annotation tag
            raw_tag = str(row['Annotation tag']).strip()

            matched_label = None
            for cname in self.class_names:
                if raw_tag.lower().replace(' ', '').replace('-', '') == cname.lower().replace(' ', '').replace('-', ''):
                    matched_label = cname
                    break

            if matched_label is None:
                matched_label = self.normalize_class_name(matched_label)
            if matched_label is None or matched_label not in self.class_to_idx:
                continue  # Skip unknown classes
            annotation = {
                'x1': int(row['Upper left corner X']),
                'y1': int(row['Upper left corner Y']),
                'x2': int(row['Lower right corner X']),
                'y2': int(row['Lower right corner Y']),
                'class': matched_label
            }
            grouped[filename].append(annotation)
        
        return grouped
    
    def compute_optimal_anchors(self, all_data):
        """Compute optimal anchor boxes using K-means clustering"""
        print("\n[Computing optimal anchors using K-means...]")
        
        # Collect all box sizes (normalized to image fractions)
        box_sizes = []
        for item in all_data:
            # Get original image dimensions
            img_path = os.path.join(item['img_dir'], item['filename'])
            if not os.path.exists(img_path):
                continue
            
            img = cv2.imread(img_path)
            if img is None:
                continue
            orig_h, orig_w = img.shape[:2]
            
            for box in item['boxes']:
                # Skip unknown classes
                if box['class'] not in self.class_to_idx:
                    continue
                
                # Normalize to [0,1] image fractions
                w = (box['x2'] - box['x1']) / orig_w
                h = (box['y2'] - box['y1']) / orig_h
                box_sizes.append([w, h])
        
        box_sizes = np.array(box_sizes)
        print(f"Collected {len(box_sizes)} bounding boxes for anchor computation")
        
        if len(box_sizes) == 0:
            print("WARNING: No valid boxes found! Using default anchors.")
            self.anchors = np.array([
                [0.05, 0.05],
                [0.08, 0.08],
                [0.12, 0.12],
                [0.15, 0.15]
            ])
            return
        
        # Print statistics
        print(f"Box size statistics (image fractions):")
        print(f"  Width  - min: {box_sizes[:, 0].min():.4f}, max: {box_sizes[:, 0].max():.4f}, mean: {box_sizes[:, 0].mean():.4f}")
        print(f"  Height - min: {box_sizes[:, 1].min():.4f}, max: {box_sizes[:, 1].max():.4f}, mean: {box_sizes[:, 1].mean():.4f}")
        
        # K-means clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=NUM_ANCHORS, random_state=42, n_init=10)
        kmeans.fit(box_sizes)
        
        # Sort anchors by area (small to large)
        anchors = kmeans.cluster_centers_
        areas = anchors[:, 0] * anchors[:, 1]
        sorted_indices = np.argsort(areas)
        self.anchors = anchors[sorted_indices]
        
        print(f"\nOptimal anchors (image fractions [width, height]):")
        for i, anchor in enumerate(self.anchors):
            area = anchor[0] * anchor[1]
            print(f"  Anchor {i}: [{anchor[0]:.4f}, {anchor[1]:.4f}] - area: {area:.6f}")
        
        # Also print in pixel dimensions for reference
        print(f"\nOptimal anchors (pixel dimensions @ {IMG_WIDTH}x{IMG_HEIGHT}):")
        for i, anchor in enumerate(self.anchors):
            w_px = anchor[0] * IMG_WIDTH
            h_px = anchor[1] * IMG_HEIGHT
            print(f"  Anchor {i}: [{w_px:.1f}px, {h_px:.1f}px]")
    
    def load_dataset(self):
        """Load all annotations from dataset - NO SPLIT, just load everything"""
        all_data = []
        
        # Training data - day
        day_train_path = os.path.join(self.annotations_root, "dayTrain")
        if os.path.exists(day_train_path):
            for clip in os.listdir(day_train_path):
                clip_path = os.path.join(day_train_path, clip)
                if os.path.isdir(clip_path):
                    bulb_csv = os.path.join(clip_path, "frameAnnotationsBULB.csv")
                    if os.path.exists(bulb_csv):
                        annotations = self.parse_annotations(bulb_csv)
                        img_dir = os.path.join(self.dataset_root, "dayTrain", "dayTrain", clip, "frames")
                        for filename, boxes in annotations.items():
                            all_data.append({
                                'filename': filename,
                                'img_dir': img_dir,
                                'boxes': boxes
                                # ← REMOVED 'split': 'train'
                            })
        
        # Training data - night
        night_train_path = os.path.join(self.annotations_root, "nightTrain")
        if os.path.exists(night_train_path):
            for clip in os.listdir(night_train_path):
                clip_path = os.path.join(night_train_path, clip)
                if os.path.isdir(clip_path):
                    bulb_csv = os.path.join(clip_path, "frameAnnotationsBULB.csv")
                    if os.path.exists(bulb_csv):
                        annotations = self.parse_annotations(bulb_csv)
                        img_dir = os.path.join(self.dataset_root, "nightTrain", "nightTrain", clip, "frames")
                        for filename, boxes in annotations.items():
                            all_data.append({
                                'filename': filename,
                                'img_dir': img_dir,
                                'boxes': boxes
                                # ← REMOVED 'split': 'train'
                            })
        
        # Test sequences - NOW INCLUDED IN TRAINING!
        for seq in ['daySequence1', 'daySequence2', 'nightSequence1', 'nightSequence2']:
            seq_path = os.path.join(self.annotations_root, seq)
            if os.path.exists(seq_path):
                bulb_csv = os.path.join(seq_path, "frameAnnotationsBULB.csv")
                if os.path.exists(bulb_csv):
                    annotations = self.parse_annotations(bulb_csv)
                    img_dir = os.path.join(self.dataset_root, seq, seq, "frames")
                    for filename, boxes in annotations.items():
                        all_data.append({
                            'filename': filename,
                            'img_dir': img_dir,
                            'boxes': boxes
                            # ← REMOVED 'split': 'test'
                        })

        # Compute optimal anchors from ALL data
        self.compute_optimal_anchors(all_data)
        
        # Set global ANCHORS variable
        global ANCHORS
        ANCHORS = self.anchors

        return all_data
    
    def encode_ground_truth(self, boxes, img_width, img_height):
        """YOLO-style anchor-based encoding"""
        target = np.zeros((GRID_SIZE, GRID_SIZE, NUM_ANCHORS, 5 + self.num_classes))
        
        for box in boxes:
            if box['class'] not in self.class_to_idx:
                continue
            
            # Normalize to [0,1] relative to full image
            x_center = ((box['x1'] + box['x2']) / 2) / img_width
            y_center = ((box['y1'] + box['y2']) / 2) / img_height
            box_w = (box['x2'] - box['x1']) / img_width
            box_h = (box['y2'] - box['y1']) / img_height
            
            # Find grid cell
            grid_x = int(x_center * GRID_SIZE)
            grid_y = int(y_center * GRID_SIZE)
            grid_x = min(grid_x, GRID_SIZE - 1)
            grid_y = min(grid_y, GRID_SIZE - 1)
            
            # Find best anchor by IoU
            best_anchor = 0
            best_iou = 0
            for anchor_idx, anchor in enumerate(self.anchors):
                anchor_w = anchor[0]
                anchor_h = anchor[1]
                
                intersection = min(box_w, anchor_w) * min(box_h, anchor_h)
                union = box_w * box_h + anchor_w * anchor_h - intersection
                iou = intersection / union if union > 0 else 0
                
                if iou > best_iou:
                    best_iou = iou
                    best_anchor = anchor_idx
            
            # Cell-relative position [0,1]
            cell_x = x_center * GRID_SIZE - grid_x
            cell_y = y_center * GRID_SIZE - grid_y
            
            # Encode box size relative to anchor
            anchor_w = self.anchors[best_anchor][0]
            anchor_h = self.anchors[best_anchor][1]
            
            # YOLO-style: log of ratio
            tw = np.log(box_w / (anchor_w + 1e-16))
            th = np.log(box_h / (anchor_h + 1e-16))
            
            # Store in target
            class_idx = self.class_to_idx[box['class']]
            target[grid_y, grid_x, best_anchor, 0] = 1.0  # objectness
            target[grid_y, grid_x, best_anchor, 1] = cell_x
            target[grid_y, grid_x, best_anchor, 2] = cell_y
            target[grid_y, grid_x, best_anchor, 3] = tw
            target[grid_y, grid_x, best_anchor, 4] = th
            target[grid_y, grid_x, best_anchor, 5 + class_idx] = 1.0
        
        return target
    
    def preprocess_image(self, img_path, apply_calibration=True):
        """Load and preprocess full image"""
        img = cv2.imread(img_path)
        
        if img is None:
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

        if apply_calibration:
            img = apply_ov2640_color_calibration(img, add_noise=True)
            
            # Random brightness jitter
            if random.random() < 0.5:
                factor = 1.0 + np.random.uniform(-0.2, 0.2)
                img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

            # Random HSV/value jitter
            if random.random() < 0.5:
                hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1.0 + np.random.uniform(-0.15, 0.15)), 0, 255)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + np.random.uniform(-0.1, 0.1)), 0, 255)
                img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

            # Small Gaussian blur sometimes
            if random.random() < 0.3:
                k = random.choice([3, 5])
                img = cv2.GaussianBlur(img, (k, k), 0)

            # Add small Gaussian noise
            if random.random() < 0.3:
                noise = np.random.normal(0, 4, img.shape).astype(np.float32)
                img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return img
    
    def create_dataset(self, data, apply_calibration=True):
        """Create dataset metadata (paths only, not loaded into memory)"""
        valid_data = []
        
        print(f"Validating samples (max {MAX_SAMPLES})...")
        
        for idx, item in enumerate(data):
            if idx % 500 == 0 and idx > 0:
                print(f"  Validated {idx} samples...")
            
            # Apply MAX_SAMPLES limit
            if len(valid_data) >= MAX_SAMPLES:
                print(f"  Reached MAX_SAMPLES limit ({MAX_SAMPLES})")
                break
            
            img_path = os.path.join(item['img_dir'], item['filename'])
            
            if not os.path.exists(img_path):
                continue
            
            # Get original image size for encoding
            orig_img = cv2.imread(img_path)
            if orig_img is None:
                continue
            orig_h, orig_w = orig_img.shape[:2]
            
            # Store metadata
            item['img_path'] = img_path
            item['orig_w'] = orig_w
            item['orig_h'] = orig_h
            item['apply_calibration'] = apply_calibration
            valid_data.append(item)
        
        print(f"  Valid samples: {len(valid_data)}")
        return valid_data
    
    def data_generator(self, data, batch_size, shuffle=True):
        """Generator that yields batches of (images, targets)"""
        indices = np.arange(len(data))
        
        while True:
            if shuffle:
                np.random.shuffle(indices)
            
            for start_idx in range(0, len(data), batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                
                batch_images = []
                batch_targets = []
                
                for idx in batch_indices:
                    item = data[idx]
                    
                    # Load and preprocess image
                    img = self.preprocess_image(item['img_path'], item['apply_calibration'])
                    if img is None:
                        continue
                    
                    # Encode ground truth
                    target = self.encode_ground_truth(item['boxes'], item['orig_w'], item['orig_h'])
                    
                    batch_images.append(img / 255.0)
                    batch_targets.append(target)
                
                if len(batch_images) > 0:
                    yield (np.array(batch_images, dtype=np.float32), 
                           np.array(batch_targets, dtype=np.float32))

def classification_metrics(y_true, y_pred):
    """Métricas más informativas"""
    cls_true = y_true[..., 5:]
    cls_pred = tf.sigmoid(y_pred[..., 5:])
    obj_true = y_true[..., 0]
    obj_pred = tf.sigmoid(y_pred[..., 0])
    
    # Accuracy solo en celdas positivas (tu métrica actual)
    cls_true_idx = tf.argmax(cls_true, axis=-1)
    cls_pred_idx = tf.argmax(cls_pred, axis=-1)
    obj_mask = tf.cast(obj_true > 0.5, tf.float32)
    
    correct = tf.cast(tf.equal(cls_true_idx, cls_pred_idx), tf.float32) * obj_mask
    pos_accuracy = tf.reduce_sum(correct) / (tf.reduce_sum(obj_mask) + 1e-7)
    
    # NUEVO: Accuracy en todas las predicciones confiadas
    pred_mask = tf.cast(obj_pred > 0.5, tf.float32)  # Predicciones confiadas
    has_gt = tf.cast(obj_true > 0.5, tf.float32)  # Tiene GT
    
    # TP: predicción confiada con GT correcto
    tp = correct
    # FP: predicción confiada sin GT o con clase incorrecta
    fp = pred_mask * (1 - has_gt) + pred_mask * has_gt * (1 - tf.cast(tf.equal(cls_true_idx, cls_pred_idx), tf.float32))
    
    precision = tf.reduce_sum(tp) / (tf.reduce_sum(tp + fp) + 1e-7)
    
    return {
        'cls_acc_positive': pos_accuracy,  # Tu métrica actual
        'cls_precision': precision  # Nueva métrica más realista
    }

def classification_accuracy(y_true, y_pred):
    """Classification accuracy metric"""
    cls_true = y_true[..., 5:]
    cls_pred = tf.sigmoid(y_pred[..., 5:])
    cls_true_idx = tf.argmax(cls_true, axis=-1)
    cls_pred_idx = tf.argmax(cls_pred, axis=-1)

    obj_mask = tf.cast(y_true[..., 0], tf.float32)
    correct = tf.cast(tf.equal(cls_true_idx, cls_pred_idx), tf.float32) * tf.cast(obj_mask, tf.float32)

    return tf.reduce_sum(correct) / (tf.reduce_sum(tf.cast(obj_mask, tf.float32)) + 1e-7)

def create_detection_model(num_classes):
    """Create detection model with PROPERLY INITIALIZED separate heads"""
    inputs = keras.Input(shape=(240, 320, 3))
    
    backbone = keras.applications.MobileNetV2(
        input_shape=(240, 320, 3),
        include_top=False,
        weights='imagenet',
        alpha=0.35
    )
    backbone.trainable = True
    for layer in backbone.layers[:-50]:
        layer.trainable = False

    x = backbone(inputs, training=False)
    
    reg = keras.regularizers.l2(1e-4)  # Reduced from 5e-4
    
    # Resize to grid
    x = layers.Resizing(16, 16)(x)
    
    # Shared feature processing
    x = layers.Conv2D(128, 3, padding='same', activation='relu', kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    # CRITICAL: Separate processing paths BEFORE final heads
    # Box/objectness branch
    box_features = layers.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=reg)(x)
    box_features = layers.BatchNormalization()(box_features)
    
    # Classification branch (MORE capacity for harder task)
    cls_features = layers.Conv2D(128, 3, padding='same', activation='relu', kernel_regularizer=reg)(x)
    cls_features = layers.BatchNormalization()(cls_features)
    cls_features = layers.Dropout(0.3)(cls_features)
    cls_features = layers.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=reg)(cls_features)
    cls_features = layers.BatchNormalization()(cls_features)
    
    # Final prediction heads with PROPER initialization
    box_head = layers.Conv2D(
        4 * 5, 1, padding='same',
        kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
        bias_initializer='zeros'
    )(box_features)
    
    cls_head = layers.Conv2D(
        4 * num_classes, 1, padding='same',
        kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
        # CRITICAL: Initialize bias to favor uniform distribution
        bias_initializer=keras.initializers.Constant(-np.log((num_classes - 1)))
    )(cls_features)
    
    # Reshape and concatenate
    box_out = layers.Reshape((16, 16, 4, 5))(box_head)
    cls_out = layers.Reshape((16, 16, 4, num_classes))(cls_head)
    outputs = layers.Concatenate(axis=-1)([box_out, cls_out])

    return keras.Model(inputs, outputs)

# Funcion alternativa para la pérdida con focal loss y pesos ajustados   
def detection_loss(y_true, y_pred):
    """
    FINAL v4: Prioriza localización sobre clasificación
    Basado en diagnóstico: IoU<0.3 en 96.5% es el problema principal
    """
    epsilon = 1e-7
    
    # PESOS CORREGIDOS
    class_weights = tf.constant([
        1.00,  # go          (13000 samples, 44.7%)
        2.27,  # goLeft      (2500 samples, 8.6%)
        1.09,  # stop        (11000 samples, 37.8%)
        2.55,  # stopLeft    (2000 samples, 6.9%)
        5.10,  # warning     (500 samples, 1.7%)
        11.40  # warningLeft (100 samples, 0.3%)
    ], dtype=tf.float32)
    
    obj_true = y_true[..., 0:1]
    box_true = y_true[..., 1:5]
    cls_true = y_true[..., 5:]
    
    obj_pred = tf.sigmoid(y_pred[..., 0:1])
    cls_pred_logits = y_pred[..., 5:]
    
    tx_ty_pred = tf.sigmoid(y_pred[..., 1:3])
    tw_th_pred = y_pred[..., 3:5]
    box_pred = tf.concat([tx_ty_pred, tw_th_pred], axis=-1)
    
    obj_mask = obj_true
    noobj_mask = 1 - obj_true
    
    # Objectness
    obj_pred_clipped = tf.clip_by_value(obj_pred, epsilon, 1 - epsilon)
    obj_bce = -(obj_true * tf.math.log(obj_pred_clipped) + 
                (1 - obj_true) * tf.math.log(1 - obj_pred_clipped))
    
    gamma_obj = 2.0
    pt_obj = tf.where(obj_true > 0.5, obj_pred, 1 - obj_pred)
    focal_weight_obj = tf.pow(1 - pt_obj, gamma_obj)
    
    obj_loss = obj_mask * obj_bce * focal_weight_obj * 6.0
    noobj_loss = noobj_mask * obj_bce * focal_weight_obj * 0.5
    objectness_loss = tf.reduce_mean(obj_loss + noobj_loss)
    
    # Box regression (Smooth L1)
    position_diff = box_true[..., :2] - box_pred[..., :2]
    size_diff = box_true[..., 2:] - box_pred[..., 2:]
    
    def smooth_l1(diff):
        abs_diff = tf.abs(diff)
        return tf.where(abs_diff < 1.0, 0.5 * tf.square(diff), abs_diff - 0.5)
    
    position_loss = tf.reduce_sum(smooth_l1(position_diff), axis=-1, keepdims=True)
    size_loss = tf.reduce_sum(smooth_l1(size_diff), axis=-1, keepdims=True)
    
    box_loss = obj_mask * (3.0 * position_loss + 1.0 * size_loss)
    box_loss = tf.reduce_sum(box_loss) / (tf.reduce_sum(obj_mask) + epsilon)
    
    # Classification
    logits_max = tf.reduce_max(cls_pred_logits, axis=-1, keepdims=True)
    logits_stable = cls_pred_logits - logits_max
    exp_logits = tf.exp(logits_stable)
    sum_exp = tf.reduce_sum(exp_logits, axis=-1, keepdims=True)
    
    probs = exp_logits / (sum_exp + epsilon)
    log_probs = logits_stable - tf.math.log(sum_exp + epsilon)
    
    gamma_cls = 2.0
    pt_cls = tf.reduce_sum(cls_true * probs, axis=-1, keepdims=True)
    focal_weight_cls = tf.pow(1 - pt_cls, gamma_cls)
    
    cls_ce_per_class = -cls_true * log_probs
    weighted_ce = cls_ce_per_class * tf.reshape(class_weights, [1, 1, 1, 1, -1])
    
    cls_loss_raw = tf.reduce_sum(weighted_ce, axis=-1, keepdims=True)
    cls_loss = obj_mask * cls_loss_raw * focal_weight_cls
    cls_loss = tf.reduce_sum(cls_loss) / (tf.reduce_sum(obj_mask) + epsilon)
    
    # Class-aware objectness
    cls_pred_idx = tf.argmax(probs, axis=-1)
    cls_true_idx = tf.argmax(cls_true, axis=-1)
    class_mismatch = tf.cast(tf.not_equal(cls_pred_idx, cls_true_idx), tf.float32)
    class_mismatch = tf.expand_dims(class_mismatch, axis=-1)
    
    cls_confidence = tf.reduce_max(probs, axis=-1, keepdims=True)
    confident_wrong = class_mismatch * tf.cast(cls_confidence > 0.5, tf.float32)
    
    class_obj_penalty = obj_pred * confident_wrong * obj_mask
    class_obj_loss = tf.reduce_mean(class_obj_penalty)
    
    # Background suppression
    high_conf_background = noobj_mask * obj_pred * tf.cast(obj_pred > 0.5, tf.float32)
    background_penalty = tf.reduce_mean(high_conf_background)
    
    # TOTAL LOSS (balanceado)
    total_loss = (
        8.0 * box_loss +
        3.0 * objectness_loss +
        8.0 * cls_loss +
        1.0 * class_obj_loss +
        0.5 * background_penalty
    )
    
    return total_loss
    
# def detection_loss(y_true, y_pred):
#     """
#     Detection loss using focal loss for BOTH objectness AND classification
#     """
#     epsilon = 1e-7
    
#     # Split predictions
#     obj_true = y_true[..., 0:1]
#     box_true = y_true[..., 1:5]
#     cls_true = y_true[..., 5:]
    
#     obj_pred = tf.sigmoid(y_pred[..., 0:1])
#     cls_pred_logits = y_pred[..., 5:]
    
#     tx_ty_pred = tf.sigmoid(y_pred[..., 1:3])
#     tw_th_pred = y_pred[..., 3:5]
#     box_pred = tf.concat([tx_ty_pred, tw_th_pred], axis=-1)
    
#     obj_mask = obj_true
#     noobj_mask = 1 - obj_true
    
#     # ============================================
#     # 1. OBJECTNESS LOSS with FOCAL LOSS
#     # ============================================
#     obj_pred_clipped = tf.clip_by_value(obj_pred, epsilon, 1 - epsilon)
    
#     # BCE
#     obj_bce = -(obj_true * tf.math.log(obj_pred_clipped) + 
#                 (1 - obj_true) * tf.math.log(1 - obj_pred_clipped))
    
#     # FOCAL LOSS for objectness
#     gamma_obj = 3.0
    
#     # pt: probability of correct prediction
#     # If obj_true=1, pt = obj_pred; if obj_true=0, pt = 1 - obj_pred
#     pt_obj = tf.where(obj_true > 0.5, obj_pred, 1 - obj_pred)
#     focal_weight_obj = tf.pow(1 - pt_obj, gamma_obj)
    
#     # Apply focal weight
#     obj_loss = obj_mask * obj_bce * focal_weight_obj * 6.0
#     noobj_loss = noobj_mask * obj_bce * focal_weight_obj * 0.3
#     objectness_loss = tf.reduce_mean(obj_loss + noobj_loss)
    
#     # ============================================
#     # 2. BOX REGRESSION LOSS
#     # ============================================
#     box_diff = box_true - box_pred
#     box_loss = obj_mask * tf.reduce_sum(tf.square(box_diff), axis=-1, keepdims=True)
#     box_loss = tf.reduce_sum(box_loss) / (tf.reduce_sum(obj_mask) + epsilon)

#     # L1 loss alternative
#     # box_diff = box_true - box_pred
#     # box_loss_l1 = tf.abs(box_diff)
#     # box_loss = obj_mask * tf.reduce_sum(box_loss_l1, axis=-1, keepdims=True)
#     # box_loss = tf.reduce_sum(box_loss) / (tf.reduce_sum(obj_mask) + epsilon)
    
#     # ============================================
#     # 3. CLASSIFICATION LOSS with FOCAL LOSS
#     # ============================================
    
#     # Softmax (numerically stable)
#     logits_max = tf.reduce_max(cls_pred_logits, axis=-1, keepdims=True)
#     logits_stable = cls_pred_logits - logits_max
#     exp_logits = tf.exp(logits_stable)
#     sum_exp = tf.reduce_sum(exp_logits, axis=-1, keepdims=True)
    
#     probs = exp_logits / (sum_exp + epsilon)
#     log_probs = logits_stable - tf.math.log(sum_exp + epsilon)
    
#     # FOCAL LOSS
#     gamma_cls = 2.0
#     pt_cls = tf.reduce_sum(cls_true * probs, axis=-1, keepdims=True)
#     focal_weight_cls = tf.pow(1 - pt_cls, gamma_cls)
    
#     cls_loss_raw = -tf.reduce_sum(cls_true * log_probs, axis=-1, keepdims=True)
#     cls_loss = obj_mask * cls_loss_raw * focal_weight_cls
#     cls_loss = tf.reduce_sum(cls_loss) / (tf.reduce_sum(obj_mask) + epsilon)
    
#     # ============================================
#     # 4. Diversity regularization (currently disabled)
#     # ============================================
#     cls_pred_mean = tf.reduce_mean(probs, axis=[0, 1, 2])
#     target_dist = tf.ones_like(cls_pred_mean) / 6.0
#     diversity_loss = tf.reduce_mean(tf.square(cls_pred_mean - target_dist))

#     # ============================================
#     # 5. Uncertainty penalty
#     # ============================================
#     # Penaliza objectness cuando la clasificación es incierta
#     cls_pred_entropy = -tf.reduce_sum(probs * tf.math.log(probs + epsilon), axis=-1, keepdims=True)
#     max_entropy = tf.math.log(6.0)  # log(num_classes)
#     normalized_entropy = cls_pred_entropy / max_entropy  # [0,1], 1=muy incierto

#     # Si objectness es alto pero clasificación incierta, penaliza
#     uncertainty_penalty = obj_pred * normalized_entropy * obj_mask
#     uncertainty_loss = tf.reduce_mean(uncertainty_penalty)

#     # ============================================
#     # 6. Class-conditional objectness penalty
#     # ============================================
#     # Only penalize objectness when classification is VERY wrong
#     # (not just uncertain, but confidently wrong)
    
#     # Get predicted and true class indices
#     cls_pred_idx = tf.argmax(probs, axis=-1) 
#     cls_true_idx = tf.argmax(cls_true, axis=-1) 
    
#     # Check if they match
#     class_mismatch = tf.cast(tf.not_equal(cls_pred_idx, cls_true_idx), tf.float32)
#     class_mismatch = tf.expand_dims(class_mismatch, axis=-1)  # ← Add dimension back
    
#     # Get confidence in predicted class
#     cls_confidence = tf.reduce_max(probs, axis=-1, keepdims=True)
    
#     # Only penalize when confidently wrong (>0.5 confidence)
#     confident_wrong = class_mismatch * tf.cast(cls_confidence > 0.7, tf.float32)
#     class_obj_penalty = obj_pred * confident_wrong * obj_mask
#     class_obj_loss = tf.reduce_mean(class_obj_penalty)
    
#     # ============================================
#     # TOTAL LOSS
#     # ============================================
#     total_loss = (
#         2.5 * box_loss + 
#         2.0 * objectness_loss + 
#         10.0 * cls_loss +
#         2.0 * class_obj_loss +
#         0.3 * diversity_loss 
#         # 1.0 * uncertainty_loss
#     )
    
#     return total_loss

# Version estable de la funcion de perdida (no focal loss, necesita ajuste de pesos)
# def detection_loss(y_true, y_pred):
#     """Custom detection loss function combining objectness, bbox, and classification losses"""
#     epsilon = 1e-7
#     # Class weights SUAVIZADOS con raíz cuadrada
#     raw_weights = np.array([
#         1.0,   # go
#         1.5,   # goLeft (era 2.0)
#         1.0,   # stop  
#         1.3,   # stopLeft (era 2.0)
#         3.0,   # warning (era 8.0)
#         4.0    # warningLeft (era 16.0)
#     ])
    
#     # Suaviza con sqrt para reducir extremos
#     class_weights = tf.constant(np.sqrt(raw_weights), dtype=tf.float32)
#     # Resultado: [1.0, 4.11, 1.07, 1.57, 2.86, 4.81]
    
#     # Split predictions / ground truth
#     obj_true = y_true[..., 0:1]
#     box_true = y_true[..., 1:5]
#     cls_true = y_true[..., 5:]

#     # Predicted: apply sigmoid for objectness and class logits
#     obj_pred = tf.sigmoid(y_pred[..., 0:1])
#     cls_pred = tf.sigmoid(y_pred[..., 5:])

#     # For box predictions: tx,ty should be sigmoided
#     tx_ty_pred = tf.sigmoid(y_pred[..., 1:3])
#     tw_th_pred = y_pred[..., 3:5]
#     box_pred = tf.concat([tx_ty_pred, tw_th_pred], axis=-1)

#     # Masks
#     obj_mask = obj_true
#     noobj_mask = 1 - obj_true

#     # Objectness BCE
#     obj_pred_clipped = tf.clip_by_value(obj_pred, epsilon, 1 - epsilon)
#     obj_bce = -(obj_true * tf.math.log(obj_pred_clipped) + (1 - obj_true) * tf.math.log(1 - obj_pred_clipped))
    
#     # Aumenta el peso de los positivos
#     obj_loss = obj_mask * obj_bce * 5.0  # Peso extra para objectness positivo
#     noobj_loss = noobj_mask * obj_bce * 0.5  # Reduce peso de negativos
#     objectness_loss = tf.reduce_mean(obj_loss + noobj_loss)

#     # Box regression (sin cambios)
#     box_diff = box_true - box_pred
#     box_loss = obj_mask * tf.reduce_sum(tf.square(box_diff), axis=-1, keepdims=True)
#     box_loss = tf.reduce_sum(box_loss) / (tf.reduce_sum(obj_mask) + epsilon)

#     # Classification BCE WITH WEIGHTS
#     cls_pred_clipped = tf.clip_by_value(cls_pred, epsilon, 1 - epsilon)
#     cls_bce = -(cls_true * tf.math.log(cls_pred_clipped) + 
#                 (1 - cls_true) * tf.math.log(1 - cls_pred_clipped))
    
#     # Apply per-class weights
#     weighted_cls_bce = cls_bce * tf.expand_dims(class_weights, axis=0)
#     cls_loss = obj_mask * tf.reduce_mean(weighted_cls_bce, axis=-1, keepdims=True)
#     cls_loss = tf.reduce_sum(cls_loss) / (tf.reduce_sum(obj_mask) + epsilon)
    
#     # Aumenta peso de box loss por IoU bajo
#     total_loss = 3.5 * box_loss + 2.0 * objectness_loss + 1.5 * cls_loss
    
#     return total_loss

def classification_precision(y_true, y_pred):
    """CORRECTED metric function - returns scalar, not dict"""
    cls_true = y_true[..., 5:]
    cls_pred = tf.sigmoid(y_pred[..., 5:])
    obj_true = y_true[..., 0]
    obj_pred = tf.sigmoid(y_pred[..., 0])
    
    cls_true_idx = tf.argmax(cls_true, axis=-1)
    cls_pred_idx = tf.argmax(cls_pred, axis=-1)
    obj_mask = tf.cast(obj_true > 0.5, tf.float32)
    
    correct = tf.cast(tf.equal(cls_true_idx, cls_pred_idx), tf.float32) * obj_mask
    
    # Precision: TP / (TP + FP)
    pred_mask = tf.cast(obj_pred > 0.5, tf.float32)
    tp = correct
    fp = pred_mask * (1 - tf.cast(tf.equal(cls_true_idx, cls_pred_idx), tf.float32))
    
    precision = tf.reduce_sum(tp) / (tf.reduce_sum(tp + fp) + 1e-7)
    return precision

def train_model(model, train_data, val_data, train_loader, val_loader):
    """Train detection model using data generators"""

    # Learning rate schedule with warmup
    def lr_schedule(epoch):
        if epoch < 3:  # Warmup
            return 0.0002 * (epoch + 1) / 3
        elif epoch < 15:
            return 0.0002
        elif epoch < 30:
            return 0.0001
        else:
            return 0.00005

    
    model.compile(
        # optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, weight_decay=1e-5),
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=detection_loss,
        metrics=[classification_precision]
    )
    
    # Calculate steps per epoch
    train_steps = len(train_data) // BATCH_SIZE
    val_steps = len(val_data) // BATCH_SIZE
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-7
        ),
        keras.callbacks.ModelCheckpoint(
            'best_detection_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        MAPCallback(val_data, val_loader, train_loader.class_names),
        DetectionPreview(val_data, val_loader, num_samples=6)
    ]
    
    # Create generators
    train_gen = train_loader.data_generator(train_data, BATCH_SIZE, shuffle=True)
    val_gen = val_loader.data_generator(val_data, BATCH_SIZE, shuffle=False)
    
    history = model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        validation_data=val_gen,
        validation_steps=val_steps,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def decode_predictions(predictions, conf_threshold=CONF_THRESHOLD):
    """YOLO-style decoding with anchors"""
    boxes = []
    
    if isinstance(predictions, tf.Tensor):
        predictions = predictions.numpy()
    
    for i in range(16):  # GRID_SIZE
        for j in range(16):
            for a in range(4):  # NUM_ANCHORS
                # Objectness
                obj_logit = np.clip(predictions[i, j, a, 0], -10, 10)
                obj_conf = 1 / (1 + np.exp(-obj_logit))
                
                if obj_conf < conf_threshold:
                    continue
                
                # Decodificar posición (sin cambios)
                tx = np.clip(predictions[i, j, a, 1], -10, 10)
                ty = np.clip(predictions[i, j, a, 2], -10, 10)
                cell_x = 1 / (1 + np.exp(-tx))
                cell_y = 1 / (1 + np.exp(-ty))
                
                tw = np.clip(predictions[i, j, a, 3], -10, 10)
                th = np.clip(predictions[i, j, a, 4], -10, 10)
                
                anchor_w = ANCHORS[a][0]
                anchor_h = ANCHORS[a][1]
                
                box_w = anchor_w * np.exp(tw)
                box_h = anchor_h * np.exp(th)
                
                x_center = (j + cell_x) / 16
                y_center = (i + cell_y) / 16
                
                # =============================================
                # MEJORA: Clip más generoso en bordes
                # =============================================
                x1 = int((x_center - box_w / 2) * 320)
                y1 = int((y_center - box_h / 2) * 240)
                x2 = int((x_center + box_w / 2) * 320)
                y2 = int((y_center + box_h / 2) * 240)
                
                # Permite boxes que se salen un poco (5 px)
                x1 = max(-5, min(325, x1))
                y1 = max(-5, min(245, y1))
                x2 = max(-5, min(325, x2))
                y2 = max(-5, min(245, y2))
                
                # Clip final
                x1 = max(0, min(320, x1))
                y1 = max(0, min(240, y1))
                x2 = max(0, min(320, x2))
                y2 = max(0, min(240, y2))
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Clasificación
                class_logits = np.clip(predictions[i, j, a, 5:], -10, 10)
                class_probs = 1 / (1 + np.exp(-class_logits))
                class_id = np.argmax(class_probs)
                class_conf = class_probs[class_id]
                
                confidence = obj_conf * class_conf
                boxes.append((x1, y1, x2, y2, confidence, int(class_id)))
    
    return boxes
    
def non_max_suppression(boxes, iou_threshold=IOU_THRESHOLD_NMS):
    """
    NMS mejorado específicamente para semáforos
    
    Cambios clave:
    1. Class-aware NMS: solo suprimir cajas de la MISMA clase
    2. Soft-NMS opcional para objetos pequeños
    3. Threshold más estricto para reducir duplicados
    
    Paper: "Improving Object Detection With One Line of Code" (ICCV 2017)
    Aplicado en LISA por: "Traffic Light Recognition in Varying Illumination" (IV 2018)
    
    Args:
        boxes: Lista de (x1, y1, x2, y2, confidence, class_id)
        iou_threshold: 0.3 es mejor para semáforos pequeños (vs 0.4-0.5 genérico)
        conf_threshold: Umbral mínimo de confianza
    """
    if len(boxes) == 0:
        return []
    
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    keep = []
    
    while len(boxes) > 0:
        current = boxes[0]
        keep.append(current)
        boxes = boxes[1:]
        
        filtered = []
        for box in boxes:
            iou = compute_iou(current, box)
            
            # Si son la MISMA clase y IoU alto → eliminar
            # Si son DIFERENTE clase → mantener incluso con IoU alto
            if box[5] == current[5]:  # Misma clase
                if iou < iou_threshold:
                    filtered.append(box)
            else:  # Diferente clase
                if iou < 0.7:  # Umbral más alto para clases diferentes
                    filtered.append(box)
        
        boxes = filtered
    
    return keep


def compute_iou(box1, box2):
    """Compute IoU between two boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1[:4]
    x1_2, y1_2, x2_2, y2_2 = box2[:4]
    
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


def convert_to_tflite(model, train_data, loader, output_path='traffic_light_detector.tflite', num_calibration_samples=250):
    """Convert to TFLite with INT8 quantization"""
    
    def representative_dataset():
        """Generate representative samples for quantization calibration"""
        calibration_samples = random.sample(train_data, min(num_calibration_samples, len(train_data)))
        
        for item in calibration_samples:
            img = loader.preprocess_image(item['img_path'], False)  # No augmentation for calibration
            if img is None:
                continue
            
            img_normalized = np.expand_dims(img / 255.0, axis=0).astype(np.float32)
            yield [img_normalized]
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    
    # Enforce full integer quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    print(f"Converting to INT8 quantized TFLite (using {num_calibration_samples} calibration samples)...")
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"\nINT8 TFLite model saved: {output_path}")
    print(f"Model size: {len(tflite_model) / 1024:.2f} KB")

    # Export anchors to a C header for firmware inclusion (if computed)
    try:
        if ANCHORS is not None:
            header_path = Path('anchors.h')
            with open(header_path, 'w') as hf:
                hf.write('// Auto-generated anchors header (from light_model.py)\n')
                hf.write('#ifndef ANCHORS_H\n')
                hf.write('#define ANCHORS_H\n\n')
                hf.write(f'#define GRID_SIZE {GRID_SIZE}\n')
                hf.write(f'#define NUM_ANCHORS {NUM_ANCHORS}\n')
                hf.write(f'static const float ANCHORS[NUM_ANCHORS][2] = {{\n')
                for a in ANCHORS:
                    hf.write(f'  {{{a[0]:.6f}f, {a[1]:.6f}f}},\n')
                hf.write('};\n\n#endif // ANCHORS_H\n')
            print(f"Exported anchors header: {header_path}")
    except Exception as e:
        print("WARNING: Failed to export anchors header:", e)
    
    return output_path

def evaluate_map(results, class_names, iou_threshold=IOU_THRESHOLD_EVAL , img_width=IMG_WIDTH, img_height=IMG_HEIGHT, verbose=True):
    """
    Compute mAP with DETAILED debugging to understand why mAP is low
    """
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    # Global stats
    total_predictions = 0
    total_gt = 0
    total_matches = 0
    
    # Track failures
    failure_reasons = {
        'no_predictions': 0,
        'wrong_class': 0,
        'low_iou': 0,
        'already_matched': 0,
        'successful_match': 0
    }
    
    # Per-class detailed tracking
    class_stats = {cn: {'tp': 0, 'fp': 0, 'fn': 0, 'gt_count': 0, 'pred_count': 0} for cn in class_names}
    
    # IoU histogram for analysis
    iou_values = []
    
    print("\n" + "="*80)
    print(f"mAP EVALUATION DEBUG (IoU threshold = {iou_threshold})")
    print("="*80)
    
    for result_idx, result in enumerate(results):
        orig_w = result.get('orig_w')
        orig_h = result.get('orig_h')
        if orig_w is None or orig_h is None:
            orig_h, orig_w = result.get('image', np.zeros((img_height, img_width))).shape[:2]
        
        # Scale GT boxes to model image space
        scaled_gts = []
        for gt in result['gt_boxes']:
            class_name = gt['class']
            if class_name not in class_to_idx:
                continue
            
            scaled_gt = {
                'x1': int(gt['x1'] * img_width / orig_w),
                'y1': int(gt['y1'] * img_height / orig_h),
                'x2': int(gt['x2'] * img_width / orig_w),
                'y2': int(gt['y2'] * img_height / orig_h),
                'class': class_name,
                'class_id': class_to_idx[class_name],
                'matched': False
            }
            scaled_gts.append(scaled_gt)
            class_stats[class_name]['gt_count'] += 1
            total_gt += 1
        
        # Get predictions
        pred_boxes = result['boxes']
        total_predictions += len(pred_boxes)
        
        if verbose and result_idx < 5:  # Show first 5 images in detail
            print(f"\n--- Image {result_idx + 1}: {result.get('filename', 'unknown')} ---")
            print(f"GT boxes: {len(scaled_gts)}, Predictions: {len(pred_boxes)}")
            print(f"Original size: {orig_w}x{orig_h}, Model size: {img_width}x{img_height}")
        
        if len(pred_boxes) == 0:
            failure_reasons['no_predictions'] += len(scaled_gts)
            for gt in scaled_gts:
                class_stats[gt['class']]['fn'] += 1
            if verbose and result_idx < 5:
                print("  ⚠️ NO PREDICTIONS for this image")
            continue
        
        # Process each prediction
        for pred_idx, pred in enumerate(pred_boxes):
            x1, y1, x2, y2, conf, pred_class_id = pred
            pred_class_name = class_names[pred_class_id]
            class_stats[pred_class_name]['pred_count'] += 1
            
            # Find best matching GT
            best_iou = 0
            best_gt = None
            best_gt_idx = None
            
            for gt_idx, gt in enumerate(scaled_gts):
                # Must match class
                if gt['class_id'] != pred_class_id:
                    continue
                
                # Compute IoU
                gt_box = (gt['x1'], gt['y1'], gt['x2'], gt['y2'], 1.0, gt['class_id'])
                iou = compute_iou(pred, gt_box)
                iou_values.append(iou)
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt
                    best_gt_idx = gt_idx
            
            # Check if this is a valid match
            if best_gt is None:
                # No GT of same class found
                failure_reasons['wrong_class'] += 1
                class_stats[pred_class_name]['fp'] += 1
                if verbose and result_idx < 5 and pred_idx < 3:
                    print(f"  Pred {pred_idx}: {pred_class_name} @ ({x1},{y1},{x2},{y2}) conf={conf:.3f}")
                    print(f"    ❌ WRONG CLASS - no GT of class '{pred_class_name}' in image")
                    
            elif best_iou < iou_threshold:
                # IoU too low
                failure_reasons['low_iou'] += 1
                class_stats[pred_class_name]['fp'] += 1
                if verbose and result_idx < 5 and pred_idx < 3:
                    gt_size = f"{best_gt['x2']-best_gt['x1']}x{best_gt['y2']-best_gt['y1']}"
                    pred_size = f"{x2-x1}x{y2-y1}"
                    print(f"  Pred {pred_idx}: {pred_class_name} @ ({x1},{y1},{x2},{y2}) conf={conf:.3f}")
                    print(f"    ❌ LOW IoU = {best_iou:.3f} < {iou_threshold}")
                    print(f"       GT: ({best_gt['x1']},{best_gt['y1']},{best_gt['x2']},{best_gt['y2']}) size={gt_size}")
                    print(f"       Pred size={pred_size}")
                    
            elif best_gt['matched']:
                # Already matched by another prediction
                failure_reasons['already_matched'] += 1
                class_stats[pred_class_name]['fp'] += 1
                if verbose and result_idx < 5 and pred_idx < 3:
                    print(f"  Pred {pred_idx}: {pred_class_name} @ ({x1},{y1},{x2},{y2}) conf={conf:.3f}")
                    print(f"    ❌ GT ALREADY MATCHED by another prediction")
                    
            else:
                # Valid match!
                best_gt['matched'] = True
                failure_reasons['successful_match'] += 1
                class_stats[pred_class_name]['tp'] += 1
                total_matches += 1
                if verbose and result_idx < 5 and pred_idx < 3:
                    print(f"  Pred {pred_idx}: {pred_class_name} @ ({x1},{y1},{x2},{y2}) conf={conf:.3f}")
                    print(f"    ✅ MATCH! IoU = {best_iou:.3f}")
        
        # Count false negatives (unmatched GTs)
        for gt in scaled_gts:
            if not gt['matched']:
                class_stats[gt['class']]['fn'] += 1
                if verbose and result_idx < 5:
                    gt_size = f"{gt['x2']-gt['x1']}x{gt['y2']-gt['y1']}"
                    print(f"  GT: {gt['class']} @ ({gt['x1']},{gt['y1']},{gt['x2']},{gt['y2']}) size={gt_size}")
                    print(f"    ❌ UNMATCHED (False Negative)")
    
    # Compute per-class AP
    aps = {}
    for class_name in class_names:
        stats = class_stats[class_name]
        if stats['gt_count'] == 0:
            continue
        
        # Simple precision/recall
        tp = stats['tp']
        fp = stats['fp']
        fn = stats['fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Approximate AP (would need sorted predictions for true AP)
        ap = precision * recall if recall > 0 else 0
        aps[class_name] = ap
    
    map_score = np.mean(list(aps.values())) if aps else 0.0
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nTotal GT boxes: {total_gt}")
    print(f"Total Predictions: {total_predictions}")
    print(f"Successful Matches: {total_matches} ({100*total_matches/total_gt if total_gt > 0 else 0:.1f}% recall)")
    
    print(f"\nFailure breakdown:")
    for reason, count in failure_reasons.items():
        if reason != 'successful_match':
            pct = 100 * count / total_gt if total_gt > 0 else 0
            print(f"  {reason}: {count} ({pct:.1f}%)")
    
    print(f"\nPer-class statistics:")
    print(f"{'Class':<15} {'GT':<6} {'Pred':<6} {'TP':<6} {'FP':<6} {'FN':<6} {'Precision':<10} {'Recall':<10} {'AP':<10}")
    print("-" * 85)
    for class_name in class_names:
        stats = class_stats[class_name]
        if stats['gt_count'] == 0:
            continue
        
        precision = stats['tp'] / (stats['tp'] + stats['fp']) if (stats['tp'] + stats['fp']) > 0 else 0
        recall = stats['tp'] / stats['gt_count'] if stats['gt_count'] > 0 else 0
        ap = aps.get(class_name, 0)
        
        print(f"{class_name:<15} {stats['gt_count']:<6} {stats['pred_count']:<6} {stats['tp']:<6} {stats['fp']:<6} {stats['fn']:<6} {precision:<10.3f} {recall:<10.3f} {ap:<10.3f}")
    
    # IoU distribution
    if iou_values:
        iou_array = np.array(iou_values)
        print(f"\nIoU distribution (all pred-GT pairs of same class):")
        print(f"  Mean: {iou_array.mean():.3f}")
        print(f"  Median: {np.median(iou_array):.3f}")
        print(f"  <0.3: {100*np.sum(iou_array < 0.3)/len(iou_array):.1f}%")
        print(f"  0.3-0.5: {100*np.sum((iou_array >= 0.3) & (iou_array < 0.5))/len(iou_array):.1f}%")
        print(f"  >=0.5: {100*np.sum(iou_array >= 0.5)/len(iou_array):.1f}%")
    
    print(f"\n{'='*80}")
    print(f"FINAL mAP @ IoU={iou_threshold}: {map_score:.3f}")
    print(f"{'='*80}\n")
    
    return aps, map_score

def test_tflite_model(tflite_path, test_data, class_names, num_samples=30):
    """Test TFLite detection model"""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_dtype = input_details[0]['dtype']
    is_quantized = input_dtype == np.uint8 or input_dtype == np.int8
    
    if is_quantized:
        input_scale, input_zero_point = input_details[0]['quantization']
        output_scale, output_zero_point = output_details[0]['quantization']
        print(f"INT8 quantized model detected")
    else:
        print(f"FLOAT32 model detected")
    
    loader = LISADetectionDataLoader(DATASET_ROOT, ANNOTATIONS_ROOT)
    loader.anchors = ANCHORS  # Use computed anchors
    
    random_samples = random.sample(test_data, min(num_samples, len(test_data)))
    
    print(f"\n{'='*80}")
    print(f"Testing TFLite Model on {len(random_samples)} samples")
    print(f"{'='*80}\n")
    
    results = []
    
    for idx, item in enumerate(random_samples):
        img = loader.preprocess_image(item['img_path'], False)  # No augmentation for testing
        
        if img is None:
            continue
        
        # Prepare input
        if is_quantized:
            img_normalized = img.astype(np.float32)
            input_data = (img_normalized / input_scale + input_zero_point).astype(input_dtype)
            input_data = np.expand_dims(input_data, axis=0)
        else:
            input_data = np.expand_dims(img / 255.0, axis=0).astype(np.float32)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        
        # Dequantize if needed
        if is_quantized:
            predictions = (predictions.astype(np.float32) - output_zero_point) * output_scale
        
        # Decode predictions
        boxes = decode_predictions(predictions)
        boxes = non_max_suppression(boxes)
        
        # Count detections
        detections = {}
        for box in boxes:
            class_id = int(box[5])
            class_name = class_names[class_id]
            detections[class_name] = detections.get(class_name, 0) + 1
        
        results.append({
            'filename': item['filename'],
            'ground_truth_count': len(item['boxes']),
            'predicted_count': len(boxes),
            'detections': detections,
            'boxes': boxes,
            'image': img,
            'gt_boxes': item['boxes'],
            'orig_w': item['orig_w'],
            'orig_h': item['orig_h']
        })
        
        det_str = ', '.join([f"{k}:{v}" for k, v in detections.items()]) if detections else "none"
        print(f"{idx+1:2d}. {item['filename']:35s} | GT: {len(item['boxes'])} | Pred: {len(boxes)} | {det_str}")
    
    print(f"\n{'='*80}\n")
    
    return results


def visualize_detections(results, num_display=6):
    """Visualize detection results"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    display_results = random.sample(results, min(num_display, len(results)))
    class_names = ['go', 'goLeft', 'stop', 'stopLeft', 'warning', 'warningLeft']
    
    for idx, result in enumerate(display_results):
        ax = axes[idx]
        ax.imshow(result['image'])
        
        orig_w = result.get('orig_w')
        orig_h = result.get('orig_h')
        
        if orig_w is None or orig_h is None:
            orig_w, orig_h = 1280, 960
        
        scale_x = IMG_WIDTH / orig_w
        scale_y = IMG_HEIGHT / orig_h
        
        # Draw ground truth (green)
        for box in result['gt_boxes']:
            x1 = int(box['x1'] * scale_x)
            y1 = int(box['y1'] * scale_y)
            x2 = int(box['x2'] * scale_x)
            y2 = int(box['y2'] * scale_y)
            
            x1 = max(0, min(IMG_WIDTH, x1))
            y1 = max(0, min(IMG_HEIGHT, y1))
            x2 = max(0, min(IMG_WIDTH, x2))
            y2 = max(0, min(IMG_HEIGHT, y2))
            
            w, h = x2 - x1, y2 - y1
            rect = patches.Rectangle((x1, y1), w, h, linewidth=2,
                                     edgecolor='green', facecolor='none', label='GT')
            ax.add_patch(rect)
        
        # Draw predictions (red)
        for box in result['boxes']:
            x1, y1, x2, y2, conf, class_id = box
            w, h = x2 - x1, y2 - y1
            rect = patches.Rectangle((x1, y1), w, h, linewidth=2,
                                     edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            label = f"{class_names[class_id]}: {conf:.2f}"
            ax.text(x1, y1-5, label, color='red', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        ax.set_title(f"{result['filename']}\nGT: {result['ground_truth_count']} | Pred: {result['predicted_count']}",
                     fontsize=9)
        ax.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    plt.savefig(f'detection_results-{timestamp}.png', dpi=150, bbox_inches='tight')
    print(f"Visualization saved: detection_results-{timestamp}.png")
    plt.close()

def debug_predictions_detailed(model, test_data, loader, num_samples=3):
    """Muestra TODAS las predicciones antes de filtrar por confianza"""
    print("\n" + "="*80)
    print("DEBUG: Predicciones antes de filtrado por confianza")
    print("="*80)
    
    samples = random.sample(test_data, min(num_samples, len(test_data)))
    
    for item in samples:
        img = loader.preprocess_image(item['img_path'], False)
        if img is None:
            continue
        
        preds = model.predict(np.expand_dims(img/255.0, axis=0), verbose=0)[0]
        
        print(f"\n{item['filename']}:")
        print(f"  Ground truth: {len(item['boxes'])} objetos")
        
        # Recoger TODAS las predicciones (sin threshold)
        all_boxes = decode_predictions(preds, conf_threshold=0.0)  # SIN FILTRO
        
        if len(all_boxes) == 0:
            print("  ⚠️ NINGUNA predicción generada (incluso con threshold=0.0)")
            continue
        
        # Ordenar por confianza
        all_boxes_sorted = sorted(all_boxes, key=lambda x: x[4], reverse=True)
        
        print(f"  Total predicciones (threshold=0.0): {len(all_boxes_sorted)}")
        print(f"\n  Top 10 predicciones más confiadas:")
        print(f"  {'#':<3} {'x1':<5} {'y1':<5} {'x2':<5} {'y2':<5} {'Conf':<7} {'Class':<10} {'Size (px)'}")
        print(f"  {'-'*70}")
        
        for i, box in enumerate(all_boxes_sorted[:10]):
            x1, y1, x2, y2, conf, class_id = box
            size = f"{x2-x1}x{y2-y1}"
            class_name = loader.class_names[class_id]
            print(f"  {i+1:<3} {x1:<5} {y1:<5} {x2:<5} {y2:<5} {conf:<7.4f} {class_name:<10} {size}")
        
        # Contar por rangos de confianza
        conf_ranges = {
            '>0.75': sum(1 for b in all_boxes_sorted if b[4] > 0.75),
            '0.5-0.75': sum(1 for b in all_boxes_sorted if 0.5 <= b[4] <= 0.75),
            '0.3-0.5': sum(1 for b in all_boxes_sorted if 0.3 <= b[4] < 0.5),
            '0.1-0.3': sum(1 for b in all_boxes_sorted if 0.1 <= b[4] < 0.3),
            '<0.1': sum(1 for b in all_boxes_sorted if b[4] < 0.1),
        }
        
        print(f"\n  Distribución de confianzas:")
        for range_name, count in conf_ranges.items():
            print(f"    {range_name}: {count}")
        
        # Mostrar tamaños de GT
        print(f"\n  Ground truth boxes (tamaños en imagen modelo):")
        for gt in item['boxes']:
            x1_scaled = int(gt['x1'] * IMG_WIDTH / item['orig_w'])
            y1_scaled = int(gt['y1'] * IMG_HEIGHT / item['orig_h'])
            x2_scaled = int(gt['x2'] * IMG_WIDTH / item['orig_w'])
            y2_scaled = int(gt['y2'] * IMG_HEIGHT / item['orig_h'])
            size = f"{x2_scaled-x1_scaled}x{y2_scaled-y1_scaled}"
            print(f"    {gt['class']}: {size} píxeles")
    
def main():
    """Main training pipeline - FIXED VERSION"""
    print("="*80)
    print("LISA Traffic Light DETECTION + CLASSIFICATION System")
    print("TensorFlow Version:", tf.__version__)
    print(f"Configuration: MAX_SAMPLES={MAX_SAMPLES}, EPOCHS={EPOCHS}, BATCH_SIZE={BATCH_SIZE}")
    print("="*80)
    
    # Load dataset and compute optimal anchors
    print("\n[1/6] Loading ALL dataset (train + test sequences)...")
    loader = LISADetectionDataLoader(DATASET_ROOT, ANNOTATIONS_ROOT)
    all_data = loader.load_dataset()  # ← This now includes EVERYTHING
    print(f"Total images loaded: {len(all_data)}")
    
    # ==========================================
    # KEY CHANGE: Random split instead of using predefined 'split' field
    # ==========================================
    
    print("\n[2/6] Creating train/val/test split from ALL data...")
    
    # First, create dataset metadata (validate images exist)
    all_items = loader.create_dataset(all_data, apply_calibration=True)
    print(f"Valid samples: {len(all_items)}")
    
    # Split: 70% train, 15% val, 15% test
    from sklearn.model_selection import train_test_split
    
    # First split: 70% train+val, 30% test
    train_val_items, test_items = train_test_split(
        all_items, 
        test_size=0.15,  # 15% for test
        random_state=42,
        shuffle=True
    )
    
    # Second split: split train+val into 70% train, 15% val
    train_items, val_items = train_test_split(
        train_val_items,
        test_size=0.176,  # 0.176 * 0.85 ≈ 0.15 of total
        random_state=42,
        shuffle=True
    )
    
    # Disable augmentation for validation and test
    for item in val_items + test_items:
        item['apply_calibration'] = False
    
    print(f"\n✅ Dataset split complete:")
    print(f"  Train: {len(train_items)} ({100*len(train_items)/len(all_items):.1f}%)")
    print(f"  Val:   {len(val_items)} ({100*len(val_items)/len(all_items):.1f}%)")
    print(f"  Test:  {len(test_items)} ({100*len(test_items)/len(all_items):.1f}%)")
    
    # Verify the split includes all sequences
    print("\n📊 Checking data distribution:")
    for split_name, split_data in [("Train", train_items), ("Val", val_items), ("Test", test_items)]:
        sources = {}
        for item in split_data:
            # Extract source from img_dir
            if 'dayTrain' in item['img_dir']:
                source = 'dayTrain'
            elif 'nightTrain' in item['img_dir']:
                source = 'nightTrain'
            elif 'daySequence' in item['img_dir']:
                source = 'daySequence'
            elif 'nightSequence' in item['img_dir']:
                source = 'nightSequence'
            else:
                source = 'unknown'
            sources[source] = sources.get(source, 0) + 1
        
        print(f"  {split_name}: {', '.join([f'{k}:{v}' for k,v in sources.items()])}")
    
    # Create model
    print("\n[3/6] Creating detection model...")
    model = create_detection_model(num_classes=loader.num_classes)
    model.summary()
    
    # Train
    print("\n[4/6] Training model...")
    history = train_model(model, train_items, val_items, loader, loader)

    # Convert to TFLite
    print("\n[5/6] Converting to TFLite...")
    tflite_path = convert_to_tflite(model, train_items, loader)
    
    # Test on the HELD-OUT test set
    print("\n[6/6] Testing on held-out test set...")
    
    # First test Keras model
    print("\n--- Testing Keras Model on Test Set ---")
    keras_test_results = []
    for item in test_items[:30]:  # Test on 30 samples
        img = loader.preprocess_image(item['img_path'], False)
        if img is None:
            continue
        preds = model.predict(np.expand_dims(img/255.0, axis=0), verbose=0)[0]
        boxes = decode_predictions(preds)
        boxes = non_max_suppression(boxes)
        keras_test_results.append({
            'filename': item['filename'],
            'boxes': boxes,
            'gt_boxes': item['boxes'],
            'orig_w': item['orig_w'],
            'orig_h': item['orig_h']
        })
    
    keras_aps, keras_map = evaluate_map(keras_test_results, loader.class_names, verbose=True)
    
    # Then test TFLite model
    print("\n--- Testing TFLite Model on Test Set ---")
    tflite_test_results = test_tflite_model(tflite_path, test_items, loader.class_names, num_samples=30)
    tflite_aps, tflite_map = evaluate_map(tflite_test_results, loader.class_names, verbose=True)

    # Visualize
    visualize_detections(keras_test_results, num_display=6)
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"Keras model mAP (test):  {keras_map:.3f}")
    print(f"TFLite model mAP (test): {tflite_map:.3f}")
    print(f"Model saved: {tflite_path}")
    print("="*80)


if __name__ == "__main__":
    main()