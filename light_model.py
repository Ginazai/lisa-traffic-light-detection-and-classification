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
IMG_HEIGHT = 240
IMG_WIDTH = 320
GRID_SIZE = 8  # Divide image into 16x16 grid
CELL_HEIGHT = IMG_HEIGHT // GRID_SIZE
CELL_WIDTH = IMG_WIDTH // GRID_SIZE
NUM_ANCHORS = 4
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 0.00025
IOU_THRESHOLD_EVAL = 0.25 
IOU_THRESHOLD_NMS = 0.2    # NMS agresivo
CONF_THRESHOLD = 0.75 # 0.75 reduced for testing
MAX_SAMPLES = 36775 # Total Lisa Traffic Light Dataset samples (36775 total)

# Knowledge Distillation & QAT config
DO_DISTILL = True           # Run teacher->student distillation after baseline training
DISTILL_TEMPERATURE = 2.0   # Softening temperature for distillation
DISTILL_ALPHA = 0.5         # Weighting between hard loss and soft (0..1)
STUDENT_ALPHA = 0.35        # MobileNet alpha for student
STUDENT_HEAD1 = 32          # Student head conv filters
STUDENT_HEAD2 = 16
DO_QAT = True               # Apply QAT fine-tuning to the student
QAT_EPOCHS = 5              # Short fine-tune epochs with QAT

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
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)
        self.anchors = None  # Will be computed dynamically
        
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
                matched_label = raw_tag

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
        # Return cached data if already loaded (avoid recomputing anchors repeatedly)
        if getattr(self, '_all_data', None) is not None:
            return self._all_data
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
    
    def preprocess_image(self, img_path, apply_calibration=True, boxes=None):
        """Load and preprocess full image
        Note: 'boxes' is optional and used only by some augmentations (copy-paste)."""
        img = cv2.imread(img_path)
        
        if img is None:
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = random_scale_and_crop(img, IMG_HEIGHT, IMG_WIDTH)

        # Prepare local boxes for augmentations (caller may not pass GT boxes)
        boxes_local = boxes if boxes is not None else []

        if apply_calibration:
            img = apply_ov2640_color_calibration(img, add_noise=True)
            img = photometric(img)
            # Random brightness jitter and copy-paste augment
            if random.random() < 0.5:
                img, boxes_local = copy_paste_augment(img, boxes_local, self)
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
        # Cache validated dataset to speed up sampling for augmentations
        self._valid_data = valid_data
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
    """Create lightweight detection model with MobileNetV2 backbone"""
    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    backbone = keras.applications.MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights='imagenet',
        alpha=0.35
    )
    backbone.trainable = True
    for layer in backbone.layers[:-30]:
        layer.trainable = False

    x = backbone(inputs, training=False)

    reg = keras.regularizers.l2(5e-4)
    x = layers.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu', kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    # Resize feature map to GRID_SIZE × GRID_SIZE
    x = layers.Resizing(GRID_SIZE, GRID_SIZE)(x)

    # Prediction head
    x = layers.Conv2D(NUM_ANCHORS * (5 + num_classes), 1, padding='same')(x)

    # Reshape to [GRID_SIZE, GRID_SIZE, NUM_ANCHORS, 5 + num_classes]
    outputs = layers.Reshape((GRID_SIZE, GRID_SIZE, NUM_ANCHORS, 5 + num_classes))(x)

    return keras.Model(inputs, outputs)

def detection_loss(y_true, y_pred):
    """Custom detection loss function combining objectness, bbox, and classification losses"""
    epsilon = 1e-7
    # Class weights SUAVIZADOS con raíz cuadrada
    raw_weights = np.array([
        1.3,   # go
        1.5,   # goLeft (era 2.0)
        1.3,   # stop  
        1.3,   # stopLeft (era 2.0)
        3.0,   # warning (era 8.0)
        4.0    # warningLeft (era 16.0)
    ])
    
    # Suaviza con sqrt para reducir extremos
    class_weights = tf.constant(np.sqrt(raw_weights), dtype=tf.float32)
    # Resultado: [1.0, 4.11, 1.07, 1.57, 2.86, 4.81]
    
    # Split predictions / ground truth
    obj_true = y_true[..., 0:1]
    box_true = y_true[..., 1:5]
    cls_true = y_true[..., 5:]

    # Predicted: apply sigmoid for objectness and class logits
    obj_pred = tf.sigmoid(y_pred[..., 0:1])
    cls_pred = tf.sigmoid(y_pred[..., 5:])

    # For box predictions: tx,ty should be sigmoided
    tx_ty_pred = tf.sigmoid(y_pred[..., 1:3])
    tw_th_pred = y_pred[..., 3:5]
    box_pred = tf.concat([tx_ty_pred, tw_th_pred], axis=-1)

    # Masks
    obj_mask = obj_true
    noobj_mask = 1 - obj_true

    # Objectness BCE
    obj_pred_clipped = tf.clip_by_value(obj_pred, epsilon, 1 - epsilon)
    obj_bce = -(obj_true * tf.math.log(obj_pred_clipped) + (1 - obj_true) * tf.math.log(1 - obj_pred_clipped))
    
    # CAMBIO CLAVE: Aumenta el peso de los positivos
    obj_loss = obj_mask * obj_bce * 5.0  # Peso extra para objectness positivo
    noobj_loss = noobj_mask * obj_bce * 0.5  # Reduce peso de negativos
    objectness_loss = tf.reduce_mean(obj_loss + noobj_loss)

    # Box regression (sin cambios)
    box_diff = box_true - box_pred
    box_loss = obj_mask * tf.reduce_sum(tf.square(box_diff), axis=-1, keepdims=True)
    box_loss = tf.reduce_sum(box_loss) / (tf.reduce_sum(obj_mask) + epsilon)

    # Classification BCE WITH WEIGHTS
    cls_pred_clipped = tf.clip_by_value(cls_pred, epsilon, 1 - epsilon)
    cls_bce = -(cls_true * tf.math.log(cls_pred_clipped) + 
                (1 - cls_true) * tf.math.log(1 - cls_pred_clipped))
    
    # Apply per-class weights
    weighted_cls_bce = cls_bce * tf.expand_dims(class_weights, axis=0)
    cls_loss = obj_mask * tf.reduce_mean(weighted_cls_bce, axis=-1, keepdims=True)
    cls_loss = tf.reduce_sum(cls_loss) / (tf.reduce_sum(obj_mask) + epsilon)
    
    # Aumenta peso de box loss por IoU bajo
    total_loss = 3.5 * box_loss + 2.0 * objectness_loss + 1.5 * cls_loss
    
    return total_loss


def train_model(model, train_data, val_data, train_loader, val_loader):
    """Train detection model using data generators"""
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, weight_decay=1e-5),
        loss=detection_loss,
        metrics=[lambda y_t, y_p: classification_metrics(y_t, y_p)['cls_precision']]
    )
    
    # Calculate steps per epoch
    train_steps = len(train_data) // BATCH_SIZE
    val_steps = len(val_data) // BATCH_SIZE
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
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


# -------------------------
# Knowledge Distillation
# -------------------------
try:
    import tensorflow_model_optimization as tfmot
except Exception:
    tfmot = None


def build_student_model(num_classes, alpha=STUDENT_ALPHA, head1=STUDENT_HEAD1, head2=STUDENT_HEAD2):
    """Smaller student detection model"""
    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    backbone = keras.applications.MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights='imagenet',
        alpha=alpha
    )
    backbone.trainable = False

    x = backbone(inputs, training=False)
    reg = keras.regularizers.l2(5e-4)
    x = layers.Conv2D(head1, 3, padding='same', activation='relu', kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Conv2D(head2, 3, padding='same', activation='relu', kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Resizing(GRID_SIZE, GRID_SIZE)(x)
    x = layers.Conv2D(NUM_ANCHORS * (5 + num_classes), 1, padding='same')(x)
    outputs = layers.Reshape((GRID_SIZE, GRID_SIZE, NUM_ANCHORS, 5 + num_classes))(x)

    return keras.Model(inputs, outputs)


def distillation_train(teacher, student, train_data, val_data, loader,
                       temperature=DISTILL_TEMPERATURE, alpha=DISTILL_ALPHA,
                       epochs=10, batch_size=BATCH_SIZE):
    """Custom distillation training loop for detection models.
    Student learns from teacher predictions (soft targets) and ground-truth (hard targets).
    """
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # Create generators
    train_gen = loader.data_generator(train_data, batch_size, shuffle=True)
    steps_per_epoch = len(train_data) // batch_size
    val_gen = loader.data_generator(val_data, batch_size, shuffle=False)
    val_steps = len(val_data) // batch_size

    # Metrics
    train_loss_metric = keras.metrics.Mean(name='train_loss')
    val_loss_metric = keras.metrics.Mean(name='val_loss')

    kl = keras.losses.KLDivergence()

    for epoch in range(epochs):
        print(f"\n=== Distillation epoch {epoch+1}/{epochs} ===")
        train_loss_metric.reset_state()

        for step in range(steps_per_epoch):
            imgs, y_true = next(train_gen)
            # Teacher predictions (soft targets)
            teacher_preds = teacher.predict_on_batch(imgs)

            with tf.GradientTape() as tape:
                student_preds = student(imgs, training=True)
                # Hard loss (detection loss wrt GT)
                hard_loss = detection_loss(y_true, student_preds)

                # Soft loss: compute KL on class logits only (slice last channels [5:])
                t = temperature
                teacher_cls_logits = teacher_preds[..., 5:]
                student_cls_logits = student_preds[..., 5:]
                # apply softmax per-anchor over classes
                teacher_soft = tf.nn.softmax(teacher_cls_logits / t, axis=-1)
                student_soft = tf.nn.softmax(student_cls_logits / t, axis=-1)
                # flatten anchors into vectors: shape [batch, -1, C]
                teacher_flat = tf.reshape(teacher_soft, [-1, tf.shape(teacher_soft)[-1]])
                student_flat = tf.reshape(student_soft, [-1, tf.shape(student_soft)[-1]])
                soft_loss = tf.reduce_mean(kl(teacher_flat, student_flat)) * (t * t)

                loss = alpha * hard_loss + (1.0 - alpha) * soft_loss

            grads = tape.gradient(loss, student.trainable_variables)
            optimizer.apply_gradients(zip(grads, student.trainable_variables))
            train_loss_metric.update_state(loss)

            if step % 50 == 0:
                print(f"Step {step}/{steps_per_epoch} - loss: {train_loss_metric.result().numpy():.4f}")

        # Validation loop (simple average loss)
        val_loss_metric.reset_state()
        for v in range(val_steps):
            imgs_v, yv = next(val_gen)
            sp = student.predict_on_batch(imgs_v)
            lv = detection_loss(yv, sp)
            val_loss_metric.update_state(lv)

        print(f"Epoch {epoch+1}: train_loss={train_loss_metric.result().numpy():.4f}, val_loss={val_loss_metric.result().numpy():.4f}")

    return student


def apply_qat(student, train_data, loader, epochs=QAT_EPOCHS):
    """Apply Quantization-Aware Training (QAT) to the student model using TF-MOT.
    This requires `tensorflow_model_optimization` to be available.
    """
    if tfmot is None:
        raise RuntimeError("tensorflow_model_optimization not available. Install it to use QAT.")

    print("Applying quantization-aware training (QAT) wrapper to student model...")
    # Add QAT wrapper
    qat_model = tfmot.quantization.keras.quantize_model(student)

    qat_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE * 0.5),
        loss=detection_loss,
        metrics=[lambda y_t, y_p: classification_metrics(y_t, y_p)['cls_precision']]
    )

    train_gen = loader.data_generator(train_data, BATCH_SIZE, shuffle=True)
    steps_per_epoch = len(train_data) // BATCH_SIZE

    qat_model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=1
    )

    return qat_model


def decode_predictions(predictions, conf_threshold=CONF_THRESHOLD):
    """YOLO-style decoding with anchors"""
    boxes = []
    
    if isinstance(predictions, tf.Tensor):
        predictions = predictions.numpy()
    
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            for a in range(NUM_ANCHORS):
                # Objectness confidence
                obj_logit = np.clip(predictions[i, j, a, 0], -10, 10)
                obj_conf = 1 / (1 + np.exp(-obj_logit))
                
                if obj_conf < conf_threshold:
                    continue
                
                # Cell-relative offsets [0,1]
                tx = np.clip(predictions[i, j, a, 1], -10, 10)
                ty = np.clip(predictions[i, j, a, 2], -10, 10)
                cell_x = 1 / (1 + np.exp(-tx))
                cell_y = 1 / (1 + np.exp(-ty))
                
                # Box size relative to anchor
                tw = np.clip(predictions[i, j, a, 3], -10, 10)
                th = np.clip(predictions[i, j, a, 4], -10, 10)
                
                # Use anchors as image fractions
                anchor_w = ANCHORS[a][0]
                anchor_h = ANCHORS[a][1]
                
                box_w = anchor_w * np.exp(tw)
                box_h = anchor_h * np.exp(th)
                
                # Convert to image coordinates
                x_center = (j + cell_x) / GRID_SIZE
                y_center = (i + cell_y) / GRID_SIZE
                
                # Convert to pixel coordinates
                x1 = int((x_center - box_w / 2) * IMG_WIDTH)
                y1 = int((y_center - box_h / 2) * IMG_HEIGHT)
                x2 = int((x_center + box_w / 2) * IMG_WIDTH)
                y2 = int((y_center + box_h / 2) * IMG_HEIGHT)
                
                # Clip to image bounds
                x1 = max(0, min(IMG_WIDTH, x1))
                y1 = max(0, min(IMG_HEIGHT, y1))
                x2 = max(0, min(IMG_WIDTH, x2))
                y2 = max(0, min(IMG_HEIGHT, y2))
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Class prediction
                class_logits = np.clip(predictions[i, j, a, 5:], -10, 10)
                class_probs = 1 / (1 + np.exp(-class_logits))
                class_id = np.argmax(class_probs)
                class_conf = class_probs[class_id]
                
                confidence = obj_conf * class_conf
                boxes.append((x1, y1, x2, y2, confidence, int(class_id)))
    
    return boxes
    
def non_max_suppression(boxes, iou_threshold=IOU_THRESHOLD_NMS ):
    """Apply NMS to remove overlapping boxes"""
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
            if iou < iou_threshold:
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
    
    # Enforce full integer quantization (signed int8) and export anchors
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Use signed int8 to match firmware int8 handling
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    print(f"Converting to INT8 (signed) quantized TFLite (using {num_calibration_samples} calibration samples)...")
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

def evaluate_map(results, class_names, iou_threshold=IOU_THRESHOLD_EVAL , img_width=IMG_WIDTH, img_height=IMG_HEIGHT, verbose=True, debug=False):
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
        
        # Debugging: compute confidence and area stats, anchors info
        if debug or (verbose and result_idx < 5):
            confs = np.array([b[4] for b in pred_boxes]) if pred_boxes else np.array([])
            areas = np.array([(b[2]-b[0])*(b[3]-b[1]) for b in pred_boxes]) if pred_boxes else np.array([])
            print(f"\n--- Image {result_idx + 1}: {result.get('filename', 'unknown')} ---")
            print(f"GT boxes: {len(scaled_gts)}, Predictions: {len(pred_boxes)}")
            print(f"Original size: {orig_w}x{orig_h}, Model size: {img_width}x{img_height}")
            if pred_boxes:
                try:
                    print(f"  conf: min={confs.min():.4f} mean={confs.mean():.4f} max={confs.max():.4f} counts>0.05:{(confs>0.05).sum()} >0.25:{(confs>0.25).sum()}")
                    print(f"  area: min={areas.min():.1f} mean={areas.mean():.1f} max={areas.max():.1f} (pixels)")
                except Exception:
                    print("  (could not compute conf/area stats)")
                x1s = np.array([b[0] for b in pred_boxes])
                y1s = np.array([b[1] for b in pred_boxes])
                x2s = np.array([b[2] for b in pred_boxes])
                y2s = np.array([b[3] for b in pred_boxes])
                oob_x = np.sum((x1s<0) | (x2s>img_width))
                oob_y = np.sum((y1s<0) | (y2s>img_height))
                print(f"  out_of_bounds: x={oob_x}, y={oob_y}")
                print("  sample preds:", pred_boxes[:5])
            else:
                print("  ⚠️ NO PREDICTIONS for this image")
            # Print anchors if available
            try:
                if ANCHORS is not None:
                    print(f"  Anchors (fractions): {ANCHORS}")
                    anchors_px = np.array(ANCHORS) * np.array([img_width, img_height])
                    print(f"  Anchors (px): {anchors_px.astype(int).tolist()}")
            except Exception as e:
                print("  (Could not print anchors):", e)
        
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
def random_scale_and_crop(img, out_h, out_w, scale_min=0.9, scale_max=1.15, p=0.6):
    if random.random() > p:
        return cv2.resize(img, (out_w, out_h))
    scale = np.random.uniform(scale_min, scale_max)
    h, w = img.shape[:2]
    img_s = cv2.resize(img, (max(1, int(w*scale)), max(1, int(h*scale))))
    ch, cw = img_s.shape[:2]
    # center-crop
    start_x = max(0, (cw - out_w)//2)
    start_y = max(0, (ch - out_h)//2)
    crop = img_s[start_y:start_y+out_h, start_x:start_x+out_w]
    if crop.shape[0] != out_h or crop.shape[1] != out_w:
        crop = cv2.resize(img, (out_w, out_h))
    return crop
def copy_paste_augment(img, boxes, loader, max_pastes=2, prob=0.4, max_scale=0.5):
    # boxes = list of dicts with x1,y1,x2,y2,class
    if random.random() > prob:
        return img, boxes
    # pick random other sample (use cached valid dataset if available to avoid recomputing anchors)
    if getattr(loader, '_valid_data', None) is None:
        # Build a cached valid dataset without augmentation (done once)
        loader._valid_data = loader.create_dataset(loader.load_dataset(), apply_calibration=False)
    sample = random.choice(loader._valid_data)
    src_img = cv2.imread(sample['img_path'])
    if src_img is None: return img, boxes
    src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
    # pick a random GT box in src
    if not sample['boxes']: return img, boxes
    sbox = random.choice(sample['boxes'])
    x1,y1,x2,y2 = sbox['x1'], sbox['y1'], sbox['x2'], sbox['y2']
    patch = src_img[y1:y2, x1:x2]
    # scale patch to small area relative to target
    h, w = img.shape[:2]

    # Guard against invalid patch
    if patch is None or patch.size == 0 or patch.shape[0] == 0 or patch.shape[1] == 0:
        return img, boxes

    scale_factor = random.uniform(0.03, max_scale)  # target area fraction
    target_w = max(4, int(w * scale_factor))
    # Keep aspect ratio of the patch
    aspect = float(patch.shape[0]) / (patch.shape[1] + 1e-6)
    target_h = max(4, int(h * scale_factor * aspect))

    # Ensure target fits inside the image; if too large, scale down preserving aspect
    if target_w > w or target_h > h:
        scale_down = min(w / target_w, h / target_h)
        target_w = max(4, int(target_w * scale_down))
        target_h = max(4, int(target_h * scale_down))

    # Resize patch safely
    patch = cv2.resize(patch, (target_w, target_h))

    # paste at random location (safe: ensure px+target_w <= w and py+target_h <= h)
    max_x = max(0, w - target_w)
    max_y = max(0, h - target_h)
    px = random.randint(0, max_x)
    py = random.randint(0, max_y)

    img_copy = img.copy()
    img_copy[py:py+target_h, px:px+target_w] = patch

    # optionally append box (in image coords)
    new_box = {'x1': px, 'y1': py, 'x2': px+target_w, 'y2': py+target_h, 'class': sbox['class']}
    boxes = boxes + [new_box]
    return img_copy, boxes
def photometric(img):
    # brightness
    if random.random() < 0.5:
        f = 1.0 + np.random.uniform(-0.18, 0.18)
        img = np.clip(img.astype(np.float32) * f, 0, 255).astype(np.uint8)
    # contrast
    if random.random() < 0.5:
        alpha = 1.0 + np.random.uniform(-0.15, 0.15)
        img = np.clip(127 + alpha * (img - 127), 0, 255).astype(np.uint8)
    # small gaussian noise
    if random.random() < 0.4:
        noise = np.random.normal(0, 3, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img
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
            'ground_truth_count': len(item['boxes']),
            'predicted_count': len(boxes),
            'gt_boxes': item['boxes'],
            'image': img,
            'orig_w': item['orig_w'],
            'orig_h': item['orig_h']
        })
    
    keras_aps, keras_map = evaluate_map(keras_test_results, loader.class_names, verbose=True)
    if keras_map == 0.0:
        print("\nNo mAP detected for Keras model — re-running evaluation with debug=True for diagnostics")
        evaluate_map(keras_test_results, loader.class_names, verbose=True, debug=True)
    
    # Then test TFLite model
    print("\n--- Testing TFLite Model on Test Set ---")
    tflite_test_results = test_tflite_model(tflite_path, test_items, loader.class_names, num_samples=30)
    tflite_aps, tflite_map = evaluate_map(tflite_test_results, loader.class_names, verbose=True)
    if tflite_map == 0.0:
        print("\nNo mAP detected for TFLite model — re-running evaluation with debug=True for diagnostics")
        evaluate_map(tflite_test_results, loader.class_names, verbose=True, debug=True)

    # Optionally perform distillation -> student -> QAT flow
    student_model = None
    if DO_DISTILL:
        print("\n=== Starting distillation (teacher -> student) ===")
        # Build student
        student_model = build_student_model(num_classes=loader.num_classes)
        student_model.summary()
        # Distillation training
        student_model = distillation_train(model, student_model, train_items, val_items, loader,
                                           temperature=DISTILL_TEMPERATURE, alpha=DISTILL_ALPHA,
                                           epochs=10, batch_size=BATCH_SIZE)

        # Optional QAT
        if DO_QAT:
            print("\n=== Applying QAT to student model ===")
            if tfmot is None:
                print("⚠ tensorflow_model_optimization not installed; skipping QAT")
            else:
                student_model = apply_qat(student_model, train_items, loader, epochs=QAT_EPOCHS)

        # Convert the student model to TFLite
        student_tflite_path = convert_to_tflite(student_model, train_items, loader, output_path='traffic_light_detector_student.tflite')
        print("\n--- Testing STUDENT TFLite Model on Test Set ---")
        student_test_results = test_tflite_model(student_tflite_path, test_items, loader.class_names, num_samples=30)
        student_aps, student_map = evaluate_map(student_test_results, loader.class_names, verbose=True)
        print(f"Student TFLite mAP: {student_map:.3f}")

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