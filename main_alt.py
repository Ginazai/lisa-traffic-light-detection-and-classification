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
IMG_HEIGHT = 480
IMG_WIDTH = 640
GRID_SIZE = 8  # Divide image into 8x8 grid
CELL_HEIGHT = IMG_HEIGHT // GRID_SIZE
CELL_WIDTH = IMG_WIDTH // GRID_SIZE
NUM_ANCHORS = 4
BATCH_SIZE = 8
EPOCHS = 15
LEARNING_RATE = 0.0005 # reduced from 0.001
IOU_THRESHOLD = 0.3
CONF_THRESHOLD = 0.3
MAX_SAMPLES = 10000  # Limit total samples to prevent memory issues

# Dataset paths
DATASET_ROOT = "LISA Traffic Light Dataset"
ANNOTATIONS_ROOT = os.path.join(DATASET_ROOT, "Annotations", "Annotations")

# Anchor boxes (width, height) normalized to cell size
ANCHORS = np.array([
    [0.2, 0.6],   # New smaller anchor for tiny/distant lights
    [0.3, 0.8],   # Tall thin traffic light
    [0.5, 1.2],   # Medium traffic light
    [0.7, 1.5]    # Large traffic light
])


def apply_ov2640_color_calibration(image):
    """Apply color calibration to match OV2640 camera characteristics"""
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
    
    # Add sensor noise
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

            # Build full result dict
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
    def __init__(self, val_data, loader, class_names, iou_threshold=0.5, num_samples=200):
        super().__init__()
        self.val_data = val_data
        self.loader = loader
        self.class_names = class_names
        self.iou_threshold = iou_threshold
        self.num_samples = num_samples

    def on_epoch_end(self, epoch, logs=None):
        # Sample a subset of validation data for speed
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
        
        # Log into Keras history
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
        
    def parse_annotations(self, csv_path):
        """Parse annotation CSV file"""
        df = pd.read_csv(csv_path, delimiter=';')
        
        # Group by filename to handle multiple boxes per image
        grouped = {}
        for _, row in df.iterrows():
            filename = row['Filename']
            
            # Fix filename path - remove any directory prefix
            # Example: "dayTraining/dayClip13--00131.jpg" -> "dayClip13--00131.jpg"
            if '/' in filename:
                filename = filename.split('/')[-1]
            if '\\' in filename:
                filename = filename.split('\\')[-1]
            
            if filename not in grouped:
                grouped[filename] = []
            
            annotation = {
                'x1': int(row['Upper left corner X']),
                'y1': int(row['Upper left corner Y']),
                'x2': int(row['Lower right corner X']),
                'y2': int(row['Lower right corner Y']),
                'class': row['Annotation tag']
            }
            grouped[filename].append(annotation)
        
        return grouped
    
    def load_dataset(self):
        """Load all annotations from dataset"""
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
                                'boxes': boxes,
                                'split': 'train'
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
                                'boxes': boxes,
                                'split': 'train'
                            })
        
        # Test sequences
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
                            'boxes': boxes,
                            'split': 'test'
                        })
        
        return all_data
    
    def encode_ground_truth(self, boxes, img_width, img_height):
        """
        YOLO-style anchor-based encoding
        """
        target = np.zeros((GRID_SIZE, GRID_SIZE, NUM_ANCHORS, 5 + self.num_classes))
        
        for box in boxes:
            if box['class'] not in self.class_to_idx:
                continue
            
            # Normalize coordinates
            x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            box_w = (x2 - x1) / img_width
            box_h = (y2 - y1) / img_height
            
            # Find grid cell
            grid_x = int(x_center * GRID_SIZE)
            grid_y = int(y_center * GRID_SIZE)
            grid_x = min(grid_x, GRID_SIZE - 1)
            grid_y = min(grid_y, GRID_SIZE - 1)
            
            # Find best anchor
            best_anchor = 0
            best_iou = 0
            for anchor_idx, anchor in enumerate(ANCHORS):
                anchor_w = anchor[0] / GRID_SIZE
                anchor_h = anchor[1] / GRID_SIZE
                intersection = min(box_w, anchor_w) * min(box_h, anchor_h)
                union = box_w * box_h + anchor_w * anchor_h - intersection
                iou = intersection / union if union > 0 else 0
                if iou > best_iou:
                    best_iou = iou
                    best_anchor = anchor_idx
            
            # Cell-relative coordinates
            cell_x = x_center * GRID_SIZE - grid_x
            cell_y = y_center * GRID_SIZE - grid_y
            
            # YOLO-style: encode as log(box_size / anchor_size)
            anchor_w = ANCHORS[best_anchor][0] / GRID_SIZE
            anchor_h = ANCHORS[best_anchor][1] / GRID_SIZE
            tw = np.log(box_w / anchor_w + 1e-16)
            th = np.log(box_h / anchor_h + 1e-16)
            
            # Encode target
            class_idx = self.class_to_idx[box['class']]
            target[grid_y, grid_x, best_anchor, 0] = 1.0
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
            img = apply_ov2640_color_calibration(img)
        
        return img
    
    def create_dataset(self, data, apply_calibration=True):
        """Create dataset metadata (paths only, not loaded into memory)"""
        valid_data = []
        
        print(f"Validating {MAX_SAMPLES} samples...")
        
        for idx, item in enumerate(data):
            if idx % 500 == 0:
                print(f"Validated {idx}/{MAX_SAMPLES} samples ({idx / MAX_SAMPLES * 100:.2f}%)")
            
            # Apply MAX_SAMPLES limit
            if len(valid_data) >= MAX_SAMPLES:
                print(f"Reached MAX_SAMPLES limit ({MAX_SAMPLES}), stopping validation")
                break
            
            img_path = os.path.join(item['img_dir'], item['filename'])
            
            # Just check if file exists
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
        
        print(f"Valid samples: {len(valid_data)}")
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

def classification_accuracy(y_true, y_pred):
    cls_true = y_true[..., 5:]
    cls_pred = tf.sigmoid(y_pred[..., 5:])
    cls_true_idx = tf.argmax(cls_true, axis=-1)
    cls_pred_idx = tf.argmax(cls_pred, axis=-1)

    # Only count where an object exists
    obj_mask = tf.cast(y_true[..., 0], tf.float32)  # shape [B, G, G, A]
    correct = tf.cast(tf.equal(cls_true_idx, cls_pred_idx), tf.float32) * tf.cast(obj_mask, tf.float32)

    return tf.reduce_sum(correct) / (tf.reduce_sum(tf.cast(obj_mask, tf.float32)) + 1e-7)

def create_detection_model(num_classes):
    """
    Create lightweight detection model
    Architecture: MobileNetV2 backbone + detection heads
    """
    # Input
    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    
    # Backbone: MobileNetV2 (feature extractor)
    backbone = keras.applications.MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights='imagenet',
        alpha=0.5  # Reduced width for efficiency
    )
    
    # Fine-tune last layers
    backbone.trainable = True
    for layer in backbone.layers[:-20]:
        layer.trainable = False
    
    # Extract features
    x = backbone(inputs, training=False)
    
    # Detection head
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)  # testing
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)  # testing
    
    # Resize to grid
    x = layers.Resizing(GRID_SIZE, GRID_SIZE)(x)
    
    # Final prediction: [GRID_SIZE, GRID_SIZE, NUM_ANCHORS * (5 + NUM_CLASSES)]
    x = layers.Conv2D(NUM_ANCHORS * (5 + num_classes), 1, padding='same')(x)
    
    # Reshape to [GRID_SIZE, GRID_SIZE, NUM_ANCHORS, 5 + NUM_CLASSES]
    outputs = layers.Reshape((GRID_SIZE, GRID_SIZE, NUM_ANCHORS, 5 + num_classes))(x)
    
    model = keras.Model(inputs, outputs)
    
    return model


def detection_loss(y_true, y_pred):
    """
    Custom detection loss function
    Combines: objectness loss + bbox regression loss + classification loss
    """
    # Split predictions
    obj_true = y_true[..., 0:1]  # [B, G, G, A, 1]
    box_true = y_true[..., 1:5]  # [B, G, G, A, 4]
    cls_true = y_true[..., 5:]   # [B, G, G, A, C]
    
    obj_pred = tf.sigmoid(y_pred[..., 0:1])
    box_pred = y_pred[..., 1:5]
    cls_pred = tf.sigmoid(y_pred[..., 5:])
    
    # Object mask (where object exists)
    obj_mask = obj_true
    noobj_mask = 1 - obj_true
    
    # Objectness loss (binary cross-entropy manually computed to preserve shape)
    epsilon = 1e-7
    obj_pred_clipped = tf.clip_by_value(obj_pred, epsilon, 1 - epsilon)
    obj_bce = -(obj_true * tf.math.log(obj_pred_clipped) + (1 - obj_true) * tf.math.log(1 - obj_pred_clipped))
    
    obj_loss = obj_mask * obj_bce
    noobj_loss = noobj_mask * obj_bce
    objectness_loss = tf.reduce_mean(obj_loss + 0.5 * noobj_loss)
    
    # Box regression loss (only where objects exist)
    box_diff = box_true - box_pred
    box_loss = obj_mask * tf.reduce_sum(tf.square(box_diff), axis=-1, keepdims=True)
    box_loss = tf.reduce_sum(box_loss) / (tf.reduce_sum(obj_mask) + epsilon)
    
    # Classification loss (binary cross-entropy for multi-label)
    cls_pred_clipped = tf.clip_by_value(cls_pred, epsilon, 1 - epsilon)
    cls_bce = -(cls_true * tf.math.log(cls_pred_clipped) + (1 - cls_true) * tf.math.log(1 - cls_pred_clipped))
    cls_loss = obj_mask * tf.reduce_mean(cls_bce, axis=-1, keepdims=True)
    cls_loss = tf.reduce_sum(cls_loss) / (tf.reduce_sum(obj_mask) + epsilon)
    
    # Total loss with weights
    # total_loss = 5.0 * box_loss + objectness_loss + cls_loss
    total_loss = 5.0 * box_loss + objectness_loss + 0.5 * cls_loss
    
    return total_loss


def train_model(model, train_data, val_data, train_loader, val_loader):
    """Train detection model using data generators"""
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, weight_decay=1e-5),
        loss=detection_loss,
        metrics=[classification_accuracy]
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
        # MAPCallback(val_data, val_loader, train_loader.class_names),
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
    """
    YOLO-style decoding with anchors
    """
    boxes = []
    
    if isinstance(predictions, tf.Tensor):
        predictions = predictions.numpy()
    
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            for a in range(NUM_ANCHORS):
                # Objectness
                raw_obj = predictions[i, j, a, 0]
                obj_conf = 1 / (1 + np.exp(-np.clip(raw_obj, -10, 10)))
                if obj_conf < conf_threshold:
                    continue
                
                # Cell offsets
                tx = predictions[i, j, a, 1]
                ty = predictions[i, j, a, 2]
                cell_x = 1 / (1 + np.exp(-np.clip(tx, -10, 10)))
                cell_y = 1 / (1 + np.exp(-np.clip(ty, -10, 10)))
                
                # Width/height with anchors
                tw = np.clip(predictions[i, j, a, 3], -10, 10)
                th = np.clip(predictions[i, j, a, 4], -10, 10)
                
                anchor_w = ANCHORS[a][0] / GRID_SIZE
                anchor_h = ANCHORS[a][1] / GRID_SIZE
                
                box_w = anchor_w * np.exp(tw)
                box_h = anchor_h * np.exp(th)
                
                # Convert to image coords
                x_center = (j + cell_x) / GRID_SIZE
                y_center = (i + cell_y) / GRID_SIZE
                
                x1 = int((x_center - box_w / 2) * IMG_WIDTH)
                y1 = int((y_center - box_h / 2) * IMG_HEIGHT)
                x2 = int((x_center + box_w / 2) * IMG_WIDTH)
                y2 = int((y_center + box_h / 2) * IMG_HEIGHT)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                x1 = max(0, min(IMG_WIDTH, x1))
                y1 = max(0, min(IMG_HEIGHT, y1))
                x2 = max(0, min(IMG_WIDTH, x2))
                y2 = max(0, min(IMG_HEIGHT, y2))
                
                # Classes
                class_logits = predictions[i, j, a, 5:]
                class_probs = 1 / (1 + np.exp(-np.clip(class_logits, -10, 10)))
                class_id = np.argmax(class_probs)
                class_conf = class_probs[class_id]
                
                confidence = obj_conf * class_conf
                boxes.append((x1, y1, x2, y2, confidence, int(class_id)))
    
    return boxes
    
def non_max_suppression(boxes, iou_threshold=IOU_THRESHOLD):
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


# def convert_to_tflite(model, output_path='traffic_light_detector.tflite'):
#     """Convert to TFLite with optimizations"""
#     converter = tf.lite.TFLiteConverter.from_keras_model(model)
#     converter.optimizations = [tf.lite.Optimize.DEFAULT]
#     converter.target_spec.supported_types = [tf.float32]
    
#     tflite_model = converter.convert()
    
#     with open(output_path, 'wb') as f:
#         f.write(tflite_model)
    
#     print(f"\nTFLite model saved: {output_path}")
#     print(f"Model size: {len(tflite_model) / 1024:.2f} KB")
    
#     return output_path

def convert_to_tflite(model, train_data, loader, output_path='traffic_light_detector.tflite', num_calibration_samples=250):
    """Convert to TFLite with INT8 quantization"""
    
    # Representative dataset generator for calibration
    def representative_dataset():
        """Generate representative samples for quantization calibration"""
        calibration_samples = random.sample(train_data, min(num_calibration_samples, len(train_data)))
        
        for item in calibration_samples:
            img = loader.preprocess_image(item['img_path'], item['apply_calibration'])
            if img is None:
                continue
            
            # Normalize and add batch dimension
            img_normalized = np.expand_dims(img / 255.0, axis=0).astype(np.float32)
            yield [img_normalized]
    
    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Enable INT8 quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    
    # Enforce full integer quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8  # or tf.int8
    converter.inference_output_type = tf.uint8  # or tf.int8
    
    # Convert model
    print("Converting to INT8 quantized TFLite model...")
    print(f"Using {num_calibration_samples} calibration samples...")
    tflite_model = converter.convert()
    
    # Save model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"\nINT8 TFLite model saved: {output_path}")
    print(f"Model size: {len(tflite_model) / 1024:.2f} KB")
    
    return output_path

def evaluate_map(results, class_names, iou_threshold=0.5, img_width=IMG_WIDTH, img_height=IMG_HEIGHT):
    """Compute mAP per class from detection results with proper scaling and class alignment"""
    aps = {}
    
    # Build class lookup
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    for class_id, class_name in enumerate(class_names):
        preds = []
        gts = []
        
        for result in results:
            orig_w = result.get('orig_w')
            orig_h = result.get('orig_h')
            if orig_w is None or orig_h is None:
                orig_h, orig_w = result['image'].shape[:2]
            
            # Scale GT boxes to resized image space
            for gt in result['gt_boxes']:
                if gt['class'] != class_name:
                    continue
                scaled_gt = {
                    'x1': int(gt['x1'] * img_width / orig_w),
                    'y1': int(gt['y1'] * img_height / orig_h),
                    'x2': int(gt['x2'] * img_width / orig_w),
                    'y2': int(gt['y2'] * img_height / orig_h),
                    'class_id': class_to_idx[gt['class']]
                }
                gts.append(scaled_gt)
            
            # Collect predictions of this class
            preds.extend([b for b in result['boxes'] if b[5] == class_id])

        # Debug logging
        print(f"[DEBUG] Class {class_name}: GT={len(gts)} | Pred={len(preds)}")
        
        if len(gts) == 0:
            continue
        
        # Sort predictions by confidence
        preds = sorted(preds, key=lambda x: x[4], reverse=True)
        
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        matched = []
        
        for i, pred in enumerate(preds):
            best_iou = 0
            best_gt = None
            for gt in gts:
                iou = compute_iou(pred, (gt['x1'], gt['y1'], gt['x2'], gt['y2'], 1.0, gt['class_id']))
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt
            if best_iou >= iou_threshold and best_gt not in matched:
                tp[i] = 1
                matched.append(best_gt)
            else:
                fp[i] = 1
        
        # Precision-recall curve
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recalls = tp_cum / len(gts)
        precisions = tp_cum / (tp_cum + fp_cum + 1e-7)
        
        # AP = area under precision-recall curve
        ap = np.trapz(precisions, recalls)
        aps[class_name] = ap
    
    # Mean AP
    map_score = np.mean(list(aps.values())) if aps else 0.0
    print("\nPer-class AP:")
    for cls, ap in aps.items():
        print(f"  {cls}: {ap:.3f}")
    print(f"\nMean Average Precision (mAP): {map_score:.3f}")
    
    return aps, map_score

def test_tflite_model(tflite_path, test_data, class_names, num_samples=30):
    """Test TFLite detection model (handles both float32 and int8 quantized models)"""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Check model type
    input_dtype = input_details[0]['dtype']
    output_dtype = output_details[0]['dtype']
    is_quantized = input_dtype == np.uint8 or input_dtype == np.int8
    is_float16 = input_dtype == np.float16
    
    if is_quantized:
        print(f"Detected INT8 quantized model")
        print(f"Input type: {input_dtype}, Output type: {output_dtype}")
        
        # Get quantization parameters
        input_scale, input_zero_point = input_details[0]['quantization']
        output_scale, output_zero_point = output_details[0]['quantization']
        print(f"Input scale: {input_scale}, zero_point: {input_zero_point}")
        print(f"Output scale: {output_scale}, zero_point: {output_zero_point}")
    elif is_float16:
        print(f"Detected FLOAT16 model")
        print(f"Input type: {input_dtype}, Output type: {output_dtype}")
    else:
        print(f"Detected FLOAT32 model")
        print(f"Input type: {input_dtype}, Output type: {output_dtype}")
    
    loader = LISADetectionDataLoader(DATASET_ROOT, ANNOTATIONS_ROOT)
    
    # Sample random images
    random_samples = random.sample(test_data, min(num_samples, len(test_data)))
    
    print(f"\n{'='*80}")
    print(f"Testing TFLite Detection Model on {len(random_samples)} samples")
    print(f"{'='*80}\n")
    
    results = []
    
    for idx, item in enumerate(random_samples):
        img = loader.preprocess_image(item['img_path'], item['apply_calibration'])
        
        if img is None:
            continue
        
        # Prepare input based on model type
        if is_quantized:
            # For quantized models: normalize to [0, 255] then quantize
            img_normalized = img.astype(np.float32)  # Already in [0, 255] range
            
            # Quantize: scale and add zero point
            input_data = (img_normalized / input_scale + input_zero_point).astype(input_dtype)
            input_data = np.expand_dims(input_data, axis=0)
        elif is_float16:
            # For float16 models: normalize to [0, 1] and cast to float16
            input_data = np.expand_dims(img / 255.0, axis=0).astype(np.float16)
        else:
            # For float32 models: normalize to [0, 1]
            input_data = np.expand_dims(img / 255.0, axis=0).astype(np.float32)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        
        # Dequantize output if needed
        if is_quantized:
            # Dequantize: (quantized_value - zero_point) * scale
            predictions = (predictions.astype(np.float32) - output_zero_point) * output_scale
        elif is_float16:
            # Convert float16 to float32 for processing
            predictions = predictions.astype(np.float32)
        
        # Decode predictions
        boxes = decode_predictions(predictions)
        boxes = non_max_suppression(boxes)
        
        # Count detections per class
        detections = {}
        for box in boxes:
            class_id = int(box[5])
            class_name = class_names[class_id]
            detections[class_name] = detections.get(class_name, 0) + 1
        
        # Ground truth count
        gt_count = len(item['boxes'])
        pred_count = len(boxes)
        
        results.append({
            'filename': item['filename'],
            'ground_truth_count': gt_count,
            'predicted_count': pred_count,
            'detections': detections,
            'boxes': boxes,
            'image': img,
            'gt_boxes': item['boxes'],
            'orig_w': item['orig_w'],
            'orig_h': item['orig_h']
        })
        
        # Print result
        det_str = ', '.join([f"{k}:{v}" for k, v in detections.items()]) if detections else "none"
        print(f"{idx+1:2d}. {item['filename']:35s} | GT: {gt_count} | Pred: {pred_count} | {det_str}")
    
    print(f"\n{'='*80}\n")
    
    return results
    
    for idx, item in enumerate(random_samples):
        img = loader.preprocess_image(item['img_path'], item['apply_calibration'])
        
        if img is None:
            continue
        
        # Run inference
        input_data = np.expand_dims(img / 255.0, axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        
        # Decode predictions
        boxes = decode_predictions(predictions)
        boxes = non_max_suppression(boxes)
        
        # Count detections per class
        detections = {}
        for box in boxes:
            class_id = int(box[5])
            class_name = class_names[class_id]
            detections[class_name] = detections.get(class_name, 0) + 1
        
        # Ground truth count
        gt_count = len(item['boxes'])
        pred_count = len(boxes)
        
        results.append({
            'filename': item['filename'],
            'ground_truth_count': gt_count,
            'predicted_count': pred_count,
            'detections': detections,
            'boxes': boxes,
            'image': img,
            'gt_boxes': item['boxes']
        })
        
        # Print result
        det_str = ', '.join([f"{k}:{v}" for k, v in detections.items()]) if detections else "none"
        print(f"{idx+1:2d}. {item['filename']:35s} | GT: {gt_count} | Pred: {pred_count} | {det_str}")
    
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
        
        # Get ORIGINAL dimensions (before resize)
        orig_w = result.get('orig_w')
        orig_h = result.get('orig_h')
        
        if orig_w is None or orig_h is None:
            # Fallback: assume standard LISA dimensions
            orig_w, orig_h = 1280, 960
        
        # Scale factors: original -> resized
        scale_x = IMG_WIDTH / orig_w   # e.g., 640/1280 = 0.5
        scale_y = IMG_HEIGHT / orig_h  # e.g., 480/960 = 0.5
        
        # Draw ground truth (green) - boxes are in ORIGINAL coordinates
        for box in result['gt_boxes']:
            x1 = int(box['x1'] * scale_x)
            y1 = int(box['y1'] * scale_y)
            x2 = int(box['x2'] * scale_x)
            y2 = int(box['y2'] * scale_y)
            
            # Clip to image bounds
            x1 = max(0, min(IMG_WIDTH, x1))
            y1 = max(0, min(IMG_HEIGHT, y1))
            x2 = max(0, min(IMG_WIDTH, x2))
            y2 = max(0, min(IMG_HEIGHT, y2))
            
            w, h = x2 - x1, y2 - y1
            rect = patches.Rectangle((x1, y1), w, h, linewidth=2,
                                     edgecolor='green', facecolor='none', label='GT')
            ax.add_patch(rect)
        
        # Draw predictions (red) - already in resized coordinates
        for box in result['boxes']:
            x1, y1, x2, y2, conf, class_id = box
            x1 = max(0, min(IMG_WIDTH, x1))
            y1 = max(0, min(IMG_HEIGHT, y1))
            x2 = max(0, min(IMG_WIDTH, x2))
            y2 = max(0, min(IMG_HEIGHT, y2))
            
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


def main():
    """Main training pipeline"""
    print("="*80)
    print("LISA Traffic Light DETECTION + CLASSIFICATION System")
    print("TensorFlow Version:", tf.__version__)
    print(f"MAX_SAMPLES: {MAX_SAMPLES}")
    print("="*80)
    
    # Load dataset
    print("\n[1/6] Loading dataset...")
    loader = LISADetectionDataLoader(DATASET_ROOT, ANNOTATIONS_ROOT)
    all_data = loader.load_dataset()
    print(f"Total images: {len(all_data)}")
    
    # Split data
    train_data = [d for d in all_data if d['split'] == 'train']
    test_data = [d for d in all_data if d['split'] == 'test']
    
    print(f"Training images: {len(train_data)}")
    print(f"Test images: {len(test_data)}")
    
    # Create dataset metadata (not loading images yet)
    print("\n[2/6] Validating dataset...")
    train_items = loader.create_dataset(train_data, apply_calibration=True)
    test_items = loader.create_dataset(test_data, apply_calibration=True)
    
    # Split training into train/val
    from sklearn.model_selection import train_test_split
    train_items, val_items = train_test_split(
        train_items, test_size=0.2, random_state=42
    )
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_items)}")
    print(f"  Val: {len(val_items)}")
    print(f"  Test: {len(test_items)}")
    
    # Create model
    print("\n[3/6] Creating detection model...")
    model = create_detection_model(num_classes=loader.num_classes)
    model.summary()
    
    # Train using generators (memory efficient)
    print("\n[4/6] Training model (using data generators)...")
    history = train_model(model, train_items, val_items, loader, loader)

    # Test Keras model directly
    print("\n[Testing Keras model directly]")
    test_item = random.choice(test_items)
    img = loader.preprocess_image(test_item['img_path'], True)
    preds = model.predict(np.expand_dims(img/255.0, axis=0))[0]
    boxes = decode_predictions(preds, conf_threshold=0.3)
    print(f"Keras model found {len(boxes)} boxes")
    
    # Convert to TFLite
    print("\n[5/6] Converting to TFLite...")
    # tflite_path = convert_to_tflite(model)
    tflite_path = convert_to_tflite(model, train_items, loader)
    
    # Test
    print("\n[6/6] Testing detection model...")
    results = test_tflite_model(tflite_path, test_items, loader.class_names, num_samples=30)
    aps, map_score = evaluate_map(results, loader.class_names)

    # Visualize
    visualize_detections(results)
    
    print("\n" + "="*80)
    print("Training complete!")
    print(f"Model saved: {tflite_path}")
    print("="*80)


if __name__ == "__main__":
    main()