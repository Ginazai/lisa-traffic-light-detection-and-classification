"""
Traffic Light Detection and Classification using MobileNetV2
For LISA Traffic Light Dataset -> TFLite -> ESP32-S3 with OV2640
"""

import os
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ============================================================================
# ENVIRONMENT SETUP INSTRUCTIONS
# ============================================================================
"""
1. Create and activate virtual environment:
   python3 -m venv tf_env
   source tf_env/bin/activate  # On Windows: tf_env\Scripts\activate

2. Install dependencies:
   pip install --upgrade pip
   pip install tensorflow==2.15.0
   pip install numpy==1.24.3
   pip install pandas==2.0.3
   pip install opencv-python==4.8.1.78
   pip install scikit-learn==1.3.2
   pip install matplotlib==3.7.3
   pip install pillow==10.0.1

3. Verify installation:
   python -c "import tensorflow as tf; print(tf.__version__)"
"""

# ============================================================================
# CONFIGURATION
# ============================================================================
DATASET_PATH = "LISA Traffic Light Dataset"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_PHASE1 = 30
EPOCHS_PHASE2 = 40
LEARNING_RATE = 0.0005  # Changed from 0.001

# Increased limits for better training
MAX_IMAGES_PER_CLIP = 500  # Increased from 200
MAX_TEST_IMAGES = 10000    # Increased from 5000

CLASS_MAP = {
    'go': 0, 'goForward': 0, 'goLeft': 0,
    'warning': 1, 'warningLeft': 1,
    'stop': 2, 'stopLeft': 2
}

CLASS_NAMES = ['green', 'yellow', 'red']

# ============================================================================
# IMPROVED PREPROCESSING
# ============================================================================

def preprocess_images_for_ov2640(images, severity=0.3):
    """
    Lighter OV2640 simulation - less aggressive preprocessing
    severity: 0.0-1.0, controls intensity of effects
    """
    processed = []
    
    for img in images:
        img_float = img.astype(np.float32)
        
        # Gentler color cast
        img_float[:,:,0] = np.clip(img_float[:,:,0] * (1.0 + 0.05 * severity), 0, 255)
        img_float[:,:,2] = np.clip(img_float[:,:,2] * (1.0 - 0.05 * severity), 0, 255)
        
        # Gentler gamma
        img_float = np.power(img_float / 255.0, 1.0 - 0.05 * severity) * 255.0
        
        # Reduced saturation boost
        img_hsv = cv2.cvtColor(img_float.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        img_hsv[:,:,1] = np.clip(img_hsv[:,:,1] * (1.0 + 0.1 * severity), 0, 255)
        img_float = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
        
        # Much lighter noise
        noise = np.random.normal(0, 1.0 * severity, img_float.shape)
        noise = noise * (img_float / 255.0) * 0.5
        img_float = np.clip(img_float + noise, 0, 255)
        
        processed.append(img_float / 255.0)
    
    return np.array(processed)


def load_annotations(csv_path):
    """Load BULB annotations from CSV"""
    df = pd.read_csv(csv_path, delimiter=';')
    return df


def parse_train_dataset(base_path, train_dir, clips):
    """Parse training dataset with improved error handling"""
    images = []
    labels = []
    
    print(f"Loading {train_dir} clips...")
    
    for clip in clips:
        annot_path = os.path.join(base_path, 'Annotations', 'Annotations', train_dir, clip, 'frameAnnotationsBULB.csv')
        
        if not os.path.exists(annot_path):
            print(f"  Warning: {annot_path} not found, skipping...")
            continue
        
        df = load_annotations(annot_path)
        
        # Sample if too many images
        if len(df) > MAX_IMAGES_PER_CLIP:
            df = df.sample(n=MAX_IMAGES_PER_CLIP, random_state=42)
        
        clip_processed = 0
        
        for idx, row in df.iterrows():
            filename = row['Filename']
            annotation = row['Annotation tag']
            
            if annotation not in CLASS_MAP:
                continue
            
            img_filename = os.path.basename(filename)
            img_path = os.path.join(base_path, train_dir, train_dir, clip, 'frames', img_filename)
            
            if not os.path.exists(img_path):
                continue
            
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            x1, y1 = int(row['Upper left corner X']), int(row['Upper left corner Y'])
            x2, y2 = int(row['Lower right corner X']), int(row['Lower right corner Y'])
            
            # Add padding around bounding box
            h, w = img.shape[:2]
            pad = 5
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)
            
            cropped = img[y1:y2, x1:x2]
            
            if cropped.size == 0 or cropped.shape[0] < 10 or cropped.shape[1] < 10:
                continue
            
            resized = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))
            
            images.append(resized)
            labels.append(CLASS_MAP[annotation])
            clip_processed += 1
        
        if clip_processed > 0:
            print(f"  {clip}: {clip_processed} images")
    
    return np.array(images), np.array(labels)


def parse_sequence_dataset(base_path, sequences):
    """Parse sequence data"""
    images = []
    labels = []
    
    for seq in sequences:
        annot_path = os.path.join(base_path, 'Annotations', 'Annotations', seq, 'frameAnnotationsBULB.csv')
        
        if not os.path.exists(annot_path):
            print(f"Warning: {annot_path} not found, skipping...")
            continue
        
        print(f"Loading {seq}...")
        df = load_annotations(annot_path)
        
        processed = 0
        
        for idx, row in df.iterrows():
            if len(images) >= MAX_TEST_IMAGES:
                break
                
            filename = row['Filename']
            annotation = row['Annotation tag']
            
            if annotation not in CLASS_MAP:
                continue
            
            img_filename = os.path.basename(filename)
            img_path = os.path.join(base_path, seq, seq, 'frames', img_filename)
            
            if not os.path.exists(img_path):
                continue
            
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            x1, y1 = int(row['Upper left corner X']), int(row['Upper left corner Y'])
            x2, y2 = int(row['Lower right corner X']), int(row['Lower right corner Y'])
            
            # Add padding
            h, w = img.shape[:2]
            pad = 5
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)
            
            cropped = img[y1:y2, x1:x2]
            if cropped.size == 0 or cropped.shape[0] < 10 or cropped.shape[1] < 10:
                continue
            
            resized = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))
            
            images.append(resized)
            labels.append(CLASS_MAP[annotation])
            processed += 1
        
        if processed > 0:
            print(f"  Processed: {processed} images")
    
    return np.array(images), np.array(labels)


def balance_dataset(X, y, method='hybrid'):
    """
    Balance dataset using different strategies
    
    Methods:
    - 'oversample': Duplicate minority class samples
    - 'undersample': Reduce majority class samples
    - 'hybrid': Oversample minority + slight undersample majority
    - 'augment': Use SMOTE-like augmentation for minority
    """
    from collections import Counter
    
    class_counts = Counter(y)
    print(f"\nOriginal distribution: {dict(class_counts)}")
    
    if method == 'oversample':
        # Oversample minority classes to match majority
        max_count = max(class_counts.values())
        
        X_balanced = []
        y_balanced = []
        
        for class_id in range(3):
            class_indices = np.where(y == class_id)[0]
            class_samples = X[class_indices]
            
            # Duplicate samples to reach max_count
            repeats = max_count // len(class_samples)
            remainder = max_count % len(class_samples)
            
            X_balanced.append(np.tile(class_samples, (repeats, 1, 1, 1)))
            y_balanced.append(np.tile([class_id], repeats * len(class_samples)))
            
            if remainder > 0:
                random_indices = np.random.choice(len(class_samples), remainder, replace=False)
                X_balanced.append(class_samples[random_indices])
                y_balanced.append(np.tile([class_id], remainder))
        
        X_balanced = np.concatenate(X_balanced)
        y_balanced = np.concatenate(y_balanced)
        
    elif method == 'undersample':
        # Undersample majority classes to match minority
        min_count = min(class_counts.values())
        
        X_balanced = []
        y_balanced = []
        
        for class_id in range(3):
            class_indices = np.where(y == class_id)[0]
            selected_indices = np.random.choice(class_indices, min_count, replace=False)
            
            X_balanced.append(X[selected_indices])
            y_balanced.append(y[selected_indices])
        
        X_balanced = np.concatenate(X_balanced)
        y_balanced = np.concatenate(y_balanced)
        
    elif method == 'hybrid':
        # Hybrid: oversample minority, slightly undersample majority
        target_count = int(np.median(list(class_counts.values())) * 1.2)
        
        X_balanced = []
        y_balanced = []
        
        for class_id in range(3):
            class_indices = np.where(y == class_id)[0]
            class_samples = X[class_indices]
            current_count = len(class_samples)
            
            if current_count < target_count:
                # Oversample
                repeats = target_count // current_count
                remainder = target_count % current_count
                
                X_balanced.append(np.tile(class_samples, (repeats, 1, 1, 1)))
                y_balanced.append(np.tile([class_id], repeats * current_count))
                
                if remainder > 0:
                    random_indices = np.random.choice(current_count, remainder, replace=False)
                    X_balanced.append(class_samples[random_indices])
                    y_balanced.append(np.tile([class_id], remainder))
            else:
                # Slight undersample or keep as is
                keep_count = min(current_count, target_count)
                selected_indices = np.random.choice(current_count, keep_count, replace=False)
                X_balanced.append(class_samples[selected_indices])
                y_balanced.append(np.tile([class_id], keep_count))
        
        X_balanced = np.concatenate(X_balanced)
        y_balanced = np.concatenate(y_balanced)
    
    elif method == 'augment':
        # Keep original + add augmented copies of minority class
        X_balanced = [X]
        y_balanced = [y]
        
        # Find minority class
        min_class = min(class_counts, key=class_counts.get)
        max_count = max(class_counts.values())
        min_count = class_counts[min_class]
        
        # How many copies needed
        copies_needed = (max_count // min_count) - 1
        
        minority_indices = np.where(y == min_class)[0]
        minority_samples = X[minority_indices]
        
        # Create augmented copies
        augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
            layers.RandomBrightness(0.15),
            layers.RandomContrast(0.15),
        ])
        
        for _ in range(copies_needed):
            augmented = augmentation(minority_samples, training=True).numpy()
            X_balanced.append(augmented)
            y_balanced.append(np.tile([min_class], len(minority_samples)))
        
        X_balanced = np.concatenate(X_balanced)
        y_balanced = np.concatenate(y_balanced)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Shuffle
    indices = np.random.permutation(len(X_balanced))
    X_balanced = X_balanced[indices]
    y_balanced = y_balanced[indices]
    
    balanced_counts = Counter(y_balanced)
    print(f"Balanced distribution: {dict(balanced_counts)}")
    print(f"Dataset size: {len(y)} → {len(y_balanced)}")
    
    return X_balanced, y_balanced


def load_lisa_dataset(base_path):
    """Load complete LISA dataset"""
    
    day_clips = [f'dayClip{i}' for i in range(1, 14)]
    night_clips = [f'nightClip{i}' for i in range(1, 6)]
    
    print("Loading training data from dayTrain...")
    X_train_day, y_train_day = parse_train_dataset(base_path, 'dayTrain', day_clips)
    
    print("\nLoading training data from nightTrain...")
    X_train_night, y_train_night = parse_train_dataset(base_path, 'nightTrain', night_clips)
    
    if len(X_train_day) > 0 and len(X_train_night) > 0:
        X_train = np.concatenate([X_train_day, X_train_night])
        y_train = np.concatenate([y_train_day, y_train_night])
    elif len(X_train_day) > 0:
        X_train = X_train_day
        y_train = y_train_day
    elif len(X_train_night) > 0:
        X_train = X_train_night
        y_train = y_train_night
    else:
        X_train = np.array([])
        y_train = np.array([])
    
    print("\nLoading test sequences...")
    test_sequences = ['daySequence1', 'daySequence2', 'nightSequence1', 'nightSequence2']
    X_test, y_test = parse_sequence_dataset(base_path, test_sequences)
    
    print(f"\nDataset loaded:")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Print class distribution
    print(f"\nTraining class distribution:")
    for i, name in enumerate(CLASS_NAMES):
        count = np.sum(y_train == i)
        print(f"  {name}: {count} ({count/len(y_train)*100:.1f}%)")
    
    return X_train, y_train, X_test, y_test


# ============================================================================
# IMPROVED MODEL ARCHITECTURE
# ============================================================================

def create_mobilenet_model(num_classes=3):
    """Create MobileNetV2-based model - Balanced version"""
    
    base_model = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet',
        alpha=0.75  # Volver a capacidad media (era 0.5)
    )
    
    base_model.trainable = False
    
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)  # Reducido de 0.5
    x = layers.Dense(256,  # Aumentado de 128
                     activation='relu',
                     kernel_regularizer=regularizers.l2(0.001))(x)  # Reducido de 0.01
    x = layers.Dropout(0.3)(x)  # Reducido de 0.4
    x = layers.Dense(128, activation='relu')(x)  # Capa adicional sin regularización
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    return model, base_model


# ============================================================================
# IMPROVED TRAINING WITH PROPER DATA AUGMENTATION
# ============================================================================

def focal_loss(gamma=2.0, alpha=0.25):
    """Focal loss for handling class imbalance"""
    def loss_fn(y_true, y_pred):
        # Convert to one-hot
        y_true = tf.cast(y_true, tf.int32)
        y_true_one_hot = tf.one_hot(y_true, depth=3)
        
        # Calculate focal loss
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        
        loss = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    
    return loss_fn


def create_augmentation_layer():
    """Create data augmentation as a Keras layer - Less aggressive"""
    # return keras.Sequential([
    #     layers.RandomFlip("horizontal"),
    #     layers.RandomRotation(0.15),      # Increased from 0.05
    #     layers.RandomZoom(0.2),           # Increased from 0.1
    #     layers.RandomBrightness(0.25),    # Increased from 0.15
    #     layers.RandomContrast(0.25),      # Increased from 0.15
    #     layers.RandomTranslation(0.1, 0.1),  # NEW
    # ], name='augmentation')
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.10),      # Reducido de 0.15
        layers.RandomZoom(0.15),          # Reducido de 0.2
        layers.RandomBrightness(0.20),    # Reducido de 0.25
        layers.RandomContrast(0.20),      # Reducido de 0.25
    ], name='augmentation')


def train_model(X_train, y_train, X_val, y_val):
    """Train with proper data augmentation pipeline"""
    
    print("\n" + "="*50)
    print("PHASE 1: Training top layers")
    print("="*50)
    
    model, base_model = create_mobilenet_model(num_classes=3)
    
    # Add augmentation to the model
    augmentation = create_augmentation_layer()
    
    # Create training model with augmentation
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = augmentation(inputs)
    outputs = model(x)
    training_model = keras.Model(inputs, outputs)
    
    training_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # Lower LR
        loss=focal_loss(gamma=2.0, alpha=0.25),  # Focal loss handles imbalance
        metrics=['accuracy']
    )

    # CAMBIO 2: Calcular class weights correctamente
    from sklearn.utils.class_weight import compute_class_weight

    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    
    print(f"\nClass weights: {class_weight_dict}")
    
    history1 = training_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS_PHASE1,
        #class_weight=class_weight_dict,  # USAR class weights
        callbacks=[
            keras.callbacks.EarlyStopping(
                patience=7,  # Aumentado de 5
                restore_best_weights=True, 
                monitor='val_loss', 
                mode='min'),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.5,  # Menos agresivo (era 0.2)
                patience=4,  # Reducido de 5
                monitor='val_loss', 
                verbose=1, 
                mode='min', 
                min_lr=1e-6),  # Aumentado de 1e-7
            keras.callbacks.ModelCheckpoint(
                'best_model_phase1.keras', 
                save_best_only=True, 
                monitor='val_loss', 
                mode='min')
        ],
        verbose=1
    )
    
    print("\n" + "="*50)
    print("PHASE 2: Fine-tuning base model")
    print("="*50)

    # Unfreeze for fine-tuning
    base_model.trainable = True

    # Freeze early layers - MORE layers this time
    for layer in base_model.layers[:100]:  # Freeze MORE layers (was 80)
        layer.trainable = False

    # CRITICAL: Use SAME loss as Phase 1 (focal loss)
    # CRITICAL: Much LOWER learning rate
    training_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),  # 10x lower than Phase 1
        loss=focal_loss(gamma=2.0, alpha=0.25),  # SAME as Phase 1
        metrics=['accuracy']
    )

    # NO class weights in Phase 2 - focal loss already handles imbalance
    history2 = training_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS_PHASE2,
        # NO class_weight parameter
        callbacks=[
            keras.callbacks.EarlyStopping(
                patience=15,  # More patience for fine-tuning
                restore_best_weights=True, 
                monitor='val_accuracy',  # Switch to accuracy
                mode='max'),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=5,
                monitor='val_accuracy',  # Switch to accuracy
                verbose=1, 
                mode='max', 
                min_lr=1e-7),
            keras.callbacks.ModelCheckpoint(
                'best_model_phase2.keras', 
                save_best_only=True, 
                monitor='val_accuracy',  # Switch to accuracy
                mode='max')
        ],
        verbose=1
    )
    
    # Compile the base model for inference (without augmentation)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE/10),
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=['accuracy']
    )
    
    return model, history1, history2


# ============================================================================
# TFLITE CONVERSION
# ============================================================================

def convert_to_tflite(model, X_test):
    """Convert to TFLite with INT8 quantization"""
    
    print("\nConverting to TFLite with INT8 quantization...")
    
    def representative_dataset():
        for i in range(min(200, len(X_test))):  # Use more samples
            yield [X_test[i:i+1].astype(np.float32)]
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    tflite_model = converter.convert()
    
    with open('traffic_light_model.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print(f"Model saved: traffic_light_model.tflite")
    print(f"Model size: {len(tflite_model) / 1024:.2f} KB")
    
    return tflite_model


def evaluate_tflite_model(tflite_model, X_test, y_test):
    """Evaluate TFLite model"""
    
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Check quantization
    print(f"\nInput: {input_details[0]['dtype']}")
    print(f"Output: {output_details[0]['dtype']}")
    
    correct = 0
    total = len(X_test)
    
    for i in range(total):
        input_data = X_test[i:i+1].astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        pred = np.argmax(output_data)
        
        if pred == y_test[i]:
            correct += 1
    
    accuracy = correct / total
    print(f"TFLite Model Accuracy: {accuracy*100:.2f}%")
    
    return accuracy

def evaluate_model(model, X_test, y_test, class_names):
    """Evaluate model performance with classification report and confusion matrix"""
    from sklearn.metrics import classification_report, confusion_matrix

    # If y_test is already integer‑encoded (0..num_classes-1), skip LabelEncoder
    # Otherwise, encode to integers
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_test_encoded = le.fit_transform(y_test)

    # Evaluate directly with sparse labels (no one-hot)
    test_loss, test_acc = model.evaluate(X_test, y_test_encoded, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test_encoded, y_pred_classes,
                                target_names=class_names))

    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test_encoded, y_pred_classes)
    print(cm)

    return test_acc, cm


def test_random_samples(tflite_model, X_test, y_test, num_samples=30):
    """Test on random samples"""
    
    print("\n" + "="*60)
    print(f"TESTING {num_samples} RANDOM SAMPLES")
    print("="*60)
    
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    np.random.seed(42)
    test_indices = np.random.choice(len(X_test), min(num_samples, len(X_test)), replace=False)
    
    correct = 0
    results = []
    
    for idx in test_indices:
        input_data = X_test[idx:idx+1].astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Handle quantized output
        if output_details[0]['dtype'] == np.uint8:
            scale, zero_point = output_details[0]['quantization']
            output_data = scale * (output_data.astype(np.float32) - zero_point)
        
        pred_class = np.argmax(output_data)
        confidence = np.max(output_data) / np.sum(output_data)
        
        true_class = y_test[idx]
        is_correct = pred_class == true_class
        
        if is_correct:
            correct += 1
        
        results.append({
            'image': X_test[idx],
            'true': CLASS_NAMES[true_class],
            'pred': CLASS_NAMES[pred_class],
            'confidence': confidence,
            'correct': is_correct
        })
        
        status = "✓" if is_correct else "✗"
        print(f"{status} Sample {len(results):2d}: True={CLASS_NAMES[true_class]:7s} | "
              f"Pred={CLASS_NAMES[pred_class]:7s} | Confidence={confidence:.2%}")
    
    accuracy = correct / len(test_indices)
    
    print("\n" + "-"*60)
    print(f"RESULTS: {correct}/{len(test_indices)} correct")
    print(f"ACCURACY: {accuracy*100:.2f}%")
    print("-"*60)
    
    return accuracy, results


def plot_training_history(history1, history2):
    """Plot training metrics"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Combine histories
    acc = history1.history['accuracy'] + history2.history['accuracy']
    val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    loss = history1.history['loss'] + history2.history['loss']
    val_loss = history1.history['val_loss'] + history2.history['val_loss']
    
    epochs = range(1, len(acc) + 1)
    
    # Accuracy
    axes[0].plot(epochs, acc, 'b-', label='Training')
    axes[0].plot(epochs, val_acc, 'r-', label='Validation')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(epochs, loss, 'b-', label='Training')
    axes[1].plot(epochs, val_loss, 'r-', label='Validation')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("\nTraining history saved as: training_history.png")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

# def main():
#     print("Traffic Light Detection & Classification - FIXED VERSION")
#     print("="*60)
    
#     # Load dataset
#     X_train, y_train, X_test, y_test = load_lisa_dataset(DATASET_PATH)
    
#     if len(X_train) == 0 or len(X_test) == 0:
#         print("\nERROR: Dataset failed to load. Check paths and file structure.")
#         return
    
#     # Balance the training dataset BEFORE preprocessing
#     # RECOMMENDED: 'augment' - Creates diverse variations of minority class
#     # Other options: 'hybrid', 'oversample', 'undersample'
#     print("\n" + "="*60)
#     print("BALANCING DATASET")
#     print("="*60)
#     X_train, y_train = balance_dataset(X_train, y_train, method='hybrid')
    
#     # Apply very light OV2640 preprocessing
#     print("\nApplying OV2640 camera preprocessing (very light)...")
#     #X_train = preprocess_images_for_ov2640(X_train, severity=0.15)
#     X_test = preprocess_images_for_ov2640(X_test, severity=0.15)
    
#     # Split with stratification - use larger validation set for more stable metrics
#     X_train, X_val, y_train, y_val = train_test_split(
#         X_train, y_train, test_size=0.20, random_state=42, stratify=y_train
#     )
    
#     print(f"\nFinal dataset split:")
#     print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
#     # Train
#     model, history1, history2 = train_model(X_train, y_train, X_val, y_val)
    
#     # Plot training history
#     plot_training_history(history1, history2)
    
#     # Evaluate
#     print("\n" + "="*60)
#     print("Evaluating on test set...")
#     test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
#     print(f"Test Accuracy: {test_acc*100:.2f}%")
    
#     # Convert to TFLite
#     tflite_model = convert_to_tflite(model, X_test)
    
#     # Evaluate TFLite
#     tflite_acc = evaluate_tflite_model(tflite_model, X_test, y_test)
    
#     # Test samples
#     accuracy, results = test_random_samples(tflite_model, X_test, y_test, num_samples=30)
    
#     print("\n" + "="*60)
#     print("SUMMARY")
#     print("="*60)
#     print(f"Keras Model Test Accuracy: {test_acc*100:.2f}%")
#     print(f"TFLite Model Accuracy: {tflite_acc*100:.2f}%")
#     print(f"Random Sample Accuracy: {accuracy*100:.2f}%")
#     print("\nModel saved as: traffic_light_model.tflite")
#     print("="*60)

# alternative Main function without balancing and preprocessing on training data
def main():
    print("Traffic Light Detection & Classification - FIXED VERSION")
    print("="*60)
    
    # Load dataset
    X_train, y_train, X_test, y_test = load_lisa_dataset(DATASET_PATH)
    
    if len(X_train) == 0 or len(X_test) == 0:
        print("\nERROR: Dataset failed to load. Check paths and file structure.")
        return
    
    # VERIFICACIÓN 1: Dataset original
    print("\n" + "="*60)
    print("DATASET ORIGINAL (antes de balancing)")
    print("="*60)
    print(f"Train samples: {len(X_train)}")
    print(f"Imágenes únicas: {len(np.unique(X_train.reshape(len(X_train), -1), axis=0))}")
    for i, name in enumerate(CLASS_NAMES):
        count = np.sum(y_train == i)
        print(f"  {name}: {count} ({count/len(y_train)*100:.1f}%)")
    
    # NO USAR AUGMENT - Causa duplicados perfectos
    # En su lugar, usar focal loss + class weights
    print("\n⚠️  SKIPPING dataset balancing to avoid data leakage")
    print("    Using focal loss + class weights instead")
    
    # NO aplicar preprocessing antes del split
    print("\n⚠️  SKIPPING OV2640 preprocessing on training data")
    print("    Will only apply to test set")
    
    # Normalizar solamente
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    
    # Split con validación
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.30, random_state=42, stratify=y_train
    )
    
    # VERIFICACIÓN 2: Después del split
    print("\n" + "="*60)
    print("DESPUÉS DEL SPLIT")
    print("="*60)
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Verificar que no hay overlap
    train_hashes = set([hash(x.tobytes()) for x in X_train[:100]])
    val_hashes = set([hash(x.tobytes()) for x in X_val[:100]])
    overlap = train_hashes & val_hashes
    print(f"Overlap check (primeras 100): {len(overlap)} duplicados")
    if len(overlap) > 0:
        print("⚠️  WARNING: Data leakage detectado!")
    
    # Train con class weights
    model, history1, history2 = train_model(X_train, y_train, X_val, y_val)
    
    # Solo aplicar OV2640 al test set
    #print("\nAplicando OV2640 preprocessing solo a test set...")
    #X_test = preprocess_images_for_ov2640(X_test, severity=0.15)
    
    # Resto del código...
    plot_training_history(history1, history2)
    
    print("\n" + "="*60)
    print("Evaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    test_acc, cm = evaluate_model(model, X_test, y_test, CLASS_NAMES)
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Confusion Matrix:\n {cm}")
    
    tflite_model = convert_to_tflite(model, X_test)
    tflite_acc = evaluate_tflite_model(tflite_model, X_test, y_test)
    accuracy, results = test_random_samples(tflite_model, X_test, y_test, num_samples=30)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Keras Model Test Accuracy: {test_acc*100:.2f}%")
    print(f"TFLite Model Accuracy: {tflite_acc*100:.2f}%")
    print(f"Random Sample Accuracy: {accuracy*100:.2f}%")
    print("\nModel saved as: traffic_light_model.tflite")
    print("="*60)


if __name__ == "__main__":
    main()