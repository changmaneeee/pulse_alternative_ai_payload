import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input, regularizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

# =========================================================
# 1. 설정
# =========================================================
DATA_PATH = "dataset_bulid/dataset_final_v1.npz"
MODEL_DIR = "models_final_v4"
os.makedirs(MODEL_DIR, exist_ok=True)

BATCH_SIZE = 32  # 배치 사이즈를 줄여서 더 세밀하게 학습 (64 -> 32)
EPOCHS = 50      # Augmentation 없으면 50번이면 충분
NUM_CLASSES = 3
INPUT_SHAPE = (32, 32, 1)

# =========================================================
# 2. 모델 정의 (Optimized Standard CNN)
# =========================================================
def create_optimized_cnn():
    inputs = Input(shape=INPUT_SHAPE)
    
    # Block 1
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Block 2
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Block 3 (깊지 않게, 딱 여기까지만)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x) # Flatten 대신 GAP
    
    # Dense Layers
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    return models.Model(inputs, outputs, name="Optimized_CNN")

# =========================================================
# 3. 메인 학습
# =========================================================
def main():
    print("🔄 데이터 로드 중...")
    data = np.load(DATA_PATH)
    X = data['X']
    y = data['y']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42, stratify=y_train)
    
    # Class Weights
    classes = np.unique(y_train)
    weights = class_weight.compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = dict(enumerate(weights))
    print(f"⚖️ Class Weights: {class_weights}")

    # [중요] Augmentation 제거! (순정 데이터로 승부)
    
    model = create_optimized_cnn()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), # 기본 학습률
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    callbacks_list = [
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
        callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint(filepath=os.path.join(MODEL_DIR, "best_model_v4.keras"),
                                  monitor='val_accuracy', save_best_only=True, verbose=1)
    ]
    
    print(f"\n🚀 Final Tuning Start (No Augmentation, Batch 32)")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=callbacks_list,
        verbose=1
    )
    
    print("\n📝 최종 평가:")
    loss, acc = model.evaluate(X_test, y_test)
    print(f"✨ Final Test Accuracy: {acc*100:.2f}%")
    
    # TFLite 변환
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open(os.path.join(MODEL_DIR, "final_model_v4.tflite"), "wb") as f:
        f.write(tflite_model)
    print(f"💾 TFLite Saved ({len(tflite_model)/1024:.2f} KB)")

if __name__ == "__main__":
    main()