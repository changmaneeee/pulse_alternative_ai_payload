import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
import glob
import pandas as pd

# ==========================================
# 설정 (Settings)
# ==========================================
DATA_DIR = '../data/dataset'
MODEL_ROOT_DIR = '../models'
os.makedirs(MODEL_ROOT_DIR, exist_ok=True)

INPUT_SHAPE = (32, 32, 1)
BATCH_SIZE = 512
EPOCHS = 50  # [수정] 10 -> 50 (충분히 주고 Early Stopping으로 제어)

# ==========================================
# 1. 데이터 제너레이터 (유지)
# ==========================================
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_list, batch_size=32, shuffle=True):
        self.file_list = file_list
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.total_samples = 0
        self.on_epoch_end()
        
        for f in file_list:
            try:
                data = np.load(f)
                self.total_samples += len(data['y'])
            except: pass
        print(f"   > Loaded {len(file_list)} files ({self.total_samples} samples)")

    def __len__(self):
        return int(np.floor(self.total_samples / self.batch_size))

    def __getitem__(self, index):
        file_idx = np.random.randint(0, len(self.file_list))
        file_path = self.file_list[file_idx]
        with np.load(file_path) as data:
            X, y = data['X'], data['y']
        
        indices = np.random.randint(0, len(X), self.batch_size)
        X_batch = X[indices][..., np.newaxis]
        y_batch = y[indices]
        return X_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.file_list)

# ==========================================
# 2. 모델 후보군 (유지)
# ==========================================
def build_simple_cnn():
    model = models.Sequential([
        layers.Input(shape=INPUT_SHAPE),
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(3, activation='softmax')
    ], name='Simple_CNN')
    return model

def build_dsc_cnn():
    model = models.Sequential([
        layers.Input(shape=INPUT_SHAPE),
        layers.Conv2D(8, (3, 3), strides=2, padding='same', activation='relu'),
        layers.DepthwiseConv2D((3, 3), padding='same', activation='relu'),
        layers.Conv2D(16, (1, 1), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.DepthwiseConv2D((3, 3), padding='same', activation='relu'),
        layers.Conv2D(32, (1, 1), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(3, activation='softmax')
    ], name='DSC_CNN')
    return model

def build_gap_cnn():
    model = models.Sequential([
        layers.Input(shape=INPUT_SHAPE),
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(3, activation='softmax')
    ], name='GAP_CNN')
    return model

# ==========================================
# 3. 메인 실행 함수 (업그레이드)
# ==========================================
def run_comparison():
    train_files = glob.glob(os.path.join(DATA_DIR, '*_train.npz'))
    val_files = glob.glob(os.path.join(DATA_DIR, '*_val.npz'))
    
    if not train_files: return

    train_gen = DataGenerator(train_files, batch_size=BATCH_SIZE)
    val_gen = DataGenerator(val_files, batch_size=BATCH_SIZE)

    candidates = [build_simple_cnn, build_dsc_cnn, build_gap_cnn]
    results = []
    
    plt.figure(figsize=(12, 6))

    for builder in candidates:
        model = builder()
        model_name = model.name
        print(f"\n{'='*40}")
        print(f"🚀 Training Candidate: {model_name} (Max Epochs: {EPOCHS})")
        print(f"{'='*40}")
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # [핵심 변경] 콜백 정의
        callbacks_list = [
            # 1. EarlyStopping: 검증 성능이 5번 연속 안 좋아지면 멈춤
            callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, restore_best_weights=True),
            # 2. ModelCheckpoint: 가장 성능 좋은 순간을 임시 저장
            callbacks.ModelCheckpoint(filepath=f"best_{model_name}.keras", monitor='val_accuracy', save_best_only=True, verbose=0)
        ]
        
        # 학습 (callbacks 추가)
        history = model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen, verbose=1, callbacks=callbacks_list)
        
        # --- 결과 처리 ---
        # TFLite 변환 (가장 좋았던 상태인 현재 가중치 사용)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        save_path = os.path.join(MODEL_ROOT_DIR, f"{model_name}.tflite")
        with open(save_path, 'wb') as f: f.write(tflite_model)
        
        size_kb = len(tflite_model) / 1024
        
        # Early Stopping으로 인해 epoch 수가 다를 수 있으므로 마지막 val_acc 사용
        final_acc = max(history.history['val_accuracy']) # 기록 중 최고 점수
        
        print(f"✨ Best Accuracy: {final_acc*100:.2f}% | Stopped at Epoch {len(history.history['loss'])}")

        results.append({
            "Model": model_name,
            "Best Accuracy": f"{final_acc*100:.2f}%",
            "Size (KB)": f"{size_kb:.2f} KB",
            "Epochs Run": len(history.history['loss'])
        })
        
        plt.plot(history.history['val_accuracy'], label=f"{model_name} (Max: {final_acc*100:.1f}%)")

        # 임시 파일 삭제
        if os.path.exists(f"best_{model_name}.keras"):
            os.remove(f"best_{model_name}.keras")

    print("\n🏆 [Final Comparison Result (Full Training)] 🏆")
    df = pd.DataFrame(results)
    print(df)
    
    plt.title("Model Accuracy Comparison (Full Training)")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(MODEL_ROOT_DIR, 'model_comparison_v2.png'))
    print(f"\n📊 Updated graph saved to {os.path.join(MODEL_ROOT_DIR, 'model_comparison_v2.png')}")

if __name__ == '__main__':
    run_comparison()