import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.callbacks import ReduceLROnPlateau # [추가됨] 자동 브레이크
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import gc

# ==========================================
# [설정] Undersampled Dataset
# ==========================================
DATA_DIR = '../data/dataset_v3_augmented' 
MODEL_DIR = '../models_new_v2'
PLOT_DIR = '../models_new_v2/plots'

if not os.path.exists(PLOT_DIR): os.makedirs(PLOT_DIR)

BATCH_SIZE = 512
MAX_EPOCHS = 200

def get_file_list():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "aug_part_*.npz")))
    if not files: raise FileNotFoundError("❌ 데이터 없음")
    return files

def count_samples(file_list):
    """단순히 총 데이터 개수만 셉니다."""
    total = 0
    print(">>> 📊 데이터 개수 스캔 중...")
    for f in file_list:
        with np.load(f, mmap_mode='r') as data:
            total += len(data['y'])
    print(f"   총 샘플 수: {total}")
    return total

def data_generator(file_list):
    """
    [Final Fix] 종료 신호 충돌 방지 및 Validation 로그 누락 해결 버전
    """
    files = file_list.copy()
    while True:
        np.random.shuffle(files)
        for f in files:
            try:
                # 1. 파일 로딩
                with np.load(f, mmap_mode='r') as data:
                    X_chunk = data['X']
                    y_chunk = data['y']
                    
                    indices = np.arange(len(X_chunk))
                    np.random.shuffle(indices)
                    
                    # 2. 데이터 공급
                    for i in indices:
                        try:
                            yield X_chunk[i], y_chunk[i]
                        except GeneratorExit:
                            return 

            except GeneratorExit:
                return
                
            except Exception as e:
                print(f"⚠️ Data Load Error: {f}")
                continue

def create_dataset(file_list, is_train=True):
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(file_list),
        output_signature=(
            tf.TensorSpec(shape=(32, 32, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    if is_train: dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# -----------------------------------------------------------
# 모델 정의 (Simple_CNN, DSC_CNN, GAP_CNN)
# -----------------------------------------------------------
def get_simple_cnn():
    model = models.Sequential([
        layers.Input(shape=(32, 32, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(3, activation='softmax')
    ], name='Simple_CNN')
    return model

def get_dsc_cnn():
    model = models.Sequential([
        layers.Input(shape=(32, 32, 1)),
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

def get_gap_cnn():
    model = models.Sequential([
        layers.Input(shape=(32, 32, 1)),
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(3, activation='softmax')
    ], name='GAP_CNN')
    return model

MODELS = {
    'Simple_CNN': get_simple_cnn,
    'DSC_CNN': get_dsc_cnn,
    'GAP_CNN': get_gap_cnn
}

# -----------------------------------------------------------
# [핵심] 학습 실행 함수 (콜백 수정됨)
# -----------------------------------------------------------
def run_comparison():
    all_files = get_file_list()
    print(f">>> 📂 Undersampled 데이터 파일 {len(all_files)}개 로드")

    train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)
    
    total_samples = count_samples(train_files)
    
    train_ds = create_dataset(train_files, is_train=True)
    val_ds = create_dataset(val_files, is_train=False)

    train_steps = int((total_samples * 0.90) // BATCH_SIZE)
    val_steps = int((total_samples * 0.25 * 0.90) // BATCH_SIZE)

    results = []

    for name, model_func in MODELS.items():
        print(f"\n🚀 Training {name} ...")
        tf.keras.backend.clear_session()
        gc.collect()

        model = model_func()

        # 학습률 초기값 1e-4 (0.0001)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        
        model.compile(optimizer=optimizer, 
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        # ======================================================
        # [수정된 부분] 콜백 리스트: 자동 브레이크 추가
        # ======================================================
        callbacks_list = [
            # 1. EarlyStopping: 20번 동안 개선 없으면 완전 종료
            callbacks.EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True),
            
            # 2. ReduceLROnPlateau (자동 브레이크):
            #    val_loss가 3번(patience) 동안 안 줄어들면 -> 학습률을 절반(0.5)으로 줄임
            #    이렇게 하면 87%에서 튀지 않고 88%, 89%로 살금살금 내려감
            ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=3, 
                min_lr=1e-7, 
                verbose=1 # 로그에 "Learning rate reduced..." 메시지 출력
            )
        ]
        
        history = model.fit(
            train_ds,
            epochs=MAX_EPOCHS,
            steps_per_epoch=train_steps,
            validation_data=val_ds,
            validation_steps=val_steps,
            callbacks=callbacks_list,
            verbose=1
        )
        
        if len(history.history['val_accuracy']) > 0:
            best_acc = max(history.history['val_accuracy'])
        else: best_acc = 0
            
        # TFLite 저장
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        tflite_path = os.path.join(MODEL_DIR, f"{name}_BALANCED.tflite")
        with open(tflite_path, 'wb') as f: f.write(tflite_model)
        size_kb = os.path.getsize(tflite_path) / 1024
        
        results.append({'Model': name, 'Acc': best_acc*100, 'Size': size_kb})
        print(f"   ✅ {name}: {best_acc*100:.2f}%")

    print(pd.DataFrame(results))

if __name__ == '__main__':
    run_comparison()