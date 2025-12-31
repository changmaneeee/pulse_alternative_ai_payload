import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 데이터 준비 (Data Preparation)
# ==========================================
# 실제 데이터가 있다면 이 함수 대신 로더를 사용하세요.
def generate_synthetic_data(num_samples=2000, img_shape=(32, 32, 1)):
    """
    VZLUSAT-1 데이터와 유사한 3가지 클래스의 가상 데이터를 생성합니다.
    0: Dot (Gamma/X-ray) - Low Priority
    1: Track (Electron) - Medium Priority
    2: Blob (Heavy Ion) - High Priority (Critical)
    """
    X = np.zeros((num_samples,) + img_shape, dtype=np.float32)
    y = np.zeros(num_samples, dtype=int)
    
    for i in range(num_samples):
        # 배경 노이즈
        X[i] = np.random.normal(0, 0.05, img_shape)
        category = np.random.randint(0, 3)
        y[i] = category
        
        if category == 0: # Dot
            for _ in range(np.random.randint(1, 3)):
                rx, ry = np.random.randint(5, 27, 2)
                X[i, ry, rx, 0] += np.random.uniform(0.5, 1.0)
        elif category == 1: # Track (Line)
            p0 = np.random.randint(5, 27, 2)
            p1 = np.random.randint(5, 27, 2)
            x_vals = np.linspace(p0[0], p1[0], 10).astype(int)
            y_vals = np.linspace(p0[1], p1[1], 10).astype(int)
            X[i, y_vals, x_vals, 0] += np.random.uniform(0.3, 0.8)
        elif category == 2: # Blob (Heavy Ion)
            cy, cx = np.random.randint(10, 22, 2)
            y_grid, x_grid = np.ogrid[-cy:32-cy, -cx:32-cx]
            mask = x_grid*x_grid + y_grid*y_grid <= np.random.randint(3, 6)**2
            X[i, :, :, 0][mask] += np.random.uniform(0.6, 1.0)
            
    return np.clip(X, 0, 1), y

# 데이터 생성 (Train/Test 분리)
print("Generating Data...")
X_train, y_train = generate_synthetic_data(2000) # 학습용 2000장
X_test, y_test = generate_synthetic_data(500)    # 평가용 500장
print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")

# ==========================================
# 2. 모델 후보군 정의 (Model Candidates)
# ==========================================

def get_model(model_type, input_shape=(32, 32, 1), num_classes=3):
    if model_type == 'Simple_CNN':
        # 후보 1: 일반적인 CNN
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ], name='Simple_CNN')
        
    elif model_type == 'DSC_CNN':
        # 후보 2: Depthwise Separable CNN (MobileNet 스타일, 초경량)
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(8, (3, 3), strides=2, padding='same', activation='relu'),
            # Depthwise Block 1
            layers.DepthwiseConv2D((3, 3), padding='same', activation='relu'),
            layers.Conv2D(16, (1, 1), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            # Depthwise Block 2
            layers.DepthwiseConv2D((3, 3), padding='same', activation='relu'),
            layers.Conv2D(32, (1, 1), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(num_classes, activation='softmax')
        ], name='DSC_CNN')
        
    elif model_type == 'MLP':
        # 후보 3: 단순 MLP (비교용)
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ], name='MLP_Baseline')
        
    return model

# ==========================================
# 3. 학습 및 벤치마크 (Training & Benchmark)
# ==========================================
candidates = ['Simple_CNN', 'DSC_CNN', 'MLP']
results = {}

print("\nStarting Training & Benchmark...")
print(f"{'Model':<15} | {'Acc':<8} | {'Params':<8} | {'Size(KB)':<10}")
print("-" * 50)

for name in candidates:
    model = get_model(name)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # 학습 (Epochs는 필요에 따라 조절)
    history = model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0, validation_data=(X_test, y_test))
    
    # 평가
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    params = model.count_params()
    
    # Estimation size of Transfer to TFLite model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    size_kb = len(tflite_model) / 1024
    
    results[name] = {'acc': acc, 'params': params, 'size_kb': size_kb, 'history': history}
    
    print(f"{name:<15} | {acc:.4f}   | {params:<8} | {size_kb:.2f} KB")

# ==========================================
# 4. 결과 시각화 (Visualization)
# ==========================================
plt.figure(figsize=(10, 5))
for name in candidates:
    plt.plot(results[name]['history'].history['accuracy'], label=f"{name} Train")
    plt.plot(results[name]['history'].history['val_accuracy'], linestyle='--', label=f"{name} Val")

plt.title('Model Accuracy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()