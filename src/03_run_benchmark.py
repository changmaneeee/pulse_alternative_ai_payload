import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time

# =========================================================
# 1. 설정 및 데이터 경로
# =========================================================
DATA_PATH = "dataset_bulid/dataset_final_v1.npz"
RESULT_DIR = "benchmark_results"
os.makedirs(RESULT_DIR, exist_ok=True)

BATCH_SIZE = 64
EPOCHS = 50  # 비교용이므로 15회 정도면 충분 (원하면 늘리세요)
NUM_CLASSES = 3
INPUT_SHAPE = (32, 32, 1)

# =========================================================
# 2. 모델 후보군 정의 (Modern Architectures)
# =========================================================

def get_model(name):
    name = name.lower()
    
    if name == "standard_cnn":
        # [Baseline] 일반적인 Conv-Pool 구조
        inputs = Input(shape=INPUT_SHAPE)
        x = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
        return models.Model(inputs, outputs, name="Standard_CNN")

    elif name == "dsc_cnn":
        # [Latency Champion] Depthwise Separable Conv (MobileNet style)
        inputs = Input(shape=INPUT_SHAPE)
        x = layers.Conv2D(16, (3, 3), strides=(2,2), padding='same', activation='relu')(inputs)
        
        # DSC Block 1
        x = layers.SeparableConv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = layers.SeparableConv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # DSC Block 2
        x = layers.SeparableConv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = layers.SeparableConv2D(64, (3, 3), padding='same', activation='relu')(x)
        
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
        return models.Model(inputs, outputs, name="DSC_CNN")

    elif name == "micro_squeezenet":
        # [Size Champion] Fire Module Structure
        def fire_module(x, squeeze, expand):
            s = layers.Conv2D(squeeze, (1, 1), activation='relu', padding='same')(x)
            e1 = layers.Conv2D(expand, (1, 1), activation='relu', padding='same')(s)
            e3 = layers.Conv2D(expand, (3, 3), activation='relu', padding='same')(s)
            return layers.concatenate([e1, e3], axis=-1)

        inputs = Input(shape=INPUT_SHAPE)
        x = layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same', activation='relu')(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = fire_module(x, squeeze=8, expand=32)
        x = fire_module(x, squeeze=8, expand=32)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
        return models.Model(inputs, outputs, name="Micro_SqueezeNet")

    elif name == "mini_resnet":
        # [Accuracy Champion] Residual Connections
        def res_block(x, filters):
            shortcut = x
            # 1x1 conv for shortcut dimension matching if needed
            if x.shape[-1] != filters:
                shortcut = layers.Conv2D(filters, (1, 1), padding='same')(x)
                
            x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
            x = layers.Conv2D(filters, (3, 3), padding='same')(x)
            x = layers.add([x, shortcut]) # Skip Connection
            x = layers.Activation('relu')(x)
            return x

        inputs = Input(shape=INPUT_SHAPE)
        x = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = res_block(x, 32)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = res_block(x, 64)
        x = layers.GlobalAveragePooling2D()(x)
        
        outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
        return models.Model(inputs, outputs, name="Mini_ResNet")
    
    else:
        raise ValueError("Unknown Model Name")

# =========================================================
# 3. 메인 벤치마크 실행
# =========================================================
def main():
    # A. 데이터 준비
    print("🔄 데이터 로드 중...")
    data = np.load(DATA_PATH)
    X = data['X']
    y = data['y']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Class Weights
    classes = np.unique(y_train)
    weights = class_weight.compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = dict(enumerate(weights))
    print(f"⚖️ Class Weights: {class_weights}")

    # B. 벤치마크 루프
    models_to_test = ["standard_cnn", "dsc_cnn", "micro_squeezenet", "mini_resnet"]
    results = []

    for model_name in models_to_test:
        print("\n" + "="*50)
        print(f"🔥 Testing Candidate: {model_name.upper()}")
        print("="*50)
        
        # 1. Build
        model = get_model(model_name)
        
        # 2. Compile
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # 3. Train (Time Check)
        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.1,
            class_weight=class_weights,
            verbose=0 # 로그 너무 길어지지 않게 생략 (Epoch마다 진행바 안나옴)
        )
        train_time = time.time() - start_time
        print(f"✅ Training Done ({train_time:.1f}s)")
        
        # 4. Evaluate
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"✨ Test Accuracy: {acc*100:.2f}%")
        
        # 5. Convert to TFLite & Check Size
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        tflite_path = os.path.join(RESULT_DIR, f"{model_name}.tflite")
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
            
        model_size_kb = len(tflite_model) / 1024
        print(f"📦 TFLite Size: {model_size_kb:.2f} KB")
        
        # 6. Save Result
        results.append({
            "Model": model_name,
            "Accuracy (%)": round(acc * 100, 2),
            "Size (KB)": round(model_size_kb, 2),
            "Params": model.count_params(),
            "Train Time (s)": round(train_time, 1)
        })

    # C. 결과 비교 및 시각화
    print("\n" + "="*50)
    print("🏆 Benchmark Summary")
    print("="*50)
    df = pd.DataFrame(results)
    print(df)
    
    # CSV 저장
    df.to_csv(os.path.join(RESULT_DIR, "benchmark_summary.csv"), index=False)
    
    # 그래프 그리기
    plt.figure(figsize=(14, 5))
    
    # 1. Accuracy
    plt.subplot(1, 3, 1)
    sns.barplot(x='Model', y='Accuracy (%)', data=df, palette='viridis')
    plt.title('Accuracy (Higher is Better)')
    plt.ylim(80, 100) # 차이를 잘 보이게
    plt.xticks(rotation=15)
    
    # 2. Size
    plt.subplot(1, 3, 2)
    sns.barplot(x='Model', y='Size (KB)', data=df, palette='magma')
    plt.title('Model Size (Lower is Better)')
    plt.xticks(rotation=15)
    
    # 3. Efficiency Score (Accuracy / Size) - 창의적인 지표
    df['Efficiency'] = df['Accuracy (%)'] / df['Size (KB)']
    plt.subplot(1, 3, 3)
    sns.barplot(x='Model', y='Efficiency', data=df, palette='coolwarm')
    plt.title('Efficiency (Acc / Size)')
    plt.xticks(rotation=15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "benchmark_chart.png"))
    print("\n📊 Chart Saved: benchmark_chart.png")

if __name__ == "__main__":
    main()