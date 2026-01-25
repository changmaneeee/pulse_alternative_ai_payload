import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# 경로 설정 (src 폴더에서 실행한다고 가정하고 상위 폴더 참조)
DATA_PATH = "../data/dataset_final_v1.npz"   # 경로 수정됨 (data 폴더)
MODEL_PATH = "../models/final_model.keras"
RESULT_DIR = "../results"

def main():
    if not os.path.exists(DATA_PATH):
        print(f"❌ 데이터 없음: {DATA_PATH}")
        return
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 모델 없음: {MODEL_PATH}")
        return

    print("📂 데이터 로드 중...")
    data = np.load(DATA_PATH)
    X = data['X']
    y = data['y']
    
    # Test 셋 분리 (Seed 42)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    
    print("🧠 모델 로드 및 추론 중...")
    model = tf.keras.models.load_model(MODEL_PATH)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    
    # 정확도
    acc = np.mean(y_test == y_pred)
    print(f"✅ Final Accuracy: {acc*100:.2f}%")

    # 혼동 행렬
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Noise', 'Track', 'Blob'], yticklabels=['Noise', 'Track', 'Blob'])
    plt.title(f'Confusion Matrix (Acc: {acc*100:.2f}%)')
    plt.ylabel('True'); plt.xlabel('Pred')
    plt.savefig(os.path.join(RESULT_DIR, "confusion_matrix.png"))
    print("🖼️ 혼동 행렬 저장 완료.")

    # 샘플 이미지
    fig, axes = plt.subplots(3, 5, figsize=(15, 8))
    class_names = ['Noise', 'Track', 'Blob']
    for i in range(3):
        idxs = np.where(y_test == i)[0]
        samples = np.random.choice(idxs, min(5, len(idxs)), replace=False)
        for j, idx in enumerate(samples):
            axes[i, j].imshow(X_test[idx].squeeze(), cmap='inferno')
            col = 'green' if y_test[idx] == y_pred[idx] else 'red'
            axes[i, j].set_title(f"T:{class_names[y_test[idx]]}\nP:{class_names[y_pred[idx]]}", color=col)
            axes[i, j].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "sample_predictions.png"))
    print("🖼️ 샘플 이미지 저장 완료.")

if __name__ == "__main__":
    main()