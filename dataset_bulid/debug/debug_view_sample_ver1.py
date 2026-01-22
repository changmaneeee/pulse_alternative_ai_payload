import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# [설정] 확인
# ==========================================
DATA_FILE = '../data/debug_dataset_sample/debug_sample.npz'
CLASS_NAMES = {0: 'Dot/Noise', 1: 'Track', 2: 'Blob'}

def inspect_debug_sample():
    if not os.path.exists(DATA_FILE):
        print("❌ 데이터 파일이 없습니다. 먼저 생성 스크립트를 실행하세요.")
        return

    with np.load(DATA_FILE) as data:
        X = data['X']
        y = data['y']

    print(f">>> 📊 데이터 분석 ({len(X)}개)")
    print(f"    전체 Min: {np.min(X):.4f}")
    print(f"    전체 Max: {np.max(X):.4f}")
    print(f"    전체 Mean: {np.mean(X):.4f}")

    # 시각화 (랜덤 20개)
    plt.figure(figsize=(12, 10))
    indices = np.random.choice(len(X), min(20, len(X)), replace=False)
    
    for i, idx in enumerate(indices):
        img = X[idx].squeeze()
        label = y[idx]
        img_max = np.max(img)
        
        ax = plt.subplot(4, 5, i + 1)
        # vmax=1.0으로 고정해야 '진짜 밝기'를 알 수 있음
        ax.imshow(img, cmap='gray', vmin=0, vmax=1.0) 
        
        # 제목에 정보 표시
        # OK 상황: 노이즈는 Max가 0.0x, 입자는 0.5~1.0
        ax.set_title(f"{CLASS_NAMES[label]}\nPx Max: {img_max:.2f}") 
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('debug_result.png')
    print("\n✅ 'debug_result.png' 저장 완료! 이미지를 열어서 확인하세요.")
    print("   [판단 기준]")
    print("   1. Track/Blob인데 화면이 까맣다? -> Max 값 확인 (너무 작으면 정규화 기준을 더 낮춰야 함)")
    print("   2. Noise인데 너무 하얗다? -> 정규화 기준을 더 높여야 함")
    print("   3. 입자가 회색~흰색으로 잘 보인다? -> 성공! 🎉")

if __name__ == '__main__':
    inspect_debug_sample()