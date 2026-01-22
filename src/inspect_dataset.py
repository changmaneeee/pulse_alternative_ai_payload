import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# ==========================================
# [설정] 확인하려는 데이터셋 경로
# ==========================================
DATA_DIR = '../data/dataset_raw_undersampled_fixed'
CLASS_NAMES = {0: 'Dot/Noise', 1: 'Track', 2: 'Blob'}

def inspect_data():
    # 1. 파일 하나 로드
    files = sorted(glob.glob(os.path.join(DATA_DIR, "raw_part_*.npz")))
    if not files:
        print("❌ 데이터 파일이 없습니다.")
        return

    target_file = files[0] # 첫 번째 파일만 확인
    print(f">>> 📂 파일 분석 중: {target_file}")

    with np.load(target_file) as data:
        X = data['X']
        y = data['y']

    print(f"   데이터 형태: {X.shape}")
    print(f"   레이블 형태: {y.shape}")

    # 2. 통계 정보 확인 (핵심!)
    print("\n>>> 📊 [픽셀 값 통계] (0~1 사이여야 정상)")
    print(f"   Min Value: {np.min(X):.4f}")
    print(f"   Max Value: {np.max(X):.4f}")
    print(f"   Mean Value: {np.mean(X):.4f}")
    
    # 0이나 1로 쏠려있는지 확인
    zero_count = np.sum(X == 0)
    one_count = np.sum(X == 1)
    total_pixels = X.size
    print(f"   완전 검은색(0.0) 비율: {zero_count / total_pixels * 100:.2f}%")
    print(f"   완전 흰색(1.0) 비율:   {one_count / total_pixels * 100:.2f}% (이게 너무 높으면 망한 것)")

    # 3. 클래스별 샘플 시각화
    print("\n>>> 🖼️ 샘플 이미지 저장 중...")
    
    plt.figure(figsize=(15, 5))
    
    for cls in [0, 1, 2]:
        # 해당 클래스의 인덱스 찾기
        idxs = np.where(y == cls)[0]
        if len(idxs) == 0: continue
        
        # 랜덤하게 5개 뽑기
        selected_idxs = np.random.choice(idxs, 5, replace=False) if len(idxs) > 5 else idxs
        
        for i, idx in enumerate(selected_idxs):
            ax = plt.subplot(3, 5, cls * 5 + i + 1)
            img = X[idx].squeeze()
            
            # 이미지 그리기
            ax.imshow(img, cmap='inferno', vmin=0, vmax=1)
            ax.set_title(f"{CLASS_NAMES[cls]}\nMax: {np.max(img):.2f}")
            ax.axis('off')

    plt.tight_layout()
    save_path = 'data_inspection.png'
    plt.savefig(save_path)
    print(f"✅ 시각화 완료: {save_path} 파일을 열어서 눈으로 확인하세요!")

if __name__ == '__main__':
    inspect_data()