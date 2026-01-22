import os
import json
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt


# ==========================================
# [설정] 강제 균형 샘플링 생성기
# ==========================================
LABEL_DIR = '/home/changmin/pulse_alternative_ai_payload/data/vzlusat1-timepix-data/data/labelled'
RAW_DIR = '/home/changmin/pulse_alternative_ai_payload/data/vzlusat1-timepix-data/data/raw'
OUTPUT_FILE = '../data/debug_dataset_sample/debug_balanced.npz'

TARGET_SIZE = (32, 32)
GLOBAL_MAX_VALUE = 150.0  # 정규화 기준

# 클래스별로 딱 이만큼만 찾고 멈춥니다.
TARGET_PER_CLASS = 20 

CLASS_PRIORITIES = {
    'dot': 0, 'drop': 0, 'other': 0, 'artefact': 0,
    'track_straight': 1, 'track_curly': 1, 'track_lowres': 1,
    'blob_big': 2, 'blob_small': 2, 'blob_branched': 2
}

# 이름 역참조 (0->Dot, 1->Track...)
CLASS_NAMES_REV = {0: 'Dot/Noise', 1: 'Track', 2: 'Blob'}

def find_raw_file(json_filename, raw_file_map):
    base_name = json_filename.replace('.clusters.txt', '.txt')
    return raw_file_map.get(base_name)

def create_balanced_debug_sample():
    print(f">>> ⚖️ [균형 샘플링] 각 클래스당 {TARGET_PER_CLASS}개씩 강제 수집 시작...")
    
    if not os.path.exists(os.path.dirname(OUTPUT_FILE)):
        os.makedirs(os.path.dirname(OUTPUT_FILE))

    # 1. Raw 파일 맵핑
    raw_file_map = {}
    for root, dirs, files in os.walk(RAW_DIR):
        for f in files:
            if f.endswith(".txt") and "fullres" in f and "metadata" not in f:
                raw_file_map[f] = os.path.join(root, f)

    # 2. Label 파일 리스트 확보 및 셔플
    label_files = []
    for root, dirs, files in os.walk(LABEL_DIR):
        for f in files:
            if f.endswith("fullres.clusters.txt"):
                label_files.append((root, f))
    random.shuffle(label_files) # 파일 순서 섞기

    # 저장소
    collected_data = {0: [], 1: [], 2: []}
    
    # 3. 데이터 수집 루프
    for root, file in label_files:
        # 모든 클래스 목표 달성했으면 종료
        if all(len(v) >= TARGET_PER_CLASS for v in collected_data.values()):
            print("✨ 모든 클래스 목표 달성! 조기 종료합니다.")
            break

        raw_path = find_raw_file(file, raw_file_map)
        if raw_path is None: continue

        try:
            full_raw_img = np.loadtxt(raw_path)
            with open(os.path.join(root, file), 'r') as f:
                clusters = json.load(f)

            for cluster in clusters:
                cls_data = cluster.get('cluster_class')
                cls_name = cls_data.get('name') if isinstance(cls_data, dict) else cls_data
                
                if cls_name not in CLASS_PRIORITIES: continue
                label = CLASS_PRIORITIES[cls_name]
                
                # 이미 목표치 채운 클래스는 패스
                if len(collected_data[label]) >= TARGET_PER_CLASS: continue

                pixels = cluster.get('pixels')
                if not pixels: continue
                
                xs = [p['x'] for p in pixels]
                ys = [p['y'] for p in pixels]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                
                roi = full_raw_img[min_y:max_y+1, min_x:max_x+1]
                h, w = roi.shape
                max_dim = max(h, w)
                canvas = np.zeros((max_dim, max_dim), dtype=np.float32)
                sy, sx = (max_dim - h) // 2, (max_dim - w) // 2
                canvas[sy:sy+h, sx:sx+w] = roi
                
                img = Image.fromarray(canvas)
                img = img.resize(TARGET_SIZE, resample=Image.NEAREST)
                resized = np.array(img, dtype=np.float32)
                
                # 정규화 및 저장
                resized = np.clip(resized, 0, GLOBAL_MAX_VALUE)
                resized /= GLOBAL_MAX_VALUE
                
                collected_data[label].append(resized)
                
        except Exception:
            continue

    # 4. 시각화 (바로 보여줌)
    print("\n>>> 🖼️ 결과 확인 (클래스별 5개씩 출력)")
    
    plt.figure(figsize=(15, 8))
    plot_idx = 1
    
    for cls_idx in [0, 1, 2]:
        samples = collected_data[cls_idx]
        print(f"   - Class {cls_idx} ({CLASS_NAMES_REV[cls_idx]}): {len(samples)}개 수집됨")
        
        # 최대 5개만 그림
        for i in range(5):
            if i >= len(samples): break
            
            img = samples[i]
            
            ax = plt.subplot(3, 5, plot_idx)
            ax.imshow(img, cmap='gray', vmin=0, vmax=1.0) # vmin/vmax 고정 중요!
            
            # 정보 표시
            max_val = np.max(img)
            raw_approx = max_val * GLOBAL_MAX_VALUE # 역산한 Raw 값
            
            ax.set_title(f"{CLASS_NAMES_REV[cls_idx]}\nNorm: {max_val:.2f}\nRaw≈{int(raw_approx)}")
            ax.axis('off')
            plot_idx += 1

    plt.tight_layout()
    plt.savefig('debug_balanced_result.png')
    print(f"\n✅ 'debug_balanced_result.png' 이미지를 확인하세요!")

if __name__ == '__main__':
    create_balanced_debug_sample()