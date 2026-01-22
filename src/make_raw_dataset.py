import os
import json
import numpy as np
from PIL import Image
import gc
import random

# ==========================================
# [설정] Undersampling 데이터셋 생성기
# ==========================================
LABEL_DIR = '../data/vzlusat1-timepix-data/data/labelled'
RAW_DIR = '../data/vzlusat1-timepix-data/data/raw'
OUTPUT_DIR = '../data/dataset_raw_undersampled_fixed' # 폴더명 변경

TARGET_SIZE = (32, 32)
CHUNK_SIZE = 50000 

# [핵심] Class 0 (Dot+Noise)를 얼마나 남길 것인가?
# 현재 Class 1, 2가 약 13~15만개이므로, Class 0도 비슷하게 맞추기 위해
# 전체의 약 8%만 가져오면 얼추 15만개가 됩니다.
CLASS_0_KEEP_PROB = 0.08 
BACKGROUND_SAMPLES_PER_FILE = 20 
GLOBAL_MAX_VALUE = 255.0


CLASS_PRIORITIES = {
    'dot': 0, 'drop': 0, 'other': 0, 'artefact': 0,
    'track_straight': 1, 'track_curly': 1, 'track_lowres': 1,
    'blob_big': 2, 'blob_small': 2, 'blob_branched': 2
}

def find_raw_file(json_filename, raw_file_map):
    base_name = json_filename.replace('.clusters.txt', '.txt')
    return raw_file_map.get(base_name)

def save_chunk(X, y, chunk_id, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.join(output_dir, f"raw_part_{chunk_id}.npz")
    X_np = np.array(X, dtype=np.float32)
    y_np = np.array(y, dtype=np.int32)
    np.savez_compressed(filename, X=X_np, y=y_np)
    print(f"   💾 [Chunk {chunk_id}] 저장 완료: {filename} (Size: {len(X)})")
    del X_np, y_np
    gc.collect()

def create_undersampled_dataset():
    print(f">>> ✂️ [Undersampling] Class 0 다이어트 시작 (Target: 1:1:1 Balance)...")
    
    raw_file_map = {}
    for root, dirs, files in os.walk(RAW_DIR):
        for f in files:
            if f.endswith(".txt") and "fullres" in f:
                raw_file_map[f] = os.path.join(root, f)

    X_buffer = []
    y_buffer = []
    chunk_counter = 0
    counts = {0: 0, 1: 0, 2: 0} # 개수 세기용
    
    for root, dirs, files in os.walk(LABEL_DIR):
        for file in files:
            if file.endswith("fullres.clusters.txt"):
                raw_path = find_raw_file(file, raw_file_map)
                if raw_path is None: continue
                
                try:
                    full_raw_img = np.loadtxt(raw_path)
                    img_h, img_w = full_raw_img.shape
                    occupancy_mask = np.zeros_like(full_raw_img, dtype=bool)
                    
                    with open(os.path.join(root, file), 'r') as f:
                        clusters = json.load(f)
                        
                    # 1. Labelled Data 처리
                    for cluster in clusters:
                        cls_data = cluster.get('cluster_class')
                        cls_name = cls_data.get('name') if isinstance(cls_data, dict) else cls_data
                        
                        if cls_name not in CLASS_PRIORITIES: continue
                        priority = CLASS_PRIORITIES[cls_name]
                        
                        # [핵심 로직] Class 0이면 92% 확률로 버림 (다이어트)
                        if priority == 0:
                            if random.random() > CLASS_0_KEEP_PROB:
                                continue # 스킵!
                        
                        # (Class 1, 2는 무조건 통과)

                        pixels = cluster.get('pixels')
                        if not pixels: continue
                        xs = [p['x'] for p in pixels]
                        ys = [p['y'] for p in pixels]
                        for x, y in zip(xs, ys): occupancy_mask[y, x] = True
                            
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
                        
                        # [변경 전] resized /= np.max(resized)  <-- 삭제!!
                        
                        # [변경 후] 절대 값으로 정규화 (노이즈는 작게, 입자는 크게 유지)
                        # 값을 0~255 범위로 클리핑하고 255로 나눔 (또는 센서 특성에 맞게 100 등 조정)
                        resized = np.clip(resized, 0, GLOBAL_MAX_VALUE) 
                        resized /= GLOBAL_MAX_VALUE 
                            
                        X_buffer.append(resized[..., np.newaxis])
                        y_buffer.append(priority)
                        counts[priority] += 1

                    # 2. Noise Data 처리 (Class 0)
                    bg_count = 0
                    attempts = 0
                    while bg_count < BACKGROUND_SAMPLES_PER_FILE and attempts < 100:
                        attempts += 1
                        
                        # [핵심 로직] Noise도 Class 0이므로 똑같이 확률적으로 버림
                        if random.random() > CLASS_0_KEEP_PROB:
                            bg_count += 1 # 카운트는 올리되 저장은 안 함
                            continue

                        rw, rh = random.randint(10, 50), random.randint(10, 50)
                        rx = random.randint(0, img_w - rw - 1)
                        ry = random.randint(0, img_h - rh - 1)
                        
                        if np.any(occupancy_mask[ry:ry+rh, rx:rx+rw]): continue
                        
                        roi = full_raw_img[ry:ry+rh, rx:rx+rw]
                        max_dim = max(rh, rw)
                        canvas = np.zeros((max_dim, max_dim), dtype=np.float32)
                        sy, sx = (max_dim - rh) // 2, (max_dim - rw) // 2
                        canvas[sy:sy+rh, sx:sx+rw] = roi
                        
                        img = Image.fromarray(canvas)
                        img = img.resize(TARGET_SIZE, resample=Image.NEAREST)
                        resized = np.array(img, dtype=np.float32)

                        # [변경 전] resized /= np.max(resized) <-- 삭제!!

                        # [변경 후] 동일한 절대 정규화 적용
                        resized = np.clip(resized, 0, GLOBAL_MAX_VALUE)
                        resized /= GLOBAL_MAX_VALUE
                        
                        X_buffer.append(resized[..., np.newaxis])
                        y_buffer.append(0)
                        counts[0] += 1
                        bg_count += 1
                    
                    if len(X_buffer) >= CHUNK_SIZE:
                        save_chunk(X_buffer, y_buffer, chunk_counter, OUTPUT_DIR)
                        X_buffer = []
                        y_buffer = []
                        chunk_counter += 1
                        print(f"   ...현재 비율: Dot {counts[0]} | Track {counts[1]} | Blob {counts[2]}")
                        
                except Exception:
                    continue

    if len(X_buffer) > 0:
        save_chunk(X_buffer, y_buffer, chunk_counter, OUTPUT_DIR)
    
    print(f"\n✅ 완료! 최종 비율: Dot {counts[0]} | Track {counts[1]} | Blob {counts[2]}")

if __name__ == '__main__':
    create_undersampled_dataset()