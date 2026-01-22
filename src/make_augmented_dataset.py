import os
import json
import numpy as np
from PIL import Image
import gc
import random

# ==========================================
# [설정] V3: 데이터 증강 (Augmentation) 적용
# ==========================================
LABEL_DIR = '../data/vzlusat1-timepix-data/data/labelled'
RAW_DIR = '../data/vzlusat1-timepix-data/data/raw'
OUTPUT_DIR = '../data/dataset_v3_augmented' # 새로운 폴더

TARGET_SIZE = (32, 32)
CHUNK_SIZE = 50000 
GLOBAL_MAX_VALUE = 150.0 

# Dot(노이즈) 비율을 조금 더 늘림 (너무 많이 버리지 않음)
CLASS_0_KEEP_PROB = 0.2 
BACKGROUND_SAMPLES_PER_FILE = 20 

CLASS_PRIORITIES = {
    'dot': 0, 'drop': 0, 'other': 0, 'artefact': 0,
    'track_straight': 1, 'track_curly': 1, 'track_lowres': 1,
    'blob_big': 2, 'blob_small': 2, 'blob_branched': 2
}

def find_raw_file(json_filename, raw_file_map):
    base_name = json_filename.replace('.clusters.txt', '.txt')
    return raw_file_map.get(base_name)

def augment_image(img_arr, label):
    """
    이미지 하나를 받아서 8개로 뻥튀기 (원본 + 회전3개 + 반전4개)
    단, Dot(0)은 형태가 단순하므로 증강하지 않음 (메모리 절약)
    """
    augmented = []
    
    # Class 0 (Dot)은 원본만 저장 (충분히 많으므로)
    if label == 0:
        augmented.append(img_arr)
        return augmented

    # Class 1, 2 (Track, Blob)은 8배 증강
    # 1. 회전 (0, 90, 180, 270)
    for k in [0, 1, 2, 3]:
        rot_img = np.rot90(img_arr, k)
        augmented.append(rot_img)
        
        # 2. 반전 (Flip) 후 회전
        flip_img = np.fliplr(rot_img)
        augmented.append(flip_img)
        
    return augmented

def save_chunk(X, y, chunk_id, output_dir):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    filename = os.path.join(output_dir, f"aug_part_{chunk_id}.npz")
    X_np = np.array(X, dtype=np.float32)
    y_np = np.array(y, dtype=np.int32)
    np.savez_compressed(filename, X=X_np, y=y_np)
    print(f"   💾 [Chunk {chunk_id}] 저장 완료: {filename} (Size: {len(X)})")
    del X_np, y_np
    gc.collect()

def create_augmented_dataset():
    print(f">>> 🌪️ [Plan B] 데이터 증강(Augmentation) 데이터셋 생성 시작...")
    
    raw_file_map = {}
    for root, dirs, files in os.walk(RAW_DIR):
        for f in files:
            if f.endswith(".txt") and "fullres" in f and "metadata" not in f:
                raw_file_map[f] = os.path.join(root, f)

    X_buffer = []
    y_buffer = []
    chunk_counter = 0
    counts = {0: 0, 1: 0, 2: 0}
    
    for root, dirs, files in os.walk(LABEL_DIR):
        for file in files:
            if file.endswith("fullres.clusters.txt"):
                raw_path = find_raw_file(file, raw_file_map)
                if raw_path is None: continue
                
                try:
                    # [유지] Transpose Fix (.T)
                    full_raw_img = np.loadtxt(raw_path).T
                    img_h, img_w = full_raw_img.shape
                    occupancy_mask = np.zeros_like(full_raw_img, dtype=bool)
                    
                    with open(os.path.join(root, file), 'r') as f:
                        clusters = json.load(f)
                        
                    # 1. Labelled Data
                    for cluster in clusters:
                        cls_data = cluster.get('cluster_class')
                        cls_name = cls_data.get('name') if isinstance(cls_data, dict) else cls_data
                        
                        if cls_name not in CLASS_PRIORITIES: continue
                        priority = CLASS_PRIORITIES[cls_name]
                        
                        if priority == 0 and random.random() > CLASS_0_KEEP_PROB: continue 
                        
                        pixels = cluster.get('pixels')
                        if not pixels: continue

                        # [유지] 절대 좌표 계산
                        anchor_x = cluster.get('pos_x', 0)
                        anchor_y = cluster.get('pos_y', 0)
                        xs = [int(anchor_x + p['x']) for p in pixels]
                        ys = [int(anchor_y + p['y']) for p in pixels]
                        
                        valid = True
                        for x, y in zip(xs, ys): 
                            if 0 <= y < img_h and 0 <= x < img_w:
                                occupancy_mask[y, x] = True
                            else: valid = False
                        if not valid: continue

                        min_x, max_x = min(xs), max(xs)
                        min_y, max_y = min(ys), max(ys)
                        roi = full_raw_img[min_y:max_y+1, min_x:max_x+1]
                        
                        h, w = roi.shape
                        if h == 0 or w == 0: continue

                        max_dim = max(h, w)
                        canvas = np.zeros((max_dim, max_dim), dtype=np.float32)
                        sy, sx = (max_dim - h) // 2, (max_dim - w) // 2
                        canvas[sy:sy+h, sx:sx+w] = roi
                        
                        img = Image.fromarray(canvas)
                        img = img.resize(TARGET_SIZE, resample=Image.NEAREST)
                        resized = np.array(img, dtype=np.float32)
                        
                        # 정규화
                        resized = np.clip(resized, 0, GLOBAL_MAX_VALUE)
                        resized /= GLOBAL_MAX_VALUE

                        # [NEW] 증강 적용!
                        aug_imgs = augment_image(resized, priority)
                        
                        for aug in aug_imgs:
                            X_buffer.append(aug[..., np.newaxis])
                            y_buffer.append(priority)
                            counts[priority] += 1

                    # 2. Noise Data (증강 안 함)
                    bg_count = 0
                    attempts = 0
                    while bg_count < BACKGROUND_SAMPLES_PER_FILE and attempts < 100:
                        attempts += 1
                        if random.random() > CLASS_0_KEEP_PROB: continue

                        rw, rh = random.randint(10, 50), random.randint(10, 50)
                        rx = random.randint(0, img_w - rw - 1)
                        ry = random.randint(0, img_h - rh - 1)
                        
                        if np.any(occupancy_mask[ry:ry+rh, rx:rx+rw]): continue
                        
                        roi = full_raw_img[ry:ry+rh, rx:rx+rw]
                        h, w = roi.shape
                        max_dim = max(h, w)
                        canvas = np.zeros((max_dim, max_dim), dtype=np.float32)
                        sy, sx = (max_dim - h) // 2, (max_dim - w) // 2
                        canvas[sy:sy+h, sx:sx+w] = roi
                        
                        img = Image.fromarray(canvas)
                        img = img.resize(TARGET_SIZE, resample=Image.NEAREST)
                        resized = np.array(img, dtype=np.float32)
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
                        print(f"   ...누적: Dot {counts[0]} | Track {counts[1]} | Blob {counts[2]}")
                        
                except Exception: continue

    if len(X_buffer) > 0:
        save_chunk(X_buffer, y_buffer, chunk_counter, OUTPUT_DIR)
    
    print(f"\n✅ V3 증강 데이터셋 완료! 최종: Dot {counts[0]} | Track {counts[1]} | Blob {counts[2]}")

if __name__ == '__main__':
    create_augmented_dataset()