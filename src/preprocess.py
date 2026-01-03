import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from utils import crop_and_resize_pad
import time
import gc

# ==========================================
# 설정
# ==========================================
DATA_ROOT_PATH = '../data/vzlusat1-timepix-data/data/labelled' 
SAVE_DIR = '../data/dataset'
os.makedirs(SAVE_DIR, exist_ok=True)

INPUT_SHAPE = (32, 32)
CHUNK_SIZE = 10000  # [수정] 10만 -> 1만으로 대폭 축소 (안전 제일)

CLASS_PRIORITIES = {
    'dot': 0, 'drop': 0, 'other': 0,
    'track_straight': 1, 'track_curly': 1, 'track_lowres': 1,
    'blob_big': 2, 'blob_small': 2, 'blob_branched': 2
}

def save_chunk(X, y, chunk_id):
    """데이터를 저장하고 메모리에서 즉시 해제합니다."""
    if len(X) == 0: return
    
    # 1. 분할
    try:
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    except ValueError:
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # 2. 저장
    np.savez(os.path.join(SAVE_DIR, f'part_{chunk_id}_train.npz'), X=X_train, y=y_train)
    np.savez(os.path.join(SAVE_DIR, f'part_{chunk_id}_val.npz'), X=X_val, y=y_val)
    np.savez(os.path.join(SAVE_DIR, f'part_{chunk_id}_test.npz'), X=X_test, y=y_test)
    
    print(f"   💾 [Chunk {chunk_id}] Saved! ({len(X)} particles)")
    
    # [중요] 함수 내부 변수 삭제
    del X_train, X_temp, y_train, y_temp, X_val, X_test, y_val, y_test
    gc.collect()

def extract_data():
    X_chunk = []
    y_chunk = []
    
    print(f">>> [Ultra Safe Mode] 데이터 탐색 시작 (Chunk: {CHUNK_SIZE})")
    start_time = time.time()
    
    chunk_counter = 0
    total_particles = 0
    file_count = 0
    
    for root, dirs, files in os.walk(DATA_ROOT_PATH):
        for file in files:
            if file.endswith("fullres.clusters.txt"):
                json_path = os.path.join(root, file)
                file_count += 1
                
                if file_count % 2000 == 0:
                    elapsed = time.time() - start_time
                    print(f"   Processing... {file_count} files scan. (Total Particles: {total_particles}, Time: {elapsed:.1f}s)")

                try:
                    with open(json_path, 'r') as f:
                        clusters = json.load(f)
                    
                    for cluster in clusters:
                        cls_data = cluster.get('cluster_class')
                        if isinstance(cls_data, dict):
                            cls_name = cls_data.get('name')
                        else:
                            cls_name = cls_data

                        if cls_name in CLASS_PRIORITIES:
                            priority = CLASS_PRIORITIES[cls_name]
                            pixels = cluster.get('pixels')
                            
                            processed_img = crop_and_resize_pad(pixels, INPUT_SHAPE)
                            
                            if processed_img is not None:
                                X_chunk.append(processed_img)
                                y_chunk.append(priority)
                                total_particles += 1
                    
                    # 사용한 JSON 객체 즉시 삭제
                    del clusters
                    
                    # === CHUNK 저장 로직 ===
                    if len(X_chunk) >= CHUNK_SIZE:
                        save_chunk(np.array(X_chunk), np.array(y_chunk), chunk_counter)
                        
                        chunk_counter += 1
                        # 리스트 초기화 (새로운 객체 할당)
                        del X_chunk, y_chunk
                        X_chunk = []
                        y_chunk = []
                        gc.collect() # 쓰레기 수집
                        
                except Exception:
                    continue

    # 남은 데이터 저장
    if len(X_chunk) > 0:
        save_chunk(np.array(X_chunk), np.array(y_chunk), chunk_counter)

    print(f"\n✅ 전체 완료! 총 {total_particles}개 입자 처리됨.")
    print(f"   데이터는 {SAVE_DIR} 폴더에 분할 저장되었습니다.")

if __name__ == '__main__':
    extract_data()