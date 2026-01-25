import os
import json
import numpy as np
import glob
from tqdm import tqdm  # 진행률 표시용 (없으면 pip install tqdm)

# =========================================================
# 1. 설정 (경로를 정확히 맞춰주세요!)
# =========================================================
# 이미지 데이터가 있는 최상위 폴더
BASE_RAW_DIR = "/home/changmin/pulse_alternative_ai_payload/data/vzlusat1-timepix-data/data/raw"
# 라벨 데이터가 있는 최상위 폴더
BASE_LABEL_DIR = "/home/changmin/pulse_alternative_ai_payload/data/vzlusat1-timepix-data/data/labelled"

# 저장할 파일명
OUTPUT_FILE = "dataset_bulid/dataset_final_v1.npz"

# AI 입력 크기
TARGET_SIZE = 32

# =========================================================
# 2. 핵심 로직 함수
# =========================================================

def get_label_from_stats(raw_path):
    """
    Raw 이미지 경로를 기반으로 Labelled 폴더의 통계 파일을 찾아
    우선순위 라벨(0, 1, 2)을 반환합니다.
    """
    # 경로 변환: raw -> labelled, .txt -> .statistics.txt
    # 예: .../data/raw/above_europe/1_fullres.txt
    #  -> .../data/labelled/above_europe/1_fullres.statistics.txt
    
    # 1. 'raw'를 'labelled'로 치환
    stats_path = raw_path.replace("/data/raw/", "/data/labelled/")
    # 2. 확장자 변경
    stats_path = stats_path.replace("_fullres.txt", "_fullres.statistics.txt")
    
    if not os.path.exists(stats_path):
        return 0 # 파일이 없으면 그냥 Noise로 취급
        
    try:
        with open(stats_path, 'r') as f:
            stats = json.load(f)
            
        # [우선순위 로직]
        # 1. Track
        if (stats.get('track_straight', 0) + 
            stats.get('track_curly', 0) + 
            stats.get('track_lowres', 0)) > 0:
            return 1 
            
        # 2. Blob
        if (stats.get('blob_big', 0) + 
            stats.get('blob_small', 0) + 
            stats.get('blob_branched', 0)) > 0:
            return 2
            
        # 3. Noise
        return 0
        
    except Exception:
        return 0

def process_image(txt_path):
    """이미지 로드 -> Transpose -> MaxPool -> Normalize"""
    try:
        # 1. Load & Transpose (.T 필수!)
        img = np.loadtxt(txt_path, dtype=np.float32).T
        
        # 2. Max Pooling (256 -> 32)
        h, w = img.shape
        factor = h // TARGET_SIZE
        if factor > 1:
            img = img.reshape(TARGET_SIZE, factor, TARGET_SIZE, factor).max(axis=(1, 3))
        
        # 3. Normalize (Log + MinMax)
        img_log = np.log1p(img)
        if img_log.max() > 0:
            img_norm = img_log / img_log.max()
        else:
            img_norm = img_log # 0인 경우
            
        return np.expand_dims(img_norm, axis=-1) # (32, 32, 1)
        
    except Exception:
        return None

# =========================================================
# 3. 메인 실행
# =========================================================
def main():
    print(f"🚀 Building Final Dataset...")
    print(f"   - Raw Source: {BASE_RAW_DIR}")
    print(f"   - Label Source: {BASE_LABEL_DIR}")
    
    # 모든 raw 파일 찾기 (재귀 검색)
    search_pattern = os.path.join(BASE_RAW_DIR, "**", "*_fullres.txt")
    raw_files = glob.glob(search_pattern, recursive=True)
    print(f"📂 Found {len(raw_files)} image files.")
    
    X_data = []
    y_data = []
    
    # 진행률 표시하며 처리
    for raw_path in tqdm(raw_files, desc="Processing"):
        # 이미지 처리
        img = process_image(raw_path)
        if img is None: continue
        
        # 라벨 결정
        label = get_label_from_stats(raw_path)
        
        X_data.append(img)
        y_data.append(label)
        
    # 배열 변환
    X_final = np.array(X_data, dtype=np.float32)
    y_final = np.array(y_data, dtype=np.int8) # 라벨은 int8로 충분
    
    # 최종 정보 출력
    print("\n" + "="*40)
    print(f"✅ Build Complete!")
    print(f"📊 Shape: X={X_final.shape}, y={y_final.shape}")
    
    unique, counts = np.unique(y_final, return_counts=True)
    print("📈 Final Distribution:")
    for u, c in zip(unique, counts):
        print(f"   Class {u}: {c} samples ({c/len(y_final)*100:.1f}%)")
        
    # 저장
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    np.savez_compressed(OUTPUT_FILE, X=X_final, y=y_final)
    print(f"💾 Saved to: {OUTPUT_FILE}")
    print("="*40)

if __name__ == "__main__":
    main()