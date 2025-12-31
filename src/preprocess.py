# 파일 경로: src/01_preprocess.py
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from utils import crop_and_resize_pad

# ==========================================
# [설정] 경로를 내 환경에 맞게 꼭 확인하세요!
# ==========================================
# 예: '../data/폴더명/data/labelled'
DATA_ROOT_PATH = '/home/changmin/pulse_alternative_ai_payload/data/vzlusat1-timepix-data/data/labelled' 
SAVE_DIR = '../data/dataset'

# 폴더가 없으면 자동으로 만듭니다.
os.makedirs(SAVE_DIR, exist_ok=True)

# AI에게 보여줄 이미지 크기
INPUT_SHAPE = (32, 32)

# 우리가 정한 우선순위 규칙 (Class Mapping)
CLASS_PRIORITIES = {
    # Priority 0: Low (무시/카운트만)
    'dot': 0, 'drop': 0, 'other': 0,
    
    # Priority 1: Medium (압축 저장)
    'track_straight': 1, 'track_curly': 1, 'track_lowres': 1,
    
    # Priority 2: High (긴급/원본 저장)
    'blob_big': 2, 'blob_small': 2, 'blob_branched': 2
}

def extract_data():
    X_all = [] # 이미지 데이터를 담을 리스트
    y_all = [] # 정답(라벨)을 담을 리스트
    
    print(f">>> 데이터를 찾고 있습니다: {DATA_ROOT_PATH}")
    
    # 모든 폴더를 돌면서 파일을 찾습니다.
    for root, dirs, files in os.walk(DATA_ROOT_PATH):
        for file in files:
            # 라벨 파일(.clusters.txt)을 발견하면 작업을 시작합니다.
            if file.endswith("fullres.clusters.txt"):
                json_path = os.path.join(root, file)
                img_path = json_path.replace(".clusters.txt", ".txt")
                
                # 원본 이미지 파일이 없으면 넘어갑니다.
                if not os.path.exists(img_path): continue
                
                try:
                    # 파일 읽기
                    with open(json_path, 'r') as f:
                        clusters = json.load(f)
                    full_image = np.loadtxt(img_path)
                    
                    # 이미지 안에 있는 입자 하나하나를 꺼냅니다.
                    for cluster in clusters:
                        cls_name = cluster.get('class')
                        pixels = cluster.get('pixels')
                        
                        # 우리가 아는 종류의 입자라면?
                        if cls_name in CLASS_PRIORITIES:
                            priority = CLASS_PRIORITIES[cls_name]
                            
                            # utils.py의 함수로 예쁘게 잘라옵니다.
                            processed = crop_and_resize_pad(full_image, pixels, INPUT_SHAPE)
                            
                            if processed is not None:
                                X_all.append(processed)
                                y_all.append(priority)
                                
                except Exception as e:
                    print(f"에러 발생 ({file}): {e}")
                    continue

    # 리스트를 numpy 배열로 변환
    X_all = np.array(X_all)
    y_all = np.array(y_all)
    
    print(f"\n>>> 총 {len(X_all)}개의 입자 데이터를 찾았습니다!")

    if len(X_all) == 0:
        print("❌ 데이터를 하나도 못 찾았습니다. DATA_ROOT_PATH 경로를 다시 확인해주세요.")
        return

    # 데이터를 학습용(Train), 검증용(Val), 테스트용(Test)으로 나눕니다.
    # 비율은 대략 80% : 10% : 10% 입니다.
    print(">>> 데이터를 나누고 저장하는 중...")
    
    X_train, X_temp, y_train, y_temp = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    
    # 파일로 저장 (.npz)
    np.savez(os.path.join(SAVE_DIR, 'train.npz'), X=X_train, y=y_train)
    np.savez(os.path.join(SAVE_DIR, 'val.npz'), X=X_val, y=y_val)
    np.savez(os.path.join(SAVE_DIR, 'test.npz'), X=X_test, y=y_test)
    
    print(f"✅ 완료! 저장 위치: {SAVE_DIR}")
    print(f"   - 학습용: {len(X_train)}개")
    print(f"   - 검증용: {len(X_val)}개")
    print(f"   - 테스트용: {len(X_test)}개")

if __name__ == '__main__':
    extract_data()