import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from utils import crop_and_resize_pad

# ==========================================
# [설정] 데이터 경로 확인
# ==========================================
# Git Clone한 실제 경로를 입력하세요.
# (폴더 구조: data/labelled/planetary_sweeps/...)
DATA_ROOT_PATH = '../data/vzlusat1-timepix-data/data/labelled' 

# 생성된 데이터셋을 저장할 위치
SAVE_DIR = '../data/dataset'
os.makedirs(SAVE_DIR, exist_ok=True)

# 모델 입력 크기 (Teensy 4.0 최적화)
INPUT_SHAPE = (32, 32)

# [우선순위 매핑 정책]
# 이 이름들과 일치하는 입자만 추출하여 학습합니다.
CLASS_PRIORITIES = {
    # Priority 0: Low (무시, 카운트만 함)
    'dot': 0, 
    'drop': 0, 
    'other': 0,
    
    # Priority 1: Medium (압축 저장)
    'track_straight': 1, 
    'track_curly': 1, 
    'track_lowres': 1,
    
    # Priority 2: High (긴급/원본 저장)
    'blob_big': 2, 
    'blob_small': 2, 
    'blob_branched': 2
}

def extract_data():
    X_all = []
    y_all = []
    
    print(f">>> 데이터 탐색 시작: {DATA_ROOT_PATH}")
    
    # 1. 파일 탐색 (os.walk가 모든 하위 폴더를 자동으로 뒤집니다)
    # planetary_sweeps, saa_and_poles 등 모든 폴더를 통합해서 처리합니다.
    file_count = 0
    
    for root, dirs, files in os.walk(DATA_ROOT_PATH):
        for file in files:
            # 우리는 오직 '정답지(Cluster JSON)'만 찾습니다.
            if file.endswith("fullres.clusters.txt"):
                json_path = os.path.join(root, file)
                file_count += 1
                
                try:
                    # JSON 파일 로드
                    with open(json_path, 'r') as f:
                        clusters = json.load(f)
                    
                    # 2. 입자 데이터 추출 및 재구성
                    for cluster in clusters:
                        # 클래스 이름 추출 (구조: cluster_class -> name)
                        cls_data = cluster.get('cluster_class')
                        if isinstance(cls_data, dict):
                            cls_name = cls_data.get('name')
                        else:
                            cls_name = cls_data # 혹시 문자열로 되어있을 경우 대비

                        # 픽셀 데이터 리스트 [{'x':.., 'y':.., 'value':..}, ...]
                        pixels = cluster.get('pixels') 
                        
                        # 우리가 정의한 우선순위 목록에 있는 입자인가?
                        if cls_name in CLASS_PRIORITIES:
                            priority = CLASS_PRIORITIES[cls_name]
                            
                            # [핵심] utils.py의 함수가 픽셀 정보를 이용해 이미지를 '재구성'합니다.
                            # 원본 이미지 파일(.txt)을 읽지 않으므로 경로 꼬일 걱정이 없습니다.
                            processed_img = crop_and_resize_pad(pixels, INPUT_SHAPE)
                            
                            if processed_img is not None:
                                X_all.append(processed_img)
                                y_all.append(priority)
                                
                except Exception as e:
                    print(f"⚠️ 파일 처리 중 에러 발생 ({file}): {e}")
                    continue

    print(f"\n>>> 탐색 완료: 총 {file_count}개의 파일 스캔됨.")
    
    # 3. 결과 확인
    X_all = np.array(X_all)
    y_all = np.array(y_all)
    
    print(f">>> 추출된 입자 데이터 총 개수: {len(X_all)}개")

    if len(X_all) == 0:
        print("❌ 데이터를 하나도 찾지 못했습니다. DATA_ROOT_PATH 경로를 다시 확인해주세요!")
        return

    # 4. 데이터 분할 (Train / Val / Test)
    # 비율: Train 80%, Val 10%, Test 10%
    print(">>> 데이터셋 분할 및 저장 중...")
    
    # 먼저 Train(80%)과 Temp(20%)로 나눔
    X_train, X_temp, y_train, y_temp = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all, random_state=42)
    
    # Temp를 다시 Val(10%)과 Test(10%)로 나눔
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    
    # 5. 파일 저장 (.npz)
    np.savez(os.path.join(SAVE_DIR, 'train.npz'), X=X_train, y=y_train)
    np.savez(os.path.join(SAVE_DIR, 'val.npz'), X=X_val, y=y_val)
    np.savez(os.path.join(SAVE_DIR, 'test.npz'), X=X_test, y=y_test)
    
    print(f"✅ 데이터셋 생성 완료! 저장 위치: {SAVE_DIR}")
    print(f"   - 학습용(Train): {len(X_train)}개")
    print(f"   - 검증용(Val)  : {len(X_val)}개")
    print(f"   - 테스트용(Test) : {len(X_test)}개")

if __name__ == '__main__':
    extract_data()