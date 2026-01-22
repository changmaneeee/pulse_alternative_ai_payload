import os
import json
import numpy as np
import random

# ==========================================
# [설정] 정답지 확인 스크립트
# ==========================================
LABEL_DIR = '/home/changmin/pulse_alternative_ai_payload/data/vzlusat1-timepix-data/data/labelled/above_europe'
RAW_DIR = '/home/changmin/pulse_alternative_ai_payload/data/vzlusat1-timepix-data/data/raw/above_europe'

def scan_nonzero_locations():
    print(">>> 📡 [정답지 확인] JSON 좌표 vs 실제 데이터 위치 비교...")

    # 1. 파일 목록 생성
    raw_file_map = {}
    for root, dirs, files in os.walk(RAW_DIR):
        for f in files:
            if f.endswith(".txt") and "fullres" in f and "metadata" not in f:
                raw_file_map[f] = os.path.join(root, f)

    label_files = []
    for root, dirs, files in os.walk(LABEL_DIR):
        for f in files:
            if f.endswith("fullres.clusters.txt"):
                label_files.append((root, f))
    
    random.shuffle(label_files)
    found = False

    for root, file in label_files:
        if found: break

        raw_name = file.replace('.clusters.txt', '.txt')
        raw_path = raw_file_map.get(raw_name)
        if raw_path is None: continue

        # 로드
        try:
            full_raw = np.loadtxt(raw_path)
            img_h, img_w = full_raw.shape
            
            with open(os.path.join(root, file), 'r') as f:
                clusters = json.load(f)
        except: continue

        # 의미 있는 데이터(Track/Blob) 찾기
        for cluster in clusters:
            cls_data = cluster.get('cluster_class')
            cls_name = cls_data.get('name') if isinstance(cls_data, dict) else cls_data
            
            # 노이즈(Dot)는 너무 작아서 패턴 찾기 힘드니 제외
            if 'track' in cls_name or 'blob' in cls_name:
                pixels = cluster.get('pixels')
                if not pixels: continue
                
                # -------------------------------------------------
                # 1. JSON 주장 (Calculated)
                # -------------------------------------------------
                pos_x = cluster.get('pos_x', 0)
                pos_y = cluster.get('pos_y', 0)
                
                print(f"\n📄 분석 대상 파일: {raw_name}")
                print(f"🧩 클래스: {cls_name}")
                print(f"📍 [JSON 주장] 기준점(Anchor): pos_x={pos_x}, pos_y={pos_y}")
                
                json_coords = []
                for p in pixels:
                    abs_x = int(pos_x + p['x'])
                    abs_y = int(pos_y + p['y'])
                    val = p.get('value', 0)
                    json_coords.append(f"(x={abs_x}, y={abs_y}, v={val})")
                
                print(f"   -> JSON이 가리키는 픽셀들 (앞쪽 5개만):")
                for c in json_coords[:5]:
                    print(f"      {c}")

                # -------------------------------------------------
                # 2. 실제 Raw 데이터의 진실 (Non-zero pixels)
                # -------------------------------------------------
                # 실제 데이터에서 값이 0보다 큰 좌표를 모두 찾음
                real_ys, real_xs = np.nonzero(full_raw)
                
                print(f"\n🔥 [실제 데이터] Raw 파일에서 값이 0이 아닌 좌표들 ({len(real_ys)}개):")
                
                # 너무 많으면 일부만 출력
                count = 0
                for r_y, r_x in zip(real_ys, real_xs):
                    r_val = full_raw[r_y, r_x]
                    # JSON 좌표 근처에 있는 게 아니라, 그냥 쌩뚱맞은 위치라도 다 출력
                    # 단, JSON 값과 비슷한지 확인
                    print(f"      Real(y={r_y}, x={r_x}) => Value={r_val}")
                    count += 1
                    if count >= 10: 
                        print("      ... (생략)")
                        break
                
                print("\n🧐 [분석 힌트]")
                print("1. JSON의 (x, y)와 Real의 (y, x) 숫자가 서로 뒤바껴 있나요? -> Transpose 필요 (.T)")
                print("2. y좌표 합이 255인가요? (예: JSON y=10, Real y=245) -> 상하 반전 필요")
                print("3. x좌표 합이 255인가요? -> 좌우 반전 필요")
                
                found = True
                break

if __name__ == '__main__':
    scan_nonzero_locations()