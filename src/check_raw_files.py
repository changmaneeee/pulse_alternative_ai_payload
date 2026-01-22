import os
import numpy as np
import glob

# 원본 텍스트 파일이 있는 경로 (사용자님 설정에 맞춤)
RAW_TXT_DIR = '../data/vzlusat1-timepix-data/data/raw'

def check_real_max_value():
    # .txt 파일 찾기
    txt_files = []
    for root, dirs, files in os.walk(RAW_TXT_DIR):
        for f in files:
            if f.endswith(".txt") and "fullres" in f:
                txt_files.append(os.path.join(root, f))
    
    if not txt_files:
        print("❌ .txt 파일을 찾을 수 없습니다.")
        return

    print(f"🔎 총 {len(txt_files)}개의 원본 파일 중 앞쪽 5개만 스캔합니다...")
    
    global_max = 0
    
    for i, f_path in enumerate(txt_files[:5]):
        try:
            # 텍스트 파일 로드
            img = np.loadtxt(f_path)
            local_max = np.max(img)
            print(f"   [{i+1}] 파일 Max 값: {local_max}")
            
            if local_max > global_max:
                global_max = local_max
                
        except Exception as e:
            print(f"   Error reading {f_path}: {e}")

    print("\n" + "="*40)
    print(f"🎯 [결론] 원본 데이터의 최대값은 약 '{global_max}' 입니다.")
    print("="*40)
    
    # 솔루션 제안
    if global_max <= 1.0:
        print("👉 솔루션: GLOBAL_MAX_VALUE = 1.0 으로 설정하세요. (나누기 금지)")
    elif global_max <= 10.0:
        print("👉 솔루션: GLOBAL_MAX_VALUE = 10.0 (또는 20.0) 정도로 설정하세요.")
    elif global_max <= 100.0:
        print("👉 솔루션: GLOBAL_MAX_VALUE = 100.0 으로 설정하세요.")
    else:
        print("👉 솔루션: 현재 255.0 설정이 맞는데, 왜 검게 나올까요? (매우 드문 케이스)")

if __name__ == '__main__':
    check_real_max_value()