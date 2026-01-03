import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import glob

# ==========================================
# 설정 (Settings)
# ==========================================
# 검증할 데이터 경로 (Labelled 폴더)
DATA_ROOT_PATH = '../data/vzlusat1-timepix-data/data/labelled' 

# 테스트할 모델 파일 경로 (학습 결과에 따라 변경하세요!)
# 예: 가장 성능 좋았던 모델 선택
MODEL_PATH = '../models/dsc_cnn.tflite' 

# 우선순위 매핑 (정답 확인용)
CLASS_PRIORITIES = {
    'dot': 0, 'drop': 0, 'other': 0,
    'track_straight': 1, 'track_curly': 1, 'track_lowres': 1,
    'blob_big': 2, 'blob_small': 2, 'blob_branched': 2
}
PRIORITY_NAMES = {0: 'Low (Dot)', 1: 'Med (Track)', 2: 'High (Blob)'}

def load_tflite_model(model_path):
    """Teensy와 동일한 환경(TFLite Interpreter)을 PC에 구성"""
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return None

def crop_from_raw_image(full_img, pixels, target_size=(32, 32)):
    """
    [핵심] 원본 Raw 이미지(256x256)에서 좌표를 이용해 입자를 잘라냄.
    이때 노이즈가 포함된 원본 픽셀값이 그대로 들어감.
    """
    if not pixels: return None
    
    # JSON 좌표 추출
    xs = [p['x'] for p in pixels]
    ys = [p['y'] for p in pixels]
    
    # Bounding Box
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    # ROI 추출 (Index 주의)
    # 실제 이미지는 (256, 256) numpy array
    roi = full_img[min_y:max_y+1, min_x:max_x+1]
    
    # Square Padding
    h, w = roi.shape
    max_dim = max(h, w)
    
    # 검은 캔버스 생성 (노이즈가 없는 배경) -> 실제 환경에서는 배경값(Bias)을 뺄셈 처리함
    # 여기서는 Raw Crop의 특성을 살리기 위해 0으로 초기화하되, ROI 값을 그대로 복사
    canvas = np.zeros((max_dim, max_dim), dtype=np.float32)
    start_y = (max_dim - h) // 2
    start_x = (max_dim - w) // 2
    canvas[start_y:start_y+h, start_x:start_x+w] = roi

    # Resize (PIL 사용)
    img = Image.fromarray(canvas)
    img = img.resize(target_size, resample=Image.NEAREST)
    resized = np.array(img, dtype=np.float32)
    
    # Normalize (0~1)
    if np.max(resized) > 0:
        resized = resized / np.max(resized)
        
    # (32, 32) -> (1, 32, 32, 1) : TFLite 입력 형태
    return resized[np.newaxis, ..., np.newaxis]

def run_demo(limit=100):
    print(f">>> 📡 PC Ground Demo 시작")
    print(f">>> Target Model: {MODEL_PATH}")
    
    interpreter = load_tflite_model(MODEL_PATH)
    if interpreter is None: return

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    correct = 0
    total = 0
    results = {0: {'ok':0, 'tot':0}, 1: {'ok':0, 'tot':0}, 2: {'ok':0, 'tot':0}}

    print(f">>> Raw 데이터 탐색 및 추론 중... (최대 {limit}개)")
    
    for root, dirs, files in os.walk(DATA_ROOT_PATH):
        if total >= limit: break
        
        for file in files:
            if total >= limit: break
            
            # JSON 라벨 파일을 찾음
            if file.endswith("fullres.clusters.txt"):
                json_path = os.path.join(root, file)
                # 짝꿍인 Raw 이미지 파일 찾기 (.clusters.txt -> .txt)
                raw_txt_path = json_path.replace(".clusters.txt", ".txt")
                
                if not os.path.exists(raw_txt_path): continue
                
                try:
                    # 1. Raw 이미지 로드 (256x256 텍스트 파일)
                    full_raw_img = np.loadtxt(raw_txt_path)
                    
                    # 2. JSON 로드
                    with open(json_path, 'r') as f:
                        clusters = json.load(f)
                    
                    for cluster in clusters:
                        if total >= limit: break

                        # 정답 확인
                        cls_data = cluster.get('cluster_class')
                        cls_name = cls_data.get('name') if isinstance(cls_data, dict) else cls_data
                        
                        if cls_name not in CLASS_PRIORITIES: continue
                        
                        gt_priority = CLASS_PRIORITIES[cls_name]
                        pixels = cluster.get('pixels')

                        # 3. 전처리 (Crop & Resize)
                        input_data = crop_from_raw_image(full_raw_img, pixels)
                        if input_data is None: continue

                        # 4. TFLite 추론 (Inference)
                        interpreter.set_tensor(input_details[0]['index'], input_data)
                        interpreter.invoke()
                        output_data = interpreter.get_tensor(output_details[0]['index'])
                        
                        # 결과 해석
                        pred_priority = np.argmax(output_data)
                        
                        # 통계
                        total += 1
                        results[gt_priority]['tot'] += 1
                        if pred_priority == gt_priority:
                            correct += 1
                            results[gt_priority]['ok'] += 1
                        
                        # 진행상황 (10개마다)
                        if total % 10 == 0:
                            print(f"   Processed {total}/{limit} | Current Acc: {correct/total*100:.1f}%")

                except Exception as e:
                    print(f"Skipping {file}: {e}")
                    continue

    # 최종 리포트
    print("\n" + "="*40)
    print("🏆 [PC Demo Result on RAW Data]")
    print("="*40)
    print(f"Total Samples: {total}")
    print(f"Final Accuracy: {correct/total*100:.2f}%")
    print("-" * 20)
    for p in [0, 1, 2]:
        tot = results[p]['tot']
        ok = results[p]['ok']
        acc = (ok/tot*100) if tot > 0 else 0
        print(f"Priority {p} ({PRIORITY_NAMES[p]}): {acc:.1f}% ({ok}/{tot})")
    print("="*40)

if __name__ == '__main__':
    # 3개 모델 다 있으면 여기만 바꿔가며 테스트 가능
    # MODEL_PATH = '../models/Simple_CNN.tflite'
    run_demo(limit=500) # 500개 입자만 테스트