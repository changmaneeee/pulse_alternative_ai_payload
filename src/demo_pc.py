import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from scipy.ndimage import label, find_objects

# ==========================================
# [설정] Real-World 시뮬레이션
# ==========================================
LABEL_DIR = '../data/vzlusat1-timepix-data/data/labelled' 
RAW_DIR = '../data/vzlusat1-timepix-data/data/raw' 
MODEL_PATH = '../models/DSC_CNN_NOISY.tflite' # 방금 학습한 모델

# [Detection 파라미터]
# "위치를 찾기 위해서"는 Threshold가 필요함.
DETECTION_THRESHOLD = 1.0 
MIN_PIXEL_COUNT = 3 # 핫픽셀 제거용 (최소한의 필터)

CLASS_PRIORITIES = {
    'dot': 0, 'drop': 0, 'other': 0,
    'track_straight': 1, 'track_curly': 1, 'track_lowres': 1,
    'blob_big': 2, 'blob_small': 2, 'blob_branched': 2
}
PRIORITY_NAMES = {0: 'Low (Dot)', 1: 'Med (Track)', 2: 'High (Blob)'}

def load_tflite_model(model_path):
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return None

def find_raw_file(json_filename, raw_file_map):
    base_name = json_filename.replace('.clusters.txt', '.txt')
    return raw_file_map.get(base_name)

def onboard_detection_simulation(full_raw_img):
    """
    [위성 탑재 로직 모사]
    1. Threshold(1.0)로 덩어리 위치(Box)를 찾음.
    2. Box 안의 이미지는 노이즈를 제거하지 않고 원본 그대로 AI에게 줌.
    """
    # 1. 위치 찾기 (Binary Masking)
    binary_img = full_raw_img >= DETECTION_THRESHOLD
    labeled_array, num_features = label(binary_img)
    objects_slices = find_objects(labeled_array)
    
    detected_crops = []
    detected_boxes = [] 
    
    for idx, slices in enumerate(objects_slices):
        if slices is None: continue
        
        y_slice, x_slice = slices
        
        # Size Filtering (너무 작은 핫픽셀 무시)
        obj_indices = np.where(labeled_array[y_slice, x_slice] == (idx + 1))
        if len(obj_indices[0]) < MIN_PIXEL_COUNT:
            continue

        # 2. Crop (Noisy Raw Data)
        # [핵심] 여기서 전처리를 안 합니다! 원본의 자글자글한 값을 그대로 가져옵니다.
        roi = full_raw_img[y_slice, x_slice]
        
        # 캔버스 만들기
        h, w = roi.shape
        max_dim = max(h, w)
        canvas = np.zeros((max_dim, max_dim), dtype=np.float32)
        start_y = (max_dim - h) // 2
        start_x = (max_dim - w) // 2
        
        # ROI 붙여넣기
        canvas[start_y:start_y+h, start_x:start_x+w] = roi
        
        # Resize & Normalize
        img = Image.fromarray(canvas)
        img = img.resize((32, 32), resample=Image.NEAREST)
        resized = np.array(img, dtype=np.float32)
        
        max_val = np.max(resized)
        if max_val > 0:
            resized = resized / max_val
            
        detected_crops.append(resized[np.newaxis, ..., np.newaxis])
        
        # 박스 좌표 저장 (min_x, min_y, max_x, max_y)
        detected_boxes.append((x_slice.start, y_slice.start, x_slice.stop-1, y_slice.stop-1))
        
    return detected_crops, detected_boxes

def match_with_ground_truth(det_box, clusters):
    """
    내가 찾은 박스가 정답지(Label)에 있는지 확인 (채점용)
    """
    dx_min, dy_min, dx_max, dy_max = det_box
    det_center_x = (dx_min + dx_max) / 2
    det_center_y = (dy_min + dy_max) / 2
    
    for cluster in clusters:
        cls_data = cluster.get('cluster_class')
        cls_name = cls_data.get('name') if isinstance(cls_data, dict) else cls_data
        
        if cls_name not in CLASS_PRIORITIES: continue
        
        pixels = cluster.get('pixels')
        if not pixels: continue
        xs = [p['x'] for p in pixels]
        ys = [p['y'] for p in pixels]
        
        gt_min_x, gt_max_x = min(xs), max(xs)
        gt_min_y, gt_max_y = min(ys), max(ys)
        
        # 중심점이 정답 박스 안에 있으면 정답으로 인정
        if (gt_min_x <= det_center_x <= gt_max_x) and \
           (gt_min_y <= det_center_y <= gt_max_y):
            return CLASS_PRIORITIES[cls_name]
            
    return None

def run_demo(limit=50):
    print(f">>> 📡 PC Ground Demo (Noisy Input Ver.)")
    print(f">>> Model: {MODEL_PATH}")
    
    interpreter = load_tflite_model(MODEL_PATH)
    if interpreter is None: return
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    raw_file_map = {}
    for root, dirs, files in os.walk(RAW_DIR):
        for f in files:
            if f.endswith(".txt") and "fullres" in f:
                raw_file_map[f] = os.path.join(root, f)
    
    correct = 0
    total_matched = 0
    results = {0: {'ok':0, 'tot':0}, 1: {'ok':0, 'tot':0}, 2: {'ok':0, 'tot':0}}
    
    print(f">>> 시뮬레이션 시작 (최대 {limit} 파일)...")

    file_cnt = 0
    for root, dirs, files in os.walk(LABEL_DIR):
        if file_cnt >= limit: break
        
        for file in files:
            if file.endswith("fullres.clusters.txt"):
                raw_path = find_raw_file(file, raw_file_map)
                if raw_path is None: continue
                
                full_raw_img = np.loadtxt(raw_path)
                
                # 1. Detection (Noisy Input 생성됨)
                crops, boxes = onboard_detection_simulation(full_raw_img)
                if len(crops) == 0: continue
                
                with open(os.path.join(root, file), 'r') as f:
                    gt_clusters = json.load(f)
                
                # 2. Inference & Scoring
                for crop, box in zip(crops, boxes):
                    interpreter.set_tensor(input_details[0]['index'], crop)
                    interpreter.invoke()
                    output_data = interpreter.get_tensor(output_details[0]['index'])
                    pred_priority = np.argmax(output_data)
                    
                    gt_priority = match_with_ground_truth(box, gt_clusters)
                    
                    if gt_priority is not None:
                        total_matched += 1
                        results[gt_priority]['tot'] += 1
                        if pred_priority == gt_priority:
                            correct += 1
                            results[gt_priority]['ok'] += 1
                            
                file_cnt += 1
                if file_cnt % 10 == 0:
                    acc = (correct / total_matched * 100) if total_matched > 0 else 0
                    print(f"   Processed {file_cnt} files | Matched {total_matched} | Acc: {acc:.1f}%")
                
                if file_cnt >= limit: break

    print("\n" + "="*40)
    print("🏆 [Final Result: Noisy Training & Noisy Inference]")
    print("="*40)
    print(f"Files: {file_cnt}")
    print(f"Matched Particles: {total_matched}")
    
    if total_matched > 0:
        print(f"Final Accuracy: {correct/total_matched*100:.2f}%")
        print("-" * 20)
        for p in [0, 1, 2]:
            tot = results[p]['tot']
            ok = results[p]['ok']
            acc = (ok/tot*100) if tot > 0 else 0
            print(f"Priority {p} ({PRIORITY_NAMES[p]}): {acc:.1f}% ({ok}/{tot})")
    print("="*40)

if __name__ == '__main__':
    run_demo(limit=50)