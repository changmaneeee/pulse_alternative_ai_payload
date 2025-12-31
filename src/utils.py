# 파일 경로: src/utils.py
import numpy as np
import tensorflow as tf

def crop_and_resize_pad(full_img, pixels, target_size=(32, 32)):
    """
    입자 좌표(pixels)를 받아 정사각형 패딩(Square Padding) 후 
    비율 왜곡 없이 리사이징합니다.
    """
    # 1. 픽셀 정보가 없으면 건너뜀
    if not pixels:
        return None

    # 2. 입자를 감싸는 가장 작은 사각형(Bounding Box) 찾기
    pixels_np = np.array(pixels)
    min_y, min_x = np.min(pixels_np, axis=0)
    max_y, max_x = np.max(pixels_np, axis=0)
    
    # 3. 이미지 범위 벗어나지 않게 자르기 (ROI 추출)
    h_img, w_img = full_img.shape
    y1, y2 = max(0, min_y), min(h_img, max_y + 1)
    x1, x2 = max(0, min_x), min(w_img, max_x + 1)
    
    roi = full_img[y1:y2, x1:x2]
    
    # 4. 정사각형 만들기 (Square Padding)
    # 가로, 세로 중 더 긴 쪽을 찾습니다.
    roi_h, roi_w = roi.shape
    max_dim = max(roi_h, roi_w)
    
    if max_dim == 0: return None
    
    # 검은색(0) 정사각형 도화지를 만듭니다.
    padded_img = np.zeros((max_dim, max_dim), dtype=np.float32)
    
    # 잘라낸 입자를 도화지 정중앙에 붙입니다.
    start_h = (max_dim - roi_h) // 2
    start_w = (max_dim - roi_w) // 2
    padded_img[start_h:start_h+roi_h, start_w:start_w+roi_w] = roi
    
    # 5. 크기 줄이기 (Resize) & 정규화 (Normalize)
    padded_img = padded_img[..., np.newaxis] # 차원 추가 (H, W, 1)
    resized = tf.image.resize(padded_img, target_size, method='nearest').numpy()
    
    # 값을 0~1 사이로 맞춰줍니다. (학습이 잘 되게 하기 위함)
    max_val = np.max(resized)
    if max_val > 0:
        resized = resized / max_val
        
    return resized