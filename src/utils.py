# src/utils.py
import numpy as np
from PIL import Image

def crop_and_resize_pad(pixels_data, target_size=(32, 32)):
    """
    [CPU Optimized] JSON 픽셀 정보를 받아 이미지를 재구성하고
    PIL을 사용하여 빠르게 리사이징합니다.
    """
    if not pixels_data:
        return None

    try:
        xs = [p['x'] for p in pixels_data]
        ys = [p['y'] for p in pixels_data]
        values = [p['value'] for p in pixels_data]
    except (KeyError, TypeError):
        return None

    # 1. Bounding Box
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    
    # 2. Square Canvas (Padding)
    max_dim = max(width, height)
    
    # PIL 변환을 위해 uint8 혹은 float32 캔버스 생성
    # 여기서는 리사이징 정확도를 위해 임시 캔버스 사용
    canvas = np.zeros((max_dim, max_dim), dtype=np.float32)
    
    start_x = (max_dim - width) // 2
    start_y = (max_dim - height) // 2
    
    for x, y, val in zip(xs, ys, values):
        px = int(start_x + (x - min_x))
        py = int(start_y + (y - min_y))
        if 0 <= px < max_dim and 0 <= py < max_dim:
            canvas[py, px] = val

    # 3. Fast Resize using PIL (CPU)
    # PIL은 array -> Image 변환 필요
    img = Image.fromarray(canvas)
    
    # 리사이징 (BILINEAR or NEAREST) - 여기선 NEAREST가 물리적 왜곡 적음
    img_resized = img.resize(target_size, resample=Image.NEAREST)
    
    resized = np.array(img_resized, dtype=np.float32)
    
    # 4. Normalize (0~1)
    max_val = np.max(resized)
    if max_val > 0:
        resized = resized / max_val
        
    # (H, W) -> (H, W, 1) 차원 추가
    return resized[..., np.newaxis]