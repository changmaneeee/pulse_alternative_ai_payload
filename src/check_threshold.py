import os
import json
import numpy as np
import matplotlib.pyplot as plt
import time

# ==========================================
# 설정 (경로 확인!)
# ==========================================
LABEL_DIR = '../data/vzlusat1-timepix-data/data/labelled'
RAW_DIR = '../data/vzlusat1-timepix-data/data/raw'

# 분석 범위 (0 ~ 100 에너지 구간을 0.1 단위로 쪼개서 집계)
BIN_EDGES = np.arange(0, 100.1, 0.1) 
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2

def run_full_analysis():
    print(">>> 🚀 전체 데이터 전수 조사(Full Census)를 시작합니다...")
    print(">>> 메모리 최적화를 위해 히스토그램 집계 방식을 사용합니다.")
    
    start_time = time.time()
    
    # 1. Raw 파일 인덱싱 (속도 향상)
    print(">>> Raw 파일 인덱싱 중...")
    raw_files_map = {}
    for root, dirs, files in os.walk(RAW_DIR):
        for f in files:
            if f.endswith(".txt") and "fullres" in f:
                raw_files_map[f] = os.path.join(root, f)
    print(f"   > Raw 파일 {len(raw_files_map)}개 발견.")

    # 2. 데이터 저장소 초기화 (누적 히스토그램)
    # 배경 노이즈용
    hist_noise = np.zeros(len(BIN_EDGES)-1, dtype=np.int64)
    # 각 클래스(Signal)용
    hist_signals = {} 
    
    # 통계용 변수 (Min/Max 추적)
    stats = {
        'noise': {'min': 9999, 'max': -9999, 'count': 0},
        'signal': {} 
    }

    # 3. Label 파일 순회
    processed_count = 0
    
    for root, dirs, files in os.walk(LABEL_DIR):
        for f in files:
            if f.endswith("fullres.clusters.txt"):
                json_path = os.path.join(root, f)
                
                # 짝꿍 Raw 찾기
                base_name = f.replace('.clusters.txt', '.txt')
                if base_name not in raw_files_map: continue
                
                try:
                    # 파일 로드
                    raw_path = raw_files_map[base_name]
                    raw_img = np.loadtxt(raw_path)
                    
                    with open(json_path, 'r') as jf:
                        clusters = json.load(jf)
                    
                    # 마스크 생성 (Signal 위치 표시)
                    mask = np.zeros_like(raw_img, dtype=bool)
                    
                    # --- Signal 처리 ---
                    for cluster in clusters:
                        cls_data = cluster.get('cluster_class')
                        cls_name = cls_data.get('name') if isinstance(cls_data, dict) else cls_data
                        
                        pixels = cluster.get('pixels')
                        if not pixels: continue
                        
                        xs = [p['x'] for p in pixels]
                        ys = [p['y'] for p in pixels]
                        vals = [p['value'] for p in pixels]
                        
                        # 마킹
                        for x, y in zip(xs, ys):
                            mask[y, x] = True
                            
                        # 히스토그램 누적 (Signal)
                        if cls_name not in hist_signals:
                            hist_signals[cls_name] = np.zeros(len(BIN_EDGES)-1, dtype=np.int64)
                            stats['signal'][cls_name] = {'min': 9999, 'max': -9999, 'count': 0}
                            
                        counts, _ = np.histogram(vals, bins=BIN_EDGES)
                        hist_signals[cls_name] += counts
                        
                        # Min/Max 업데이트
                        curr_min = min(vals)
                        curr_max = max(vals)
                        if curr_min < stats['signal'][cls_name]['min']: stats['signal'][cls_name]['min'] = curr_min
                        if curr_max > stats['signal'][cls_name]['max']: stats['signal'][cls_name]['max'] = curr_max
                        stats['signal'][cls_name]['count'] += len(vals)

                    # --- Noise 처리 ---
                    # 마스크가 False인 곳(배경) 추출
                    noise_vals = raw_img[~mask]
                    
                    if len(noise_vals) > 0:
                        # 노이즈는 너무 많으니, 파일당 일부만 샘플링하지 않고
                        # 전수조사니까 다 넣되, np.histogram으로 바로 카운트만 더함 (메모리 안전)
                        counts, _ = np.histogram(noise_vals, bins=BIN_EDGES)
                        hist_noise += counts
                        
                        # 통계 업데이트
                        curr_min = np.min(noise_vals)
                        curr_max = np.max(noise_vals)
                        if curr_min < stats['noise']['min']: stats['noise']['min'] = curr_min
                        if curr_max > stats['noise']['max']: stats['noise']['max'] = curr_max
                        stats['noise']['count'] += len(noise_vals)

                    processed_count += 1
                    if processed_count % 1000 == 0:
                        elapsed = time.time() - start_time
                        print(f"   Processed {processed_count} files... ({elapsed:.1f}s)")
                        
                except Exception as e:
                    continue

    # 4. 결과 출력 및 시각화
    print("\n" + "="*60)
    print("📊 [전수 조사 완료: Signal vs Noise]")
    print("="*60)
    
    # Noise 통계 계산 (Histogram 기반 백분위수 추정)
    total_noise = stats['noise']['count']
    cumsum_noise = np.cumsum(hist_noise)
    p99_idx = np.searchsorted(cumsum_noise, total_noise * 0.99)
    noise_p99 = BIN_CENTERS[p99_idx]
    
    print(f"🔊 Noise (Background)")
    print(f"   - Total Pixels: {total_noise}")
    print(f"   - Range: {stats['noise']['min']:.2f} ~ {stats['noise']['max']:.2f}")
    print(f"   - 99% Percentile: {noise_p99:.2f} (노이즈의 99%는 이 값 아래)")
    print("-" * 40)
    
    # 그래프 그리기
    plt.figure(figsize=(14, 7))
    
    # Noise (Log Scale Y축 권장 - 노이즈 픽셀 수가 압도적으로 많음)
    plt.plot(BIN_CENTERS, hist_noise, label='Noise (Background)', color='gray', alpha=0.5)
    plt.fill_between(BIN_CENTERS, hist_noise, color='gray', alpha=0.1)

    # 주요 Signal
    target_classes = ['track_straight', 'track_lowres', 'dot', 'blob_small']
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, cls in enumerate(target_classes):
        if cls in hist_signals:
            plt.plot(BIN_CENTERS, hist_signals[cls], label=f'{cls}', color=colors[i])
            
            s_min = stats['signal'][cls]['min']
            print(f"✨ Signal: {cls}")
            print(f"   - Min Value: {s_min:.2f}")
            print(f"   - Count: {stats['signal'][cls]['count']}")
    
    # Threshold Line
    plt.axvline(x=3, color='purple', linestyle='--', linewidth=2, label='Proposed TH=3')
    plt.axvline(x=noise_p99, color='black', linestyle=':', label=f'Noise 99% ({noise_p99:.1f})')

    plt.yscale('log') # 로그 스케일 (중요: 노이즈와 시그널 개수 차이가 큼)
    plt.xlim(0, 30)   # 0~30 구간 확대
    plt.title('Pixel Value Histogram (Full Census) - Log Scale')
    plt.xlabel('Pixel Value (Energy)')
    plt.ylabel('Pixel Count (Log)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('full_census_analysis.png')
    print(f"\n✅ 그래프 저장 완료: full_census_analysis.png")
    print(f"⏳ 총 소요 시간: {(time.time() - start_time)/60:.1f}분")

if __name__ == '__main__':
    run_full_analysis()