import os
import json
import glob
from collections import Counter

# =========================================================
# 1. 설정 (경로 확인!)
# =========================================================
# 창민님의 터미널 경로 기반으로 설정했습니다.
# 만약 statistics 파일이 raw가 아니라 labelled 폴더에 있다면 아래 경로를 'labelled'로 바꿔주세요.
BASE_DATA_DIR = "/home/changmin/pulse_alternative_ai_payload/data/vzlusat1-timepix-data/data/labelled"

# 분석할 대상 폴더 목록
TARGET_DIRS = ["above_europe", "planetary_sweeps", "saa_and_poles"]

# =========================================================
# 2. 라벨 결정 로직 (우리가 정한 Priority Rule)
# =========================================================
def get_label_from_stats(stats):
    # 1순위: Track (전자)
    n_tracks = (stats.get('track_straight', 0) + 
                stats.get('track_curly', 0) + 
                stats.get('track_lowres', 0))
    if n_tracks > 0:
        return 1 # Track
        
    # 2순위: Blob (양성자/이온)
    n_blobs = (stats.get('blob_big', 0) + 
               stats.get('blob_small', 0) + 
               stats.get('blob_branched', 0))
    if n_blobs > 0:
        return 2 # Blob

    # 3순위: Noise
    return 0 # Noise

# =========================================================
# 3. 메인 분석 루프
# =========================================================
def main():
    print(f"🚀 Fast Distribution Check started at: {BASE_DATA_DIR}")
    print("-" * 60)
    print(f"{'Region':<20} | {'Total':<8} | {'Noise(0)':<10} | {'Track(1)':<10} | {'Blob(2)':<10}")
    print("-" * 60)

    grand_total_counts = Counter()

    for folder in TARGET_DIRS:
        search_path = os.path.join(BASE_DATA_DIR, folder, "**", "*_fullres.statistics.txt")
        
        # recursive=True를 쓰면 하위 폴더(01_2019...)까지 다 찾습니다.
        file_list = glob.glob(search_path, recursive=True)
        
        folder_counts = Counter()
        
        for file_path in file_list:
            try:
                with open(file_path, 'r') as f:
                    stats = json.load(f)
                    label = get_label_from_stats(stats)
                    folder_counts[label] += 1
            except Exception:
                continue # 파일 깨짐 등은 무시

        # 결과 집계
        grand_total_counts += folder_counts
        total = sum(folder_counts.values())
        
        # 폴더별 출력
        print(f"{folder:<20} | {total:<8} | "
              f"{folder_counts[0]:<10} ({folder_counts[0]/total*100 if total else 0:.1f}%) | "
              f"{folder_counts[1]:<10} ({folder_counts[1]/total*100 if total else 0:.1f}%) | "
              f"{folder_counts[2]:<10} ({folder_counts[2]/total*100 if total else 0:.1f}%)")

    print("-" * 60)
    
    # 최종 합계 출력
    total_all = sum(grand_total_counts.values())
    print(f"{'GRAND TOTAL':<20} | {total_all:<8} | "
          f"{grand_total_counts[0]:<10} ({grand_total_counts[0]/total_all*100:.1f}%) | "
          f"{grand_total_counts[1]:<10} ({grand_total_counts[1]/total_all*100:.1f}%) | "
          f"{grand_total_counts[2]:<10} ({grand_total_counts[2]/total_all*100:.1f}%)")
    print("-" * 60)

    # 팁 출력
    print("\n💡 [Analysis Tip]")
    if grand_total_counts[0] / total_all > 0.8:
        print("⚠️ 데이터 불균형 심각: Noise(0)가 압도적으로 많습니다.")
        print("   -> 학습 시 'class_weights'를 적용하거나, Noise 데이터를 일부 버리는(Undersampling) 전략이 필수입니다.")
    else:
        print("✅ 데이터 균형이 비교적 양호합니다.")

if __name__ == "__main__":
    main()