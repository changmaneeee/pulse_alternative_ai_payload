import os
import random

# ==========================================
# [설정] 경로 확인
# ==========================================
LABEL_DIR = '/home/changmin/pulse_alternative_ai_payload/data/vzlusat1-timepix-data/data/labelled'
RAW_DIR = '/home/changmin/pulse_alternative_ai_payload/data/vzlusat1-timepix-data/data/raw'

def verify_file_matching():
    print(">>> 🕵️‍♂️ Raw 데이터와 Label 데이터 짝꿍(Matching) 점검 시작...")
    
    # 1. Raw 파일 목록 구축 (Dictionary for O(1) lookup)
    # Key: 파일명 (예: "2023_01_01.txt"), Value: 전체 경로
    raw_file_map = {}
    total_raw_files = 0
    ignored_metadata = 0

    print(f"    📂 Raw 폴더 스캔 중: {RAW_DIR}")
    for root, dirs, files in os.walk(RAW_DIR):
        for f in files:
            # .txt 파일만 수집
            if f.endswith(".txt") and "fullres" in f:
                # [중요] 메타데이터 파일은 짝꿍 후보에서 제외해야 함
                if "metadata" in f:
                    ignored_metadata += 1
                    continue
                
                raw_file_map[f] = os.path.join(root, f)
                total_raw_files += 1
    
    print(f"       -> 총 Raw 파일: {total_raw_files}개 (메타데이터 {ignored_metadata}개 제외됨)")

    # 2. Label 파일 목록 구축
    label_files = []
    print(f"    📂 Label 폴더 스캔 중: {LABEL_DIR}")
    for root, dirs, files in os.walk(LABEL_DIR):
        for f in files:
            if f.endswith("fullres.clusters.txt"):
                label_files.append((root, f))
    
    print(f"       -> 총 Label 파일: {len(label_files)}개")

    # 3. 매칭 테스트
    print("\n>>> 🔗 매칭 테스트 진행 중...")
    
    match_success = 0
    match_fail = 0
    examples = [] # 성공 사례 (Label명, Raw명)
    failures = [] # 실패 사례 (Label명)

    # 진행 상황 확인을 위해 섞지 않고 순서대로 하되, 결과 출력용은 따로 뽑음
    check_list = label_files.copy()
    
    for root, label_file in check_list:
        # [매칭 로직 점검]
        # 현재 로직: "file.clusters.txt" -> "file.txt" 로 변환하여 찾기
        expected_raw_name = label_file.replace('.clusters.txt', '.txt')
        
        if expected_raw_name in raw_file_map:
            match_success += 1
            if len(examples) < 5: # 5개만 샘플로 저장
                raw_path = raw_file_map[expected_raw_name]
                examples.append((label_file, os.path.basename(raw_path)))
        else:
            match_fail += 1
            if len(failures) < 5:
                failures.append(label_file)

    # 4. 결과 리포트
    print("\n" + "="*40)
    print("📊 [최종 진단 리포트]")
    print(f"   ✅ 매칭 성공: {match_success}건")
    print(f"   ❌ 매칭 실패: {match_fail}건")
    print("="*40)

    print("\n🔎 [매칭 성공 예시 (제대로 연결된 것)]")
    for lbl, raw in examples:
        print(f"   Label: {lbl}")
        print(f"   Raw:   {raw}")
        print(f"   {'✅ 일치' if lbl.replace('.clusters.txt', '.txt') == raw else '⚠️ 이름 다름?!'}")
        print("-" * 20)

    if failures:
        print("\n🔥 [매칭 실패 예시 (Raw 파일을 못 찾음)]")
        print("   이 파일들은 이름 규칙이 다르거나 Raw 데이터가 유실된 것입니다.")
        for lbl in failures:
            expected = lbl.replace('.clusters.txt', '.txt')
            print(f"   Label: {lbl}")
            print(f"   Expected Raw: {expected} (이 파일이 Raw 폴더에 없음)")
            print("-" * 20)
    else:
        print("\n🎉 모든 라벨 파일이 정상적으로 원본 파일과 연결되었습니다!")

if __name__ == '__main__':
    verify_file_matching()