import json
from pathlib import Path

SEED = 20
folder_prnt_path = Path(f"./weak-to-strong/results/sample_difficulty/seed={SEED}")
folder_li = [path.name for path in folder_prnt_path.iterdir() if path.is_dir()]

weak_model_sizes = ["gpt2-large", "gpt2-xl", "QwenQwen-1_8B", "QwenQwen-7B"]

total_dict = {}

# 중첩 딕셔너리의 값을 안전하게 더해주는 함수
def accumulate_dict(target, source):
    for k, v in source.items():
        if isinstance(v, dict):
            if k not in target:
                target[k] = {}
            accumulate_dict(target[k], v)
        else:
            target[k] = target.get(k, 0) + v

# 최종 합산된 값을 파일 개수로 나누어 평균을 내는 함수
def divide_dict(target, count):
    result = {}
    for k, v in target.items():
        if isinstance(v, dict):
            result[k] = divide_dict(v, count)
        else:
            result[k] = v / count
    return result

for wms in weak_model_sizes:
    obj_folder_li = []
    
    # 1. wms별로 폴더 분류
    for folder_name in folder_li:
        if f"wms:{wms}" in folder_name:
            obj_folder_li.append(folder_name)

    if not obj_folder_li:
        continue

    sum_dict = {}

    # 2. 파일 데이터 누적 합산
    for obj_fol in obj_folder_li:
        file_dir = obj_fol + "/" + "matchedness_accuracy.json"

        with open(folder_prnt_path / file_dir, "r") as f:
            mtcd_dict = json.load(f)
            accumulate_dict(sum_dict, mtcd_dict)

    # 3. 파일 총 개수로 나누어 평균 계산
    total_count = len(obj_folder_li)
    total_dict[wms] = divide_dict(sum_dict, total_count)

# 4. 결과 저장
result_dir = folder_prnt_path / "result_4_wms.json"
with open(result_dir, "w") as f:
    json.dump(total_dict, f, indent=2)