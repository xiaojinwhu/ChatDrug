import json
from pathlib import Path

data_dir = "./data/kg_pdf_data"

import json
from pathlib import Path


# 定义一个函数来读取 JSON 文件并返回其内容
def read_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


# 递归遍历 data 目录，读取所有 JSON 文件
data_folder = Path(data_dir)
json_files = list(data_folder.glob("**/*.json"))

# # 将所有 JSON 记录拼接成一个字符串
# combined_data = ""
for json_file in json_files:
    json_data = read_json_file(json_file)
    file_name = json_file.stem
    record = "\n".join(
        [f"{key}：{str(value).strip()}" for key, value in json_data.items()]
    )
    # combined_data += record + "\n"

    with open(f"data/txt/{file_name}.txt", "w", encoding="utf-8") as output_file:
        output_file.write(record)

print("所有 JSON 记录已成功拼接到 combined_data.txt 文件中。")
