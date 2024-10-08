import pandas as pd
import os
import json
from tqdm import tqdm

# 定义文件夹路径
input_folder = 'QA'
output_folder = 'QA_embed'
json_folder = 'input'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for file_name in tqdm(os.listdir(input_folder), desc="Processing files"):
    if file_name.endswith('.csv'):
        input_file_path = os.path.join(input_folder, file_name)
        output_file_path = os.path.join(output_folder, file_name)

        # 对应的 JSON 文件路径
        json_file_name = os.path.splitext(file_name)[0] + '.json'
        json_file_path = os.path.join(json_folder, json_file_name)

        # 读取CSV文件
        df = pd.read_csv(input_file_path)

        # 读取 JSON 文件
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r', encoding='utf-8') as json_file:
                json_data = json.load(json_file)
        else:
            json_data = {}

        # 初始化新的DataFrame
        new_rows = []

        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Processing {file_name}"):
            row_data = row.to_dict()

            row_data['embed'] = f"{row['Text_pure']}\n\n({row['Position']})"
            for key, value in json_data.items():
                row_data[key] = value
            new_rows.append(row_data)
                
            if pd.notnull(row['Question']):
                row_data_copy = row.copy()
                row_data_copy['embed'] = f"{row['Question']}"
                for key, value in json_data.items():
                    row_data_copy[key] = value
                new_rows.append(row_data_copy.to_dict())

        new_df = pd.DataFrame(new_rows)

        # 保存新的CSV文件
        new_df.to_csv(output_file_path, index=False, encoding='utf-8')

        print(f"File {file_name} processed and saved to {output_file_path}.")
