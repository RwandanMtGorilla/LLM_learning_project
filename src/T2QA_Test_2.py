import pandas as pd
import re
import os
import ollama.client as client
from tqdm import tqdm

pattern = re.compile(r"-?\s*question: (.*?)\s*-?\s*answer: (.+)", re.DOTALL)

# 定义文件夹路径
input_folder = 'output'
output_folder = 'QA'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

rounds = 3  # 轮次

for file_name in tqdm(os.listdir(input_folder), desc="Processing files"):
    if file_name.endswith('.csv'):
        input_file_path = os.path.join(input_folder, file_name)
        output_file_path = os.path.join(output_folder, file_name)

        if os.path.exists(output_file_path):
            new_df = pd.read_csv(output_file_path)
        else:
            new_df = pd.DataFrame(columns=['Question', 'Answer'])

        df = pd.read_csv(input_file_path)

        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Processing {file_name}"):
            if index < len(new_df) and pd.notnull(new_df.loc[index, 'Question']) and pd.notnull(new_df.loc[index, 'Answer']):
                continue  # 跳过非空行

            text = row['Text_pure']
            if not isinstance(text, str):
                print(f"Skipping row {index} in file {file_name}: Text is not a string")
                continue

            best_question = ''
            best_answer = ''
            best_length = 0

            for _ in range(rounds): # todo: 如果你认为这些内容是没有营养的 输出0(counting stars) 分三段 1. reference 2. question 3. answer (相当于是优化+QA 合并)
                sys_prompt = "你是一个QA对生成大师"
                prompt = (
                    "对于以下文本内容，请生成一个相关的问题，然后用文本中的信息提供答案。"
                    f"当前文本：{text}\n"
                    "请根据以上信息优化当前文本，并以以下格式返回结果：\n"
                    "- question: 请提供和上述文本相关的问题，问题清晰，完整。\n"
                    "- answer:  请根据文本中的信息提供准确且完整的答案\n"
                )

                response, _ = client.generate(
                    model_name="qwen2:7b-instruct-fp16",
                    system=sys_prompt, 
                    prompt=prompt
                )

                match = pattern.search(response)
                if match:
                    print("\nSplited!")
                    question = match.group(1).strip()
                    answer = match.group(2).strip()
                    length = len(question) + len(answer)
                    if length > best_length:
                        best_question = question
                        best_answer = answer
                        best_length = length

            # 更新DataFrame
            row_data = row.to_dict()
            row_data['Question'] = best_question
            row_data['Answer'] = best_answer

            if index < len(new_df):
                new_df.iloc[index] = row_data
            else:
                new_df = pd.concat([new_df, pd.DataFrame([row_data])], ignore_index=True)

            new_df.to_csv(output_file_path, index=False)  # 逐行保存

        print(f"File {file_name} processed and saved to {output_file_path}.")
