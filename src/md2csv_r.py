import os
import pandas as pd
import re
from tqdm import tqdm
from collections import defaultdict
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 定义输入和输出文件夹
input_folder = "./input/"
output_folder = "./output/"

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 定义文本切分器
text_splitter = RecursiveCharacterTextSplitter(
    separators=[
        "\n\n",
        "\n",
        " ",
        ".",
        ",",
        "\u200b",  # Zero-width space
        "\uff0c",  # Fullwidth comma
        "\u3001",  # Ideographic comma
        "\uff0e",  # Fullwidth full stop
        "\u3002",  # Ideographic full stop
        "",
    ],
    chunk_size=250,
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=False,
)

# 获取输入文件夹中的所有.md和.txt文件
files = [f for f in os.listdir(input_folder) if f.endswith(".md") or f.endswith(".txt")]

# 字典用于存储替换信息
replacement_dict = defaultdict(list)

def replace_img(match, img_counter, replacement_dict, img_urls):
    key = f"<img{img_counter}>"
    replacement_dict[key] = match.group(0)
    img_urls.append(match.group(1))  # 保存URL
    return key, img_counter + 1

def replace_text(match, text_counter, replacement_dict):
    text = match.group(1)
    text_counter[text] += 1
    key = f"<{text}{text_counter[text] if text_counter[text] > 1 else ''}>"
    replacement_dict[key] = match.group(0)
    return key

# 遍历文件并显示进度条
for filename in tqdm(files, desc="Processing files"):
    input_filepath = os.path.join(input_folder, filename)
    
    # 读取文件内容
    with open(input_filepath, encoding='utf8') as f:
        file_content = f.read()

    # 预处理步骤
    # 替换连续超过两个的*
    file_content = re.sub(r'\*{3,}', '**', file_content)
    # 替换连续超过三个的- 或 =
    file_content = re.sub(r'\-{4,}', '---', file_content)
    file_content = re.sub(r'\={4,}', '===', file_content)

    img_counter = 1
    text_counter = defaultdict(int)
    img_urls_per_line = []

    def replace_img(match, img_counter, replacement_dict):
        key = f"<img{img_counter}>"
        replacement_dict[key] = match.group(0)
        return key, img_counter + 1, match.group(1)

    def replace_text(match, text_counter, replacement_dict):
        text = match.group(1)
        text_counter[text] += 1
        key = f"<{text}{text_counter[text] if text_counter[text] > 1 else ''}>"
        replacement_dict[key] = match.group(0)
        return key

    # 替换 ![img](url)
    img_pattern = re.compile(r'!\[.*?\]\((.*?)\)')
    img_matches = list(img_pattern.finditer(file_content))
    for match in img_matches:
        replacement, img_counter, img_url = replace_img(match, img_counter, replacement_dict)
        file_content = file_content.replace(match.group(0), replacement)
        img_urls_per_line.append(img_url)

    # 替换 [text](url)
    text_pattern = re.compile(r'\[(.*?)\]\(.*?\)')
    text_matches = list(text_pattern.finditer(file_content))
    for match in text_matches:
        replacement = replace_text(match, text_counter, replacement_dict)
        file_content = file_content.replace(match.group(0), replacement)

    
    # 执行文本切分
    texts = text_splitter.create_documents([file_content])

    # 创建DataFrame
    df = pd.DataFrame([text.page_content for text in texts], columns=['Text'])

    # 还原替换的 <img%d> 和 <text%d>
    def restore_replacement(text):
        for key, value in replacement_dict.items():
            text = text.replace(key, value)
        return text

    # 创建 Text_pure 列
    df['Text_pure'] = df['Text']

    def extract_img_urls(text):
        img_keys = re.findall(r'<img\d+>', text)
        img_urls = []
        for key in img_keys:
            if key in replacement_dict:
                url_match = re.search(r'\((.*?)\)', replacement_dict[key])
                if url_match:
                    img_urls.append(url_match.group(1))
        return ';'.join(img_urls)

    df['Img_url'] = df['Text_pure'].apply(extract_img_urls)
    # 还原 Text 列中的内容
    df['Text'] = df['Text'].apply(restore_replacement)


    # 保存为CSV文件
    output_filename = f"{os.path.splitext(filename)[0]}.csv"
    output_filepath = os.path.join(output_folder, output_filename)
    df.to_csv(output_filepath, index=False, encoding='utf-8')

    print(f"Processed and saved: {output_filepath}")

    # 定义Markdown语法及其优先级
    markdown_priority = {
        r'^# ': 0,
        r'^## ': 1,
        r'^### ': 2,
        r'^#### ': 3,
        r'^##### ': 4,
        r'^###### ': 5,
        r'^\d+\. ': 6,
        r'^- ': 6,
        r'^\* ': 6,
        r'^\*\*.*\*\*$': 6
    }

    # 特殊Markdown语法检测
    special_syntax = {
        r'^\s*=+\s*$': 0,  # 一级标题
        r'^\s*-+\s*$': 1   # 二级标题
    }
    
    # 读取CSV文件内容
    df = pd.read_csv(output_filepath, encoding='utf-8')
    
    # 初始化栈并将文件名作为初始位置
    initial_position = os.path.splitext(filename)[0]
    position_stack = [initial_position]
    priority_stack = [-1]  # 初始优先级为-1，以确保文件名始终在最顶部

    # 定义位置列
    df['Position'] = ""
    
    # 用于记录连续相同位置的次数
    last_position = ""
    position_counter = 0

    # 逐行处理文本
    for idx, row in df.iterrows():
        text = row['Text']
        sentences = re.split(r'\n+', text)  # 以换行符进行分句
        
        # 首先对每一行赋初始位置
        current_position = ' > '.join(position_stack)
        
        # 检查是否与上一行位置相同
        if current_position == last_position:
            position_counter += 1
            if position_counter == 2:  # 在第二次遇到时给上一个添加序号
                df.at[idx - 1, 'Position'] += f" (part 1)"
            current_position += f" (part {position_counter})"
        else:
            last_position = current_position
            position_counter = 1
        
        df.at[idx, 'Position'] = current_position
        
        # 逐句处理文本
        previous_sentence = ""
        for sentence in sentences:
            current_priority = None
            
            # 判断当前句子的标题信息及其优先级
            for grammar, priority in markdown_priority.items():
                if re.match(grammar, sentence.strip()):
                    current_priority = priority
                    break
            
            # 特殊处理 `===` 和 `---` 的情况
            for special_grammar, special_priority in special_syntax.items():
                if re.match(special_grammar, sentence.strip()):
                    current_priority = special_priority
                    sentence = previous_sentence
                    break
            
            # 更新位置栈和优先级栈
            if current_priority is not None:
                while priority_stack and priority_stack[-1] >= current_priority:
                    position_stack.pop()
                    priority_stack.pop()
                
                position_stack.append(sentence.strip())
                priority_stack.append(current_priority)
            
            previous_sentence = sentence

    # 保存更新后的CSV文件
    df.to_csv(output_filepath, index=False, encoding='utf-8')

    print(f"Refined and saved: {output_filepath}")
