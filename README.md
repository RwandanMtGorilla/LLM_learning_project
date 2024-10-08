## LLM-learning-project

## 介绍
本项目包含完整的从处理文档到启动一个streamlit服务的代码流程

# 栈
- 使用 Ollama管理 Qwen大模型本地运行
- 采用BGE-m3向量化模型
- 采用ChromaDB作为向量化存储库

## 环境准备
- 详见 `workbench.md`

## 代码作用
- `md2csv_r.py` 将markdown文档转化为csv文件便于后续处理
- `T2QA_Test_2.py` 生成问答对
- `QA2data.py` 数据规范化方便后续处理
- `csvDBtest.py` 存入Chroma
- `Script_S2.py` 即将运行的src代码
- `app_3.py` streamlit前端代码

## 运行
```sh
conda activate AI

python md2csv_r.py
python T2QA_Test_2.py
python QA2data.py
python csvDBtest.py

python app_3.py
```

