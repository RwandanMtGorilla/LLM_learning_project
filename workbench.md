```bash
tmux ls

tmux new -s workbench

tmux attach -t workbench

```
```bash
set http_proxy=http://127.0.0.1:7890 
set https_proxy=http://127.0.0.1:7890

set http_proxy=http://10.50.30.40:7890 # win
set https_proxy=http://10.50.30.40:7890

export HTTP_PROXY="http://10.50.30.40:7890" #linux
export HTTPS_PROXY="http://10.50.30.40:7890"

set http_proxy=http://10.61.176.1:7890 
set https_proxy=http://10.61.176.1:7890
set all_proxy=socks5://10.61.176.1:7890

```


```bash
git config --global http.postBuffer 524288000
git config --global http.lowSpeedLimit 0
git config --global http.lowSpeedTime 999999


```


```bash
conda env list

which conda
/home/dell/miniconda3/bin/conda create --name AI python=3.11
/home/dell/miniconda3/bin/conda create --name MinerU python=3.10
conda create --name AI python=3.11
conda create --name FinLLM python=3.10

/home/dell/miniconda3/bin/conda activate AI
/home/dell/miniconda3/bin/conda init

curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen:14b

pip install langchain_community langchain tqdm pandas streamlit -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install sentence-transformers -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install chromadb -i https://pypi.tuna.tsinghua.edu.cn/simple 
pip install streamlit-authenticator -i https://pypi.tuna.tsinghua.edu.cn/simple 
pip install passlib  -i https://pypi.tuna.tsinghua.edu.cn/simple 
pip install Flask Flask-CORS  -i https://pypi.tuna.tsinghua.edu.cn/simple 

pip install -U langchain-huggingface  -i https://pypi.tuna.tsinghua.edu.cn/simple 

pip install langchain_community langchain tqdm pandas streamlit sentence-transformers chromadb Flask Flask-CORS -i https://pypi.tuna.tsinghua.edu.cn/simple 
pip install langchain_community  -i https://pypi.tuna.tsinghua.edu.cn/simple 

export HF_ENDPOINT="https://hf-mirror.com"
set HF_ENDPOINT=https://hf-mirror.com
set HF_ENDPOINT=
pip install --upgrade pymilvus
pip install "pymilvus[model]"
pip install datasets peft -i https://pypi.tuna.tsinghua.edu.cn/simple 
set HF_HUB_CACHE=C:\path\to\cache
pip install pymilvus  -i https://pypi.tuna.tsinghua.edu.cn/simple 
milvus_lite
pip install -U langchain-huggingface  -i https://pypi.tuna.tsinghua.edu.cn/simple 

pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple 


openai
set PYTHONIOENCODING=utf-8
```
curl -X POST "http://127.0.0.1:8501/ask_question" -H "Content-Type: application/json" -d "{\"question\": \"我想查看9月20日 5点到6点的UPS2的输出电流\"}"

```
curl http://localhost:11434/api/generate -d '{"model": "qwen:32b-chat-v1.5-q8_0","prompt":"Why is the sky blue?"}'

```