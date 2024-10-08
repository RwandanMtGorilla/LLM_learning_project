import os
# from config import OPENAI_API_KEY, OPENAI_API_BASE
import re
import json

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import SpacyTextSplitter
from tqdm import tqdm
import time
import requests


import logging
import csv

print("start loading")
embedding_function = SentenceTransformerEmbeddings(
    # model_name='/home/AI/petRAG/cache/models--BAAI--bge-m3/snapshots/5a212480c9a75bb651bcb894978ed409e4c47b82',
    # model_name='D:\\schoolRAG\\embedding\\models--moka-ai--m3e-base\\snapshots\\764b537a0e50e5c7d64db883f2d2e051cbe3c64c',
    model_name='C:\\Users\\WorkPC\\.cache\\huggingface\\hub\\models--BAAI--bge-m3\\snapshots\\5a212480c9a75bb651bcb894978ed409e4c47b82',

    # model_name='moka-ai/m3e-base',
    model_kwargs={'device': 'cpu'},
    # cache_folder='D:\\schoolRAG\\embedding'
    )
print("loading end")

identity = "专家"
job = "接收用户消息并对其做出简明扼要的回答。你的回答不应该包含任何多余信息。"
BASE_URL = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')

def SearchAndResponse(query, reference):
    sys_prompt = (
        f"您是一位{identity}，工作是{job}\n"
    )
    prompt = f"您正在接受用户的提问\n以下是在数据库中找到的可能相关内容:\n```\n{reference}\n```\n以下是提问:<{query}>\n您的回答:"
    model_name = "qwen2:7b"
    system = sys_prompt
    keep_alive = "300m"

    try:
        start_time = time.time()  # 开始计时
        url = f"{BASE_URL}/api/generate"
        payload = {
            "model": model_name, 
            "prompt": prompt, 
            "system": system, 
            "keep_alive": keep_alive
        }

        payload = {k: v for k, v in payload.items() if v is not None}

        with requests.post(url, json=payload, stream=True) as response:
            response.raise_for_status()

            collected_sentence = ''
            all_passed_text = ''
            violation_reason = None

            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)

                    if 'response' in chunk:
                        text = chunk['response']
                        collected_sentence += text

                        while any(punct in collected_sentence for punct in ['。', '，', '；', '！', '？']):
                            # 找到第一个出现的标点符号
                            punct_positions = {punct: collected_sentence.find(punct) for punct in ['。', '，', '；', '！', '？'] if collected_sentence.find(punct) != -1}
                            first_punct_position = min(punct_positions.values())
                            first_punct = [punct for punct, pos in punct_positions.items() if pos == first_punct_position][0]

                            # 以第一个出现的标点符号分割文本
                            sentence, collected_sentence = collected_sentence.split(first_punct, 1)
                            sentence += first_punct

                            is_ok, reason = check_sensitive_content(sentence)
                            if not is_ok:
                                violation_reason = reason
                                print(violation_reason)
                                # violation_reason="抱歉，我们的回答可能包含不合适的内容，已经终止回答。"
                                yield None, violation_reason

                            all_passed_text += sentence
                            end_time = time.time()  # 结束计时
                            print(f"gen Time: {end_time - start_time} seconds")  # 打印使用时间
                            yield sentence, violation_reason
                            # print(all_passed_text, end='', flush=True)

            if collected_sentence:
                is_ok, reason = check_sensitive_content(collected_sentence)
                if not is_ok:
                    print(reason)
                    # reason="抱歉，我们的回答可能包含不合适的内容，已经终止回答。"

                    yield collected_sentence, reason
                else:
                    all_passed_text += collected_sentence
                    yield collected_sentence, violation_reason
                    # print(all_passed_text, end='', flush=True)

            # yield all_passed_text, violation_reason
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        yield None, None




def search_relevant_texts(user_input, k=6, score_threshold=0.65):
    persist_directory = "BGE_DB4"
    # persist_directory = "BGE_DB_SCHOOL"
    # embedding_function = SentenceTransformerEmbeddings(model_name='moka-ai/m3e-base')
    # embedding_function = SentenceTransformerEmbeddings(# C:\\Users\\WorkPC\\.cache\\huggingface\\hub\\
    #     model_name='C:\\Users\\WorkPC\\.cache\\huggingface\\hub\\models--BAAI--bge-m3\\snapshots\\5a212480c9a75bb651bcb894978ed409e4c47b82',
    #     model_kwargs={'device': 'cpu'},
    # )
    vectordb = Chroma(embedding_function=embedding_function, persist_directory=persist_directory)

    results = vectordb.similarity_search_with_relevance_scores(query=user_input, k=k, score_threshold=score_threshold)
    
    # print(results)

    # 检查results是否为空
    if not results:
        page_contents = ["没有找到相关信息!"]
        details = []
    else:
        # 使用集合来跟踪已见过的文本
        seen_texts = set()
        filtered_results = []
        
        # 过滤重复的Document
        for doc, score in results:
            text = doc.metadata['Text']
            if text not in seen_texts:
                seen_texts.add(text)
                filtered_results.append((doc, score))
        
        # 创建第一个列表，包含所有Document的page_content
        page_contents = [f"{doc.metadata['Text']}\n\n({doc.metadata['Position']})" for doc, _ in filtered_results]
        # 创建第二个列表，包含\ufeffOriginal_Text、name、url和相似性得分
        details = [
            {
                'Original_Text': doc.metadata['Text'],
                'name': doc.metadata.get('name', None),
                'url': doc.metadata.get('url', None),
                'position': doc.metadata['Position'],
                'Question': doc.metadata.get('Question', None),  # 如果Question不存在，则提供默认值
                'Img_url': doc.metadata.get('Img_url', None),  # 如果Img_url不存在，则提供默认值
                'score': score,
                'id': doc.metadata['row']
            }
            for doc, score in filtered_results
        ]

    i = 0
    texts_length = 0
    for doc, score in results:
        i += 1
        texts_length += len(doc.page_content)

    return page_contents, texts_length, details, results


def check_sensitive_content(text):
    
    return True, None


    
# def main(question):
#     start_time = time.time()
#     print("Main function is called.")
#     is_valid, reason = check_sensitive_content(question)
#     if not is_valid:
#         parsed_data = json.loads(reason)
#         risk_tips = parsed_data['riskTips']
#         yield f"抱歉，您的回答可能包含以下不合适的内容: {risk_tips}"
#         return

#     print("Query is being processed.")
#     page_contents, texts_length, details, results = search_relevant_texts(question)
#     end_time = time.time()  # 结束计时
#     print(f"respond Time: {end_time - start_time} seconds")  # 打印使用时间

#     for partial_answer, reason in SearchAndResponse(question, page_contents):
#         if reason:
#             parsed_data = json.loads(reason)
#             risk_tips = parsed_data['riskTips']
#             yield f"抱歉，我们的回答可能包含不合适的内容,回答中止。"
#             break
#         else:
#             yield partial_answer



class CsvLogHandler(logging.Handler):
    def __init__(self, filename, mode='a', encoding='utf-8'):
        super().__init__()
        self.filename = filename
        self.mode = mode
        self.encoding = encoding

    def emit(self, record):
        # Formatting the time
        log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(record.created))
        message = record.getMessage()  # Get the fully formatted log message
        with open(self.filename, self.mode, newline='', encoding=self.encoding) as f:
            writer = csv.writer(f)
            writer.writerow([log_time, record.levelname, message])



# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('user_interaction_logger')
logger.addHandler(CsvLogHandler('user_interactions.csv'))

def main(question):
    start_time = time.time()
    logger.info(f"Received question: {question}")
    print("Main function is called.")
    is_valid, reason = check_sensitive_content(question)
    if not is_valid:
        parsed_data = json.loads(reason)
        risk_tips = parsed_data['riskTips']
        yield None, f"抱歉，您的问题可能包含以下不合适的内容: {risk_tips}"
        return 0

    print("Query is being processed.")
    page_contents, texts_length, details, results = search_relevant_texts(question)
    logger.info(f"Details provided: {details}")
    yield details, None

    end_time = time.time()  # 结束计时
    print(f"respond Time: {end_time - start_time} seconds")  # 打印使用时间

    for partial_answer, reason in SearchAndResponse(question, page_contents):
        if reason:
            parsed_data = json.loads(reason)
            risk_tips = parsed_data['riskTips']
            logger.info(response_message)
            yield None, f"抱歉，我们的回答可能包含不合适的内容,回答中止。"
            break
        else:
            logger.info(f"Partial answer provided: {partial_answer}")
            yield None, partial_answer
