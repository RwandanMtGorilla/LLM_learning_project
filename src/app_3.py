import streamlit as st
import Script_S2 as sc  # 确保你的 Script_S2 脚本文件名正确
import time
import os

def main():
    st.title('问答bot')

    # 初始化会话状态
    if 'input_question' not in st.session_state:
        st.session_state.input_question = ""
        st.session_state.clear_answer = True  # 用于指示是否清除答案区域

    question = st.text_input('请输入的问题：', value=st.session_state.input_question, on_change=reset_question, key="question_input")

    if st.button('发送'):
        if question:
            st.session_state.input_question = question
            st.session_state.clear_answer = True  # 当发送新问题时设置清除答案的标记
            display_answer(question)

def reset_question():
    # 重置问题时不清除答案
    st.session_state.input_question = ""

def display_answer(question):
    if st.session_state.clear_answer:
        # 清除之前的答案
        st.session_state.answer_container = st.container()
        st.session_state.clear_answer = False
    else:
        # 如果不需要清除，则直接使用现有的容器
        st.session_state.answer_container.empty()  # 清空之前的内容

    with st.spinner('正在获取信息...'):
        generator = sc.main(question)
        details, _ = next(generator, (None, None))

        if not details:
            with st.session_state.answer_container:
                answer_placeholder = st.empty()
                complete_answer = ""
                for _, answer in generator:
                    for char in answer:
                        complete_answer += char
                        answer_placeholder.markdown(f"答案：\n{complete_answer}")
                        time.sleep(0.002)
            st.session_state.answer_container.write("没有找到相关信息。")
            display_questions(details if details else [])
            return

        # 汇总所有的图像URL
        all_img_urls = []
        for detail in details:
            if detail.get('Img_url'):
                img_urls = detail['Img_url'].split(';')
                img_urls = [os.path.normpath(img_url.strip()) for img_url in img_urls]
                all_img_urls.extend(img_urls)

        # 限制展示的图片数量
        max_img_to_display = 1
        img_display_count = 0

        # 显示答案
        with st.session_state.answer_container:
            answer_placeholder = st.empty()
            complete_answer = ""
            for _, answer in generator:
                for char in answer:
                    complete_answer += char
                    answer_placeholder.markdown(f"答案：{complete_answer}")
                    time.sleep(0.005)
            
            # 在答案末尾展示所有图像
            if all_img_urls:
                for img_url in all_img_urls:
                    if img_display_count >= max_img_to_display:
                        break
                    if os.path.exists(img_url):
                        st.image(img_url, use_column_width=True)
                        img_display_count += 1
                    else:
                        st.warning(f"图像文件未找到：{img_url}")

        # 展示问题和详情
        show_details_and_questions(details)

def show_details_and_questions(details):
    with st.expander("点击查看详细信息"):
        for detail in details:
            st.write(f"来源：[{detail['name']}]({detail['url']})")
            st.write(f"相似度得分：{detail['score']}")
            st.write(f"摘要：{detail['Original_Text']}")
            st.write(f"id：{detail['id']}")
            st.write("---")

    display_questions(details)

def display_questions(details):
    default_questions = ["培训认定系统有哪些业务?", "培训系统包含了哪些主要功能模块？", "如何将当前课程数据导出为excel文件？"]
    questions = [detail.get('Question') for detail in details if detail.get('Question') is not None]

    questions += default_questions[:max(0, 3 - len(questions))]

    for i, question in enumerate(questions, start=1):
        # 现在按钮点击只会更新问题到输入框
        st.button(f'问题{i}: {question}', key=f'question_{i}', on_click=set_question, args=(question,))

def set_question(question):
    st.session_state.input_question = question

if __name__ == '__main__':
    main()
