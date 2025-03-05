# -*- coding: utf-8 -*-
import time
from time import sleep
import gradio as gr
from storage import load_vectorstore, get_query


class_ls = ['law']
# 1. 通过读取faiss的所有.pkl or .faiss文件读取向量库列表
# 2. 用户可以自行上传文件，默认保存在filepath = './user/xxx'下，用户需要输入classname
# 3. 用户可以删除知识库

def respond(classname, question):
    """
    根据question进行检索增强生成回答
    TODO:
    1. openai api 流式输出
    2. 多轮问答
    3. 展示参考知识，跳转溯源知识板块
    """
    ans, text_ls = get_query(classname, question)
    bot_message = ans
    for i in range(len(text_ls)):
        bot_message += text_ls[i] + "\n"
    return "", [[question, bot_message]]
    # 还需要根据检索到的知识进行展出


# TODO: 增加上传文件的功能
def run_app():
    with gr.Blocks(title='法律知识库问答') as demo:
        with gr.Row():
            with gr.Column():
                classname = gr.Dropdown(choices=class_ls, label="知识库选择")
                question = gr.Textbox(placeholder="请输入你的问题……（enter 发送）", label="提问")
            with gr.Column(scale=2):
                chatbot = gr.Chatbot()
                question.submit(fn=respond, inputs=[classname, question], outputs=[question, chatbot])
                clear = gr.ClearButton(chatbot)
    demo.launch()


if __name__ == "__main__":
    run_app()