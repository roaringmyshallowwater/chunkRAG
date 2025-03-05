from storage import load_vectorstore
from langchain.prompts import ChatPromptTemplate

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

import streamlit as sl
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import pandas as pd
import time

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
WEAVIATE_URL = os.getenv('WEAVIATE_URL')


PROMPT_TEMPLATE = dict(
    RAG_PROMPT_TEMPLATE="""
You are a useful assistant for provide professional customer question answering. 
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Must reply using the language used in the user’s question.
Context: {context}
Question: {question}
Answer:""",
    RAG_PROMPT_TEMPLATE_2="""使用以上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
        问题: {question}
        可参考的上下文：
        ···
        {context}
        ···
        如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。"""
)

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# 治标不治本
# 删除文件..\\Anaconda3\\envs\\work\\Library\\bin\\libiomp5md.dll13。

def load_prompt():
        prompt_template = PROMPT_TEMPLATE['RAG_PROMPT_TEMPLATE']
        prompt = ChatPromptTemplate.from_template(prompt_template)
        return prompt
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# to load the OPENAI LLM
def load_llm():
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model_name='gpt-4-0125-preview', temperature=0, api_key=OPENAI_API_KEY)
    return llm

def generate_response(query):
    knowledgebase = load_vectorstore(classname)
    llm = load_llm()
    prompt = load_prompt()

    if (query):
        # getting only the chunks that are similar to the query for llm to produce the output
        similar_embeddings = knowledgebase.similarity_search(query)
        similar_embeddings = FAISS.from_documents(documents=similar_embeddings,
                                                  embedding=OpenAIEmbeddings(api_key=openai_api_key))

        # creating the chain for integrating llm,prompt,stroutputparser
        retriever = similar_embeddings.as_retriever()
        # retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": .5})

        rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
        )

        response = rag_chain.invoke(query)
        return response


if __name__ == '__main__':
    sl.header("welcome to the law RAG bot")

    df = pd.DataFrame({
        'knowledge base': ["example", "law"]
    })

    # query = sl.text_input('Enter some text')
    # classname = sl.text_input('Enter knowledge base class name', value="example")

    classname = sl.sidebar.selectbox(
        'Which knowledge bas you want to select',
        df['knowledge base']
    ) # sidebar代表侧边栏

    openai_api_key = sl.sidebar.text_input('OpenAI API Key', type='password', value=OPENAI_API_KEY)



    with sl.form('my_form'):
        query = sl.text_area('Enter text:', '请解释什么是合同？')
        submitted = sl.form_submit_button('Submit')
        if not openai_api_key.startswith('sk-'):
            sl.warning('Please enter your OpenAI API key!', icon='⚠')
        if submitted and openai_api_key.startswith('sk-'):
            response = generate_response(query)
            sl.write(response)