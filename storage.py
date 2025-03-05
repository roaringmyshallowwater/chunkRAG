import os
from pathlib import Path
import torch

from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from RAG.utils import ReadFiles, JSONDocumentLoader
from RAG.LLM import OpenAIChat, Deepseek
from RAG.Embeddings import OpenAIEmbedding, ZhipuEmbedding

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

PROMPT_TEMPLATE = dict(
    RAG_PROMPT_TEMPLATE="""
You are a useful assistant for provide professional customer question answering. 
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Must reply using the language used in the user's question.
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

def get_embeddings(embedding_type: str = "zhipu"):
    """
    获取指定的embedding模型
    Args:
        embedding_type: embedding类型，可选：
            - "zhipu": 智谱AI API（推荐，支持中文）
            - "openai": OpenAI API
    Returns:
        embedding模型实例
    """
    if embedding_type == "zhipu":
        return ZhipuEmbedding(is_api=True)
    elif embedding_type == "openai":
        return OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    else:
        raise ValueError(f"Unsupported embedding type: {embedding_type}")

def load_vectorstore(class_name: str, file_path: str, embeddings) -> FAISS:
    """
    加载向量库，如果不存在则创建
    """
    # 确保faiss目录存在
    os.makedirs("./faiss", exist_ok=True)
    
    # 检查是否已存在向量库
    if os.path.exists(f"./faiss/{class_name}.index"):
        print(f"Loading existing vector store for class: {class_name}")
        return FAISS.load_local(f"./faiss/{class_name}", embeddings)
    
    print(f"Creating new vector store for class: {class_name}")
    
    # 根据文件类型选择加载器
    if file_path.endswith('.json'):
        # 使用JSONDocumentLoader处理JSON文件
        loader = JSONDocumentLoader(file_path)
        docs = loader.load()
    else:
        # 使用ReadFiles处理其他类型文件
        reader = ReadFiles(file_path)
        docs = reader.get_content()
    
    # 创建向量库
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    # 保存向量库
    vectorstore.save_local(f"./faiss/{class_name}")
    print(f"Vector store saved to ./faiss/{class_name}")
    
    return vectorstore


def get_query(classname, query, template=PROMPT_TEMPLATE['RAG_PROMPT_TEMPLATE'], 
             embedding_type="zhipu", model_type="openai"):
    """
    使用向量库进行问答
    Args:
        classname: 向量库名称
        query: 查询问题
        template: 提示词模板
        embedding_type: embedding类型，可选 "zhipu", "bge", "text2vec"
        model_type: 使用的模型类型，可选 "openai" 或 "deepseek"
    """
    # 获取embedding模型
    embeddings = get_embeddings(embedding_type)
    
    # 读取已有向量库进行问答
    vectorstore = load_vectorstore(classname, embeddings=embeddings)
    
    # ---------- 1. 检索 ---------- #
    retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": .5})
    
    # ---------- 2. 增强 ---------- #
    prompt = ChatPromptTemplate.from_template(template)
    
    # ---------- 3. 生成 ---------- #
    if model_type == "openai":
        llm = ChatOpenAI(model_name='gpt-4-0125-preview', api_key=OPENAI_API_KEY)
        rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
        )
        ans = rag_chain.invoke(query)
    elif model_type == "deepseek":
        # 获取相关文档
        docs = retriever.get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in docs])
        
        # 使用Deepseek模型
        deepseek = Deepseek()
        ans = deepseek.chat(query, [], context)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    #  ---------- 4. 返回结果 ---------- #
    print(f"Searching for relevant documents...")
    docs = retriever.get_relevant_documents(query)
    text_ls = [doc.page_content[:] for doc in docs]
    return ans, text_ls
