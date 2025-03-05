#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
from typing import List

import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import MarkdownTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader, UnstructuredMarkdownLoader
from langchain_community.document_loaders import UnstructuredPDFLoader, UnstructuredWordDocumentLoader
from langchain_core.documents import Document
from transformers import AutoTokenizer

enc = tiktoken.get_encoding("cl100k_base")


class JSONDocumentLoader:
    """
    Load documents from JSON format and convert them to LangChain Document format
    """
    def __init__(self, json_path: str):
        self.json_path = json_path

    def load(self) -> List[Document]:
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        for item in data:
            # 假设JSON中的每个item包含content和metadata字段
            content = item.get('content', '')
            metadata = item.get('metadata', {})
            
            # 创建LangChain Document对象
            doc = Document(
                page_content=content,
                metadata=metadata
            )
            documents.append(doc)
        
        return documents


class ReadFiles:
    """
    class to read files
    """

    def __init__(self, path: str, splitter: str = 'RecursiveCharacterTextSplitter') -> None:
        self._path = path
        self.file_list = self.get_files()
        self.splitter = splitter

    def get_files(self):
        # args：dir_path，目标文件夹路径
        file_list = []
        for filepath, dirnames, filenames in os.walk(self._path):
            # os.walk 函数将递归遍历指定文件夹
            for filename in filenames:
                # print(filename)
                # 通过后缀名判断文件类型是否满足要求
                if filename.endswith(".md") or filename.endswith(".txt") or filename.endswith(".docx") or filename.endswith(".pdf") or filename.endswith(".doc") or filename.endswith(".json"):
                    # 如果满足要求，将其绝对路径加入到结果列表
                    file_list.append(os.path.join(filepath, filename))
        # # print(self._path)
        # # print(file_list)
        return file_list

    def get_content(self, chunk_size=512, chunk_overlap=64):
        docs = []
        # 读取文件内容
        for file in self.file_list:
            if file.endswith('.json'):
                # 处理JSON文件
                loader = JSONDocumentLoader(file)
                json_docs = loader.load()
                docs.extend(json_docs)
            else:
                # 处理其他类型文件
                content = self.read_file_content(file)
                chunk_content = self.get_chunk(
                    content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                docs.extend(chunk_content)
        return docs

    def get_chunk(self, doc: str, chunk_size: int, chunk_overlap: int) -> list[Document]:
        if self.splitter == 'RecursiveCharacterTextSplitter':
            return self.chunk_recursive_character(doc, chunk_size, chunk_overlap)
        elif self.splitter == 'MarkdownTextSplitter':
            return self.chunk_markdown(doc, chunk_size, chunk_overlap)
        else:
            raise ValueError(f"Unsupported splitter: {self.splitter}")

    def chunk_recursive_character(self, doc: str, chunk_size: int, chunk_overlap: int) -> list[Document]:
        model_path = os.environ.get("BAICHUAN_PATH", "baichuan-inc/Baichuan2-13B-Chat")
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        length_function = lambda text: len(tokenizer.tokenize(text))

        splitter = RecursiveCharacterTextSplitter(
            separators=["，", "。", '\\n\\n', '\\n'],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function
        )
        docs = splitter.create_documents([doc])
        # print(docs)
        return docs
    
    def chunk_markdown(self, markdown_text: str, chunk_size: int, chunk_overlap: int) -> list[Document]:
        markdown_splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = markdown_splitter.create_documents([markdown_text])
        # print(docs[0].page_content[:])
        # text_chunks = [doc.page_content[:] for doc in docs]
        return docs
    @classmethod
    def read_file_content(cls, file_path: str):
        # 根据文件扩展名选择读取方法
        if file_path.endswith('.pdf'):
            return cls.read_pdf(file_path)
        elif file_path.endswith('.md'):
            return cls.read_markdown(file_path)
        elif file_path.endswith('.txt'):
            return cls.read_text(file_path)
        elif file_path.endswith('doc'):
            return cls.read_doc(file_path)
        elif file_path.endswith('docx'):
            return cls.read_doc(file_path)
        else:
            raise ValueError("Unsupported file type")

    @classmethod
    def read_pdf(cls, file_path: str):
        loader = UnstructuredPDFLoader(file_path)
        docs = loader.load()
        return docs[0].page_content[:]
    
    @classmethod
    def read_doc(cls, file_path: str): 
        loader = UnstructuredWordDocumentLoader(file_path)
        docs = loader.load() # list(document)
        return docs[0].page_content[:]
    
    @classmethod
    def read_docx(cls, file_path: str): 
        loader = UnstructuredWordDocumentLoader(file_path)
        docs = loader.load() # list(document)
        return docs[0].page_content[:]

    @classmethod
    def read_markdown(cls, file_path: str):
        loader = UnstructuredMarkdownLoader(file_path)
        docs = loader.load()
        return docs[0].page_content[:]

    @classmethod
    def read_text(cls, file_path: str):
        loader = UnstructuredFileLoader(file_path)
        docs = loader.load()
        return docs[0].page_content[:]


class Documents:
    """
        获取已分好类的json格式文档
    """
    def __init__(self, path: str = '') -> None:
        self.path = path
    
    def get_content(self):
        with open(self.path, mode='r', encoding='utf-8') as f:
            content = json.load(f)
        return content
    

def format_document_output(doc: Document, chunk_number: int = None) -> str:
    """
    格式化文档输出，使其更易读
    """
    output = []
    
    # 添加文档编号
    if chunk_number is not None:
        output.append(f"\n{'='*50}")
        output.append(f"文档块 #{chunk_number}")
        output.append(f"{'='*50}")
    
    # 添加内容
    output.append("\n内容:")
    output.append("-" * 30)
    content = doc.page_content
    # 如果内容太长，只显示前200个字符
    if len(content) > 200:
        content = content[:200] + "..."
    output.append(content)
    
    # 添加元数据
    output.append("\n元数据:")
    output.append("-" * 30)
    for key, value in doc.metadata.items():
        output.append(f"{key}: {value}")
    
    return "\n".join(output)

def save_document_output(docs: List[Document], output_dir: str = "output", filename: str = "document_output.txt"):
    """
    将文档输出保存到文件
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # 写入文档处理结果
        f.write("\n文档处理结果:\n")
        f.write("=" * 50 + "\n")
        f.write(f"总文档块数: {len(docs)}\n")
        f.write("=" * 50 + "\n")
        
        # 写入每个文档块
        for i, doc in enumerate(docs, 1):
            f.write(format_document_output(doc, i))
            f.write("\n")
        
        # 写入统计信息
        f.write("\n统计信息:\n")
        f.write("=" * 50 + "\n")
        f.write(f"总文档块数: {len(docs)}\n")
        f.write(f"平均内容长度: {sum(len(doc.page_content) for doc in docs) / len(docs):.2f} 字符\n")
        f.write(f"元数据字段: {list(docs[0].metadata.keys()) if docs else '无'}\n")
        f.write("=" * 50 + "\n")
    
    return output_path

if __name__ == "__main__":
    # 测试文件读取
    docs = ReadFiles("./test/").get_content(chunk_size=256, chunk_overlap=32)
    
    # 保存输出到文件
    output_path = save_document_output(docs)
    print(f"\n输出已保存到: {output_path}")
    
    # 同时在控制台显示
    print("\n文档处理结果:")
    print("=" * 50)
    print(f"总文档块数: {len(docs)}")
    print("=" * 50)
    
    # 打印每个文档块
    for i, doc in enumerate(docs, 1):
        print(format_document_output(doc, i))
    
    # 打印统计信息
    print("\n统计信息:")
    print("=" * 50)
    print(f"总文档块数: {len(docs)}")
    print(f"平均内容长度: {sum(len(doc.page_content) for doc in docs) / len(docs):.2f} 字符")
    print(f"元数据字段: {list(docs[0].metadata.keys()) if docs else '无'}")
    print("=" * 50)