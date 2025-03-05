#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from typing import Dict, List, Optional, Tuple, Union
import json

from weaviate.util import generate_uuid5

from RAG.Embeddings import BaseEmbeddings, OpenAIEmbedding, ZhipuEmbedding
import numpy as np
from tqdm import tqdm

from langchain_community.vectorstores import Weaviate
import weaviate
import os
from weaviate.auth import AuthApiKey

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())


# getpass.getpass(':') # 在终端显示提示


class VectorBase:
    """
    基础向量存储类，定义了存储和查询向量的接口。
    """

    def __init__(self, class_name: str = None, docs: List[str] = None):
        if docs is None:
            docs = []
        self.docs = docs
        self.class_name = class_name
        self.vectors = []

    def get_vector(self, EmbeddingModel: BaseEmbeddings) -> List[List[float]]:
        """
        根据提供的嵌入模型，为文档生成向量。
        """
        raise NotImplementedError("Must be implemented by subclass.")

    def query(self, query: str, EmbeddingModel: BaseEmbeddings, k: int = 1) -> List[str]:
        """
        查询与给定查询最相似的文档。
        """
        raise NotImplementedError("Must be implemented by subclass.")


class BaseVectorBase(VectorBase):
    def __init__(self, class_name: str = None, docs: List[str] = None):
        super().__init__(class_name, docs)

    def get_vector(self, EmbeddingModel: BaseEmbeddings) -> List[List[float]]:
        for doc in tqdm(self.docs, desc="Calculating embeddings"):
            self.vectors.append(EmbeddingModel.get_embedding(doc))
        return self.vectors

    def persist(self):
        path = self.class_name
        if not os.path.exists(path):
            os.makedirs(path)
        with open(f"{path}/docs.json", 'w', encoding='utf-8') as f:
            json.dump(self.docs, f, ensure_ascii=False)
        if self.vectors:
            with open(f"{path}/vectors.json", 'w', encoding='utf-8') as f:
                json.dump(self.vectors, f)

    def load_vector(self):
        path = self.class_name
        with open(f"{path}/vectors.json", 'r', encoding='utf-8') as f:
            self.vectors = json.load(f)
        with open(f"{path}/docs.json", 'r', encoding='utf-8') as f:
            self.docs = json.load(f)

    def get_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        return BaseEmbeddings.cosine_similarity(vector1, vector2)

    def query(self, query: str, EmbeddingModel: BaseEmbeddings, k: int = 1) -> List[str]:
        query_vector = EmbeddingModel.get_embedding(query)
        result = np.array([self.get_similarity(query_vector, vector)
                           for vector in self.vectors])
        return np.array(self.docs)[result.argsort()[-k:][::-1]].tolist()


class WeaviateVectorBase(VectorBase):
    """
    使用Weaviate的向量存储和查询。
    """

    def __init__(self, class_name: str = None, docs: List[str] = None):
        super().__init__(class_name, docs)
        self.OPEN_API_KEY = os.getenv("OPENAI_API_KEY")
        self.WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
        self.WEAVIATE_URL = os.getenv('WEAVIATE_URL')

        # --------------------- 连接weaviate --------------------- #
        self.client = weaviate.Client(
            url=self.WEAVIATE_URL,
            auth_client_secret=AuthApiKey(self.WEAVIATE_API_KEY),
            additional_headers={
                'X-OpenAI-Api-Key': self.OPEN_API_KEY
            }
        )
        self.initialize_schema()

    def initialize_schema(self):
        # --------------------- 创建新的class --------------------- #
        class_obj = {
            "class": self.class_name,  # class name
            "description": self.class_name,
            "vectorizer": "text2vec-openai",
            "moduleConfig": {
                "text2vec-openai": {  # 生成嵌入的模型。对于文本对象，您通常会根据您正在使用的提供程序选择其中一个 text2vec 模块
                    "vectorizeClassName": False,  # 类名不会被矢量化
                    "model": "ada",
                    "modelVersion": "002",
                    "type": "text"
                }
            },
            "properties": [  # 属性 相当于属性列
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "The content",
                    "moduleConfig": {  # 在这里，您可以定义所用模块的详细信息。例如，矢量化器是一个模块，您可以为其定义要使用的模型和版本。
                        "text2vec-openai": {
                            "skip": False,
                            "vectorizePropertyName": False
                        }
                    }
                },
            ],
        }  # 需要创建的类
        # --------------------- 检查是否已经建立 --------------------  #
        schema = self.client.schema.get()
        class_names = [s["class"].lower() for s in schema["classes"]]
        if self.class_name in class_names:
            print(f"Class: \"{self.class_name}\" already exists.")
        else:
            print(f"Class: \"{self.class_name}\" does not exist, create the class.")
            self.client.schema.create_class(class_obj)
            print(f"Class \"{self.class_name}\" schema initialization complete.")

        # self.client.schema.delete_all() # 清空
        # self.client.schema.get()
        # client.schema.get(self.class_name)展示库的架构

    def get_vector(self, EmbeddingModel: BaseEmbeddings):
        """
        覆盖基类方法，使用Weaviate添加文档及其向量。
        """
        if len(self.docs):
            # --------------------- 文本向量化 --------------------  #
            self.vectors = []  # 更新self.vectors，get_vectors()返回值。
            self.chunks = [doc.page_content[:] for doc in self.docs]
            for doc in tqdm(self.chunks, desc="Calculating embeddings"):
                self.vectors.append(EmbeddingModel.get_embedding(doc))
            self.upload_docs()
        else:
            # --------------------- 查询返回class中所有向量 --------------------  #
            response = (
                self.client.query.get(self.class_name, ["content"])
                .with_additional("vector")
                .do()
            )
            classname = self.class_name[0].upper() + self.class_name[1:] # 大写首字母
            self.vectors = [item['_additional']['vector'] for item in response['data']['Get'][classname]]
        return self.vectors

    def upload_docs(self):
        # --------------------- 上传向量库 --------------------  #
        print(f"Uploading {len(self.docs)} documents to the vector store.")
        self.client.batch.configure(batch_size=100)
        with self.client.batch as batch:
            for chunk, vector in zip(self.chunks, self.vectors):  # 使用self.vectors中的向量
                properties = {
                    "content": chunk,
                }
                uuid = generate_uuid5(chunk)  # 假设使用文档内容生成UUID
                batch.add_data_object(
                    properties,
                    class_name=self.class_name,
                    vector=vector,  # 将向量与文档一起插入
                    uuid=uuid
                )
        print("Upload complete.")

    def query(self, query: str, EmbeddingModel: BaseEmbeddings, k: int = 5):
        """
        def query_weaviate(query, collection_name, top_k=20):

            # Creates embedding vector from user query
            embedded_query = openai.Embedding.create(
                input=query,
                model=EMBEDDING_MODEL,
            )["data"][0]['embedding']

            near_vector = {"vector": embedded_query}

            # Queries input schema with vectorised user query
            query_result = (
                client.query
                .get(collection_name, ["title", "content", "_additional {certainty distance}"])
                .with_near_vector(near_vector)
                .with_limit(top_k)
                .do()
            )

            return query_result
        """
        # retriever = self.vectors.as_retriever()
        # return retriever

        # return self.vectors.similarity_search(query)

        query_embedding = EmbeddingModel.get_embedding(query)
        result = self.client.query.get(self.class_name, ["content"]).with_near_vector({
            "vector": query_embedding
        }).with_limit(k).do()
        return [item["content"] for item in result["data"]["Get"][self.class_name]]


    def persist(self):
        """
        在Weaviate中，数据自动被持久化。这个方法可以用于额外的数据处理或记录日志。
        """
        print("Data is automatically persisted in Weaviate.")

    def load_vector(self):
        """
        从Weaviate重新加载文档到`self.docs`。这个例子中假设我们查询所有的self.class_name类实例。
        """
        self.docs = []
        query_result = self.client.query.get(self.class_name, properties=["content"]).with_limit(100).do()
        if query_result and 'data' in query_result and 'Get' in query_result['data']:
            for item in query_result['data']['Get']['law']:
                self.docs.append(item['content'])
        print("docs reloaded from Weaviate.")
