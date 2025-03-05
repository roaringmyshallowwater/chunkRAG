#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from copy import copy
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import logging

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Config:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.zhipu_api_key = os.getenv("ZHIPUAI_API_KEY")
        self.validate()
    
    def validate(self):
        if not self.zhipu_api_key:
            raise ValueError("ZHIPUAI_API_KEY is required")


class BaseEmbeddings:
    """
    Base class for embeddings
    """

    def __init__(self, path: str, is_api: bool) -> None:
        """
        :param path: 本地embeddings
        :param is_api:
        """
        self.path = path
        self.is_api = is_api

    def get_embedding(self, text: str, *args, **kwargs) -> List[float]:
        raise NotImplementedError

    @classmethod
    def cosine_similarity(cls, vector1: List[float], vector2: List[float]) -> float:
        """
        calculate cosine similarity between two vectors
        """
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude


class OpenAIEmbedding(BaseEmbeddings):
    """
    class for OpenAI embeddings
    """

    def __init__(self, path: str = '', is_api: bool = True) -> None:
        super().__init__(path, is_api)
        if self.is_api:
            from openai import OpenAI
            self.client = OpenAI()
            self.client.api_key = os.getenv("OPENAI_API_KEY")
            self.client.base_url = os.getenv("OPENAI_BASE_URL")

    def get_embedding(self, text: str, model: str = "text-embedding-3-large") -> List[float]:
        if self.is_api:
            text = text.replace("\n", " ")
            return self.client.embeddings.create(input=[text], model=model).data[0].embedding
        else:
            raise NotImplementedError


class ZhipuEmbedding(BaseEmbeddings):
    """
    class for Zhipu embeddings
    """

    def __init__(self, path: str = '', is_api: bool = True) -> None:
        super().__init__(path, is_api)
        if self.is_api:
            from zhipuai import ZhipuAI
            self.client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))

    def get_embedding(self, text: str, *args, **kwargs) -> List[float]:
        logger.info(f"Getting embedding for text: {text[:100]}...")
        try:
            response = self.client.embeddings.create(
                model="embedding-2",
                input=text,
                *args,
                **kwargs
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to get embedding: {str(e)}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using Zhipu AI."""
        embeddings = []
        for text in texts:
            embedding = self.get_embedding(text)
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using Zhipu AI."""
        return self.get_embedding(text)