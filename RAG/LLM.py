#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from typing import Dict, List, Optional, Tuple, Union
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import TextStreamer

PROMPT_TEMPLATE = dict(
    RAG_PROMPT_TEMPALTE="""
You are a useful assistant for provide professional customer question answering. 
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Must reply using the language used in the user's question.
Context: {context}
Question: {question}
Answer:""",
    RAG_PROMPT_TEMPALTE_2="""使用以上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
        问题: {question}
        可参考的上下文：
        ···
        {context}
        ···
        如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。"""
)


class BaseModel:
    def __init__(self, path: str = '') -> None:
        self.path = path

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        pass

    def load_model(self):
        pass


class OpenAIChat(BaseModel):
    def __init__(self, path: str = '', model: str = "gpt-3.5-turbo") -> None:
        super().__init__(path)
        self.model = model

    def chat(self, query: str, history: List[dict], retriever) -> str:
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE['RAG_PROMPT_TEMPALTE'])
        llm = ChatOpenAI(model_name=self.model, temperature=0)
        rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
        )
        return rag_chain.invoke(query)

    def chat_2(self, prompt: str, history: List[dict], content: str) -> str:
        from openai import OpenAI
        client = OpenAI()
        history.append({'role': 'user', 'content': PROMPT_TEMPLATE['RAG_PROMPT_TEMPALTE'].format(question=prompt, context=content)})
        response = client.chat.completions.create(
            model=self.model,
            messages=history,
            max_tokens=2048,
            temperature=0.1
        )
        return response.choices[0].message.content

    def get_stream_response(self, prompt: str, content: str) -> str:
        from openai import OpenAI
        client = OpenAI()
        history = [{'role': 'user', 'content': PROMPT_TEMPLATE['RAG_PROMPT_TEMPALTE'].format(question=prompt, context=content)}]
        response = client.chat.completions.create(
            model=self.model,
            messages=history,
            max_tokens=150,
            temperature=0.1,
            stream=True
        )

        if response.status_code == 200:
            for chunk in response.iter_content(chunk_size=128):
                if chunk:
                    yield chunk.decode('utf-8')
        else:
            raise Exception(f"Request failed with status code {response.status_code}")


class Deepseek(BaseModel):
    def __init__(self, path: str = '', model: str = "deepseek-chat") -> None:
        super().__init__(path)
        self.model = model
        self.tokenizer = None
        self.model = None

    def load_model(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        model_path = os.environ.get("DEEPSEEK_PATH", "deepseek-ai/deepseek-coder-33b-instruct")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype="auto"
        )

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        if self.model is None:
            self.load_model()
            
        # Format the prompt with context
        formatted_prompt = PROMPT_TEMPLATE['RAG_PROMPT_TEMPALTE'].format(
            question=prompt,
            context=content
        )
        
        # Add the formatted prompt to history
        history.append({'role': 'user', 'content': formatted_prompt})
        
        # Convert history to the format expected by the model
        messages = []
        for msg in history:
            if msg['role'] == 'user':
                messages.append(f"Human: {msg['content']}")
            elif msg['role'] == 'assistant':
                messages.append(f"Assistant: {msg['content']}")
        
        # Join messages with newlines
        full_prompt = "\n".join(messages)
        
        # Generate response
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.1,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        response_parts = response.split("Assistant: ")
        if len(response_parts) > 1:
            return response_parts[-1].strip()
        return response.strip()

    def get_stream_response(self, prompt: str, content: str):
        if self.model is None:
            self.load_model()
            
        # Format the prompt with context
        formatted_prompt = PROMPT_TEMPLATE['RAG_PROMPT_TEMPALTE'].format(
            question=prompt,
            context=content
        )
        
        # Convert to model input format
        messages = [f"Human: {formatted_prompt}"]
        full_prompt = "\n".join(messages)
        
        # Generate streaming response
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        streamer = TextStreamer(self.tokenizer, skip_prompt=True)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.1,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            streamer=streamer
        )
        
        for output in outputs:
            yield output.decode('utf-8')