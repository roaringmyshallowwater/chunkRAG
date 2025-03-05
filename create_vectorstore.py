from RAG.Embeddings import OpenAIEmbedding
from storage import load_vectorstore, get_embeddings
import os
import argparse

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='创建向量库')
    parser.add_argument('--file', type=str, default=r"E:\2025\lawRAG-master\test\2数据安全与治理-陈庄-清华大学出版社.json",
                      help='输入文件路径')
    parser.add_argument('--class_name', type=str, default="test",
                      help='向量库名称')
    parser.add_argument('--embedding_type', type=str, default="zhipu",
                      choices=["zhipu", "openai"],
                      help='embedding类型')
    
    args = parser.parse_args()
    
    # 获取embedding模型
    embeddings = get_embeddings(args.embedding_type)
    
    # 加载JSON文件并创建向量库
    vectorstore = load_vectorstore(
        class_name=args.class_name,
        file_path=args.file,
        embeddings=embeddings
    )
    
    print(f"Faiss向量库创建完成！使用 {args.embedding_type} embedding模型")

if __name__ == "__main__":
    main() 