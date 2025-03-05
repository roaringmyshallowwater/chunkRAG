from storage import get_query

PROMPT_TEMPLATE = dict(
    ANALYSIS="""
使用参考的法律知识来分析这段社交文本。总是使用中文回答。
社交文本: {question}
可参考的法律知识：{context}
"""
)


def analyze_text(context, classname='law'):
    """
    :param context: 需要分析的法律文本
    :param classname: 查询的法律知识库
    :return: 分析文本与查询到的法律内容
    """
    ans, text_ls = get_query(classname, context, template=PROMPT_TEMPLATE['RAG_PROMPT_TEMPALTE'])
    text = ans
    for i in range(len(text_ls)):
        text += text_ls[i] + "\n"
    return ans, text
