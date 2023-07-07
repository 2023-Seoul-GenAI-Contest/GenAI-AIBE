from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.agents import AgentType

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

from langchain.chains import RetrievalQA
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory

embeddings = OpenAIEmbeddings()

db1 = FAISS.load_local("Recommand", embeddings)
db2 = FAISS.load_local("QA", embeddings)

prompt_template1="""
추천 강의 목록:
{context}

요청: 
{question}

답변은 [응답 양식]에 작성해줘.
text에는 위 요청 내용을 토대로 강의를 추천하고 그 이유를 작성해줘.
status에는 "True"를 작성해줘.
ImgUrl에는 해당 강의의 이미지url을 작성해줘.
lectureUrl에는 해당 강의의 강의url을 작성해줘.


[응답 양식]은 아래 예시와 같은 JSON 형식으로 작성해야 합니다.
[응답 양식] 출력 예시:
{{"text":"추천하는 강의와 그 이유", "status":"True", "ImgUrl":"이미지url", "lectureUrl":"강의url"}}
"""

PROMPT1 = PromptTemplate(
    template=prompt_template1, input_variables=["context", "question"]
)

chain_type_kwargs1 = {"prompt": PROMPT1}

qa1 = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-3.5-turbo-16k", max_tokens=6000), chain_type="stuff", retriever=db1.as_retriever(search_kwargs={"k": 5}), chain_type_kwargs=chain_type_kwargs1)

prompt_template2="""
context:
{context}

질문: 
{question}

답변은 [응답 양식]에 작성해줘.
text에는 context를 참고하여 위 질문에 답변하고 답변의 이유와 출처 강의 이름을 작성해줘.
status에는 반드시 "False"를 작성해줘.
ImgUrl에는 반드시 "None"을 작성해줘.
lectureUrl에는 반드시 "None"을 작성해줘.


[응답 양식]은 아래 예시와 같은 JSON 형식으로 작성해야 합니다.
[응답 양식] 출력 예시:
{{"text":"질문에 대한 답변과 이유 그리고 출처", "status":"False", "ImgUrl":"None", "lectureUrl":"None"}}
"""

PROMPT2 = PromptTemplate(
    template=prompt_template2, input_variables=["context", "question"]
)

chain_type_kwargs2 = {"prompt": PROMPT2}

qa2 = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-3.5-turbo-16k", max_tokens=6000), chain_type="stuff", retriever=db2.as_retriever(search_kwargs={"k": 5}), chain_type_kwargs=chain_type_kwargs2)

tools = [
    Tool(
        name = "단순 질의응답 시스템",
        func=qa2.run,
        description="일반적인 질의 응답에 대해 답을 해야할 때 유용합니다.",
        return_direct=True,
    ),
    Tool(
        name = "강의 추천 시스템",
        func=qa1.run,
        description="질문에 '강의추천'이라는 키워드가 포함되었을 때 유용합니다.",
        return_direct=True,
    ),
]

memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)

agent_chain = initialize_agent(tools, ChatOpenAI(model_name="gpt-3.5-turbo-16k", max_tokens=6000), agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)

def validate_data(data):
    stack = []
    for item in data:
        if item == "[":
            stack.append("[")
        elif item == "{":
            stack.append("{")
        elif item == "]":
            if len(stack) == 0 or stack[-1] != "[":
                return False
            stack.pop()
        elif item == "}":
            if len(stack) == 0 or stack[-1] != "{":
                return False
            stack.pop()

    return len(stack) == 0
def generate_chat(request):
    res = agent_chain.run(request)
    if validate_data(res) == True:
        return res
    else:
        generate_chat(request)