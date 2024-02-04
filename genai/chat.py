from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain import hub
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_json_chat_agent

llm = ChatOpenAI(model_name="gpt-4-0125-preview", max_tokens=4000)
embeddings = OpenAIEmbeddings()

db1 = FAISS.load_local("QA", embeddings)
db2 = FAISS.load_local("Recommand", embeddings)

prompt_template1="""
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

prompt_template2="""
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
PROMPT2 = PromptTemplate(
    template=prompt_template2, input_variables=["context", "question"]
)

chain_type_kwargs1 = {"prompt": PROMPT1}
chain_type_kwargs2 = {"prompt": PROMPT2}

qa1 = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-4-0125-preview", max_tokens=4000),
        chain_type="stuff",
        retriever=db1.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs=chain_type_kwargs1,
    )

qa2 = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-4-0125-preview", max_tokens=4000),
        chain_type="stuff",
        retriever=db2.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs=chain_type_kwargs2,
    )

@tool(return_direct=True)
async def simple_qa(query: str) -> str:
    """일반적인 질의 응답에 대해 답을 해야할 때 유용합니다."""
    return await qa1.arun(query)

@tool(return_direct=True)
async def course_reco(query: str) -> str:
    """질문에 '강의추천'이라는 키워드가 포함되었을 때 유용합니다."""
    return await qa2.arun(query)


# 추후에 세션과같은 정보를 받고 각 유저마다 다른 메모리를 제공
# memory = ConversationBufferWindowMemory(
#     k=5,
#     memory_key="chat_history",
#     return_messages=True
# )

prompt = hub.pull("hwchase17/react-chat-json")
tools = [simple_qa, course_reco]

agent = create_json_chat_agent(
    ChatOpenAI(model_name="gpt-4-0125-preview"),
    tools,
    prompt
    )

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)

async def generate_chat(request):
    res = await agent_executor.ainvoke(
        {
            "input": request,
        },
    )
    return res["output"]