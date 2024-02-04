from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import List, Dict
from langchain_core.pydantic_v1 import BaseModel, Field

chat_llm = ChatOpenAI(model_name="gpt-4-0125-preview")

prompt_template="""
강의 내용 : 
{lecture}

위 강의 내용을 토대로 퀴즈 3가지를 생성해줘.
퀴즈를 생성하고 [응답 양식]의 question에 퀴즈 내용을 작성해줘.
lectureCode에는 강의의 lectureCode를 작성해줘.
example에는 퀴즈 선택지들을 작성해줘.
answer에는 퀴즈 선택지들 중 퀴지의 정답 선택지를 작성해줘.
explain에는 정답 선택지의 이유를 작성해줘.
questionNum에는 퀴즈의 번호를 매겨줘. 01, 02, 03 이렇게.

[응답 양식]은 아래 예시와 같은 JSON 형식의 List으로 작성해야 합니다.
[응답 양식] 출력 예시:
[{{"question":"퀴즈 내용", "lectureCode":"001", "example":["선택지1", "선택지2", "선택지3", "선택지4"], "answer":"선택지2","explain":"정답이유", "questionNum":"01"}}, {{"question":"퀴즈 내용", "lectureCode":"001", "example":["선택지1", "선택지2", "선택지3", "선택지4"], "answer":"선택지1","explain":"정답이유", "questionNum":"02"}}, {{"question":"퀴즈 내용", "lectureCode":"001", "example":["선택지1", "선택지2", "선택지3", "선택지4"], "answer":"선택지4","explain":"정답이유", "questionNum":"03"}}]
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["lecture"]
)

class Quiz(BaseModel):
    answer: str = Field(description="정답")
    example: List[str] = Field(description="선지")
    explain: str = Field(description="정답의 이유")
    lectureCode: str = Field(description="해당 내용이 나온 강의코드")
    question: str = Field(description="문제")
    questionNum: str = Field(description="정답번호")

class Output(BaseModel):
    Responses: List[Quiz]

parser = JsonOutputParser(pydantic_object=Output)

chain = PROMPT | chat_llm | parser

async def generate_quiz(request):
    res = await chain.ainvoke({"lecture": request})
    return res