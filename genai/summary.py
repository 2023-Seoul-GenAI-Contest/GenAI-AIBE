from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import List, Dict
from langchain_core.pydantic_v1 import BaseModel, Field

chat_llm = ChatOpenAI(model_name="gpt-4-0125-preview")

prompt_template="""
강의 내용 : 
{lecture}

위 강의 내용의 핵심내용들을 포함한 요약을 5문장으로 작성해줘.
lectureCode에는 강의의 lectureCode를 작성해줘.
summary에는 강의 요약 내용을 작성해줘.

[응답 양식]은 아래 예시와 같은 JSON 형식으로 작성해야 합니다.
[응답 양식] 출력 예시:
{{"lectureCode":"001", "summary":"강의 요약 내용"}}
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["lecture"]
)

class Output(BaseModel):
    lectureCode: str = Field(description="강의 코드")
    summary: str = Field(description="강의 요약")

parser = JsonOutputParser(pydantic_object=Output)

chain = PROMPT | chat_llm | parser

async def generate_summary(request):
    res = await chain.ainvoke({"lecture": request})
    return res
