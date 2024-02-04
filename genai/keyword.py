from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import List, Dict
from langchain_core.pydantic_v1 import BaseModel, Field

chat_llm = ChatOpenAI(model_name="gpt-4-0125-preview")

prompt_template="""
강의 내용 : 
{lecture}

위 강의 내용의 각타임라인의 키워드를 작성하고 해당 키워드에 대한 설명을 작성해줘.
lectureCode에는 강의의 lectureCode를 작성해줘.
time에는 해당 키워드를 작성한 타임라인을 작성해줘.
name에는 해당 키워드를 작성해줘.
describe에는 해당 키워드에 대한 설명을 작성해줘.

[응답 양식]은 아래 예시와 같은 JSON 형식으로 작성해야 합니다.
[응답 양식] 출력 예시:
{{"lectureCode":"001", "keyword":[{{"time":"00:05", "name":"키워드1", "describe":"키워드1에대한 설명"}}, {{"time":"01:07", "name":"키워드2", "describe":"키워드2에대한 설명"}}, {{"time":"03:08", "name":"키워드3", "describe":"키워드3에대한 설명"}}]}}  
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["lecture"]
)
class Keyword(BaseModel):
    time: str = Field(description="해당 내용이 언급된 시간")
    name: str = Field(description="키워드")
    describe: str = Field(description= "키워드에 대한 설명")

class Output(BaseModel):
    lectureCode: str = Field(description="강의코드")
    keyword: List[Keyword]

parser = JsonOutputParser(pydantic_object=Output)

chain = PROMPT | chat_llm | parser

async def generate_keyword(request):
    res = await chain.ainvoke({"lecture": request})
    return res
