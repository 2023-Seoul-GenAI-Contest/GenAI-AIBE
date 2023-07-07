from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
)

chat_llm = ChatOpenAI(model_name="gpt-3.5-turbo")

system_template="""
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

system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])

llm_chain = LLMChain(
    llm=chat_llm,
    prompt=chat_prompt,
)

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

def generate_keyword(request):
    res = llm_chain(str(request))['text']
    if validate_data(res) == True:
        return res
    else:
        generate_keyword(request)
