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

위 강의 내용의 핵심내용들을 포함한 요약을 5문장으로 작성해줘.
lectureCode에는 강의의 lectureCode를 작성해줘.
summary에는 강의 요약 내용을 작성해줘.

[응답 양식]은 아래 예시와 같은 JSON 형식으로 작성해야 합니다.
[응답 양식] 출력 예시:
{{"lectureCode":"001", "summary":"강의 요약 내용"}}
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

def generate_summary(request):
    res = llm_chain(str(request))['text']
    if validate_data(res) == True:
        return res
    else:
        generate_summary(request)
