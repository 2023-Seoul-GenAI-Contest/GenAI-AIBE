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

def generate_quiz(request):
    res = llm_chain(str(request))['text']
    if validate_data(res) == True:
        return res
    else:
        generate_quiz(request)
