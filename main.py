from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import json

from genai.chat import generate_chat
from genai.keyword import generate_keyword
from genai.quiz import generate_quiz
from genai.summary import generate_summary

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    msgNum: int
    msgType: str
    text: str
    clientId: str
    sessionId: str

class ChatResponse(BaseModel):
    msgNum: int
    msgType: str = "1"
    text: str
    clientId: str
    sessionId: str
    status: str
    imgUrl: str
    lectureUrl: str

class KeywordRequest(BaseModel):
    lectureCode: str
    lectureText: str

class Keyword(BaseModel):
    time: str
    name: str
    describe: str
class KeywordResponse(BaseModel):
    lectureCode: str
    keyword: List[Keyword]

class QuizRequest(BaseModel):
    lectureCode: str
    lectureText: str

class Quiz(BaseModel):
    answer: str
    example: List[str]
    explain: str
    lectureCode: str
    question: str
    questionNum: str

class QuizResponse(BaseModel):
    Responses: List[Quiz]

class SummaryRequest(BaseModel):
    lectureCode: str
    lectureText: str

class SummaryResponse(BaseModel):
    lectureCode: str
    summary: str

@app.post("/genai/chat")
async def chat(request_data: ChatRequest):
    msgNum = request_data.msgNum
    sessionId = request_data.sessionId
    clientId = request_data.clientId
    gen = json.loads(await generate_chat(request_data.text))
    text = gen["text"]
    status = gen["status"]
    imgUrl = gen["ImgUrl"]
    lectureUrl = gen["lectureUrl"]

    result = ChatResponse(
        msgNum = msgNum + 1,
        text = text,
        clientId = clientId,
        sessionId = sessionId,
        status = status,
        imgUrl = imgUrl,
        lectureUrl = lectureUrl
    )
    return result

@app.post("/genai/keyword")
async def keyword(request_data: KeywordRequest):
    gen = await generate_keyword(str(request_data))
    lectureCode = gen["lectureCode"]
    keyword = gen["keyword"]

    result = KeywordResponse(
        lectureCode = lectureCode,
        keyword = keyword
    )
    return result

@app.post("/genai/quiz")
async def quiz(request_data: QuizRequest):
    gen = await generate_quiz(str(request_data))
    result = QuizResponse(
        Responses = gen
    )
    return result.Responses

@app.post("/genai/summary")
async def summary(request_data: SummaryRequest):
    gen = await generate_summary(str(request_data))
    lectureCode = gen["lectureCode"]
    summary = gen["summary"]
    result = SummaryResponse(
        lectureCode = lectureCode,
        summary = summary
    )
    return result

