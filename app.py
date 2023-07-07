from flask import Flask, jsonify
from flask_cors import CORS
from flask import request
import json

from genai.quiz import generate_quiz
from genai.chat import generate_chat
from genai.summary import generate_summary
from genai.keyword import generate_keyword

app = Flask(__name__)

#CORS(app, origins='http://203.250.148.52:28881', supports_credentials=True)
CORS(app)

@app.route('/genai/quiz', methods=['POST'])
def make_quiz():
    lectureData = str(request.json)
    response = generate_quiz(lectureData)
    return jsonify(json.loads(response))
#    return json.dumps(str(response), ensure_ascii=False)

@app.route('/genai/chat', methods=['POST'])
def make_chat():
    msgNum = int(request.json.get("msgNum"))
    sessionId = request.json.get("sessionId")
    clientId = request.json.get("clientId")
    gen = generate_chat(request.json.get("text"))
    gen = json.loads(str(gen))
    text = gen["text"]
    status = gen["status"]
    ImgUrl = gen["ImgUrl"]
    lectureUrl = gen["lectureUrl"]
    response = {"msgNum":str(msgNum+1), "msgType":"1", "text":text, "clientId":clientId, "status":status, "imgUrl":ImgUrl, "lectureUrl":lectureUrl, "sessionId": sessionId}
    return jsonify(response)

@app.route('/genai/summary', methods=['POST'])
def make_summary():
    lectureData = str(request.json)
    response = generate_summary(lectureData)
    return jsonify(json.loads(response))
    # return json.dumps(str(response), ensure_ascii=False)

@app.route('/genai/keyword', methods=['POST'])
def make_keyword():
    lecturData = str(request.json)
    response = generate_keyword(lecturData)
    return jsonify(json.loads(response))
    # return json.dumps(str(response), ensure_ascii=False)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000')
    
#application = app