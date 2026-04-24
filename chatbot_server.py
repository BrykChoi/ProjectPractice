from flask import Flask, request, jsonify
from flask_cors import CORS
import boto3
from langchain_aws import ChatBedrock
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import PromptTemplate

app = Flask(__name__)
CORS(app)

bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

bedrock = ChatBedrock(
    client=bedrock_client,
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    model_kwargs={"anthropic_version": "bedrock-2023-05-31"},
)

summary_prompt = PromptTemplate(
    input_variables=["summary", "new_lines"],
    template=(
        "당신은 한국 생활 RPG 게임의 도우미 챗봇입니다. "
        "외국인이 한국에서 겪는 상황(공항 환전, 분실물 센터, 출입국관리사무소, 편의점)에 대해 도움을 줍니다. "
        "AI와 사용자의 대화를 반드시 한글로 간략하게 요약해서 메모리를 유지해라.\n"
        "기존 내용: {summary}\n새로운 내용: {new_lines}"
    ),
)

memory = ConversationSummaryMemory(
    llm=bedrock,
    memory_key="history",
    return_messages=True,
    max_token_limit=1000,
    prompt=summary_prompt,
)

SYSTEM_PREFIX = (
    "당신은 '한국 생활 RPG' 게임의 도우미 챗봇입니다. "
    "외국인이 한국에서 겪는 다양한 상황에 대해 친절하고 쉽게 안내해 주세요. "
    "게임에는 4가지 미션이 있습니다: "
    "1) 공항 환전 - 달러를 원으로 환전하기 "
    "2) 공항 분실물 센터 - 잃어버린 휴대폰 찾기 "
    "3) 출입국관리사무소 - 외국인 등록 신청하기 "
    "4) 편의점 - 교통 카드 구매 및 충전하기. "
    "사용자가 게임 관련 질문이나 한국 생활 관련 질문을 하면 도움을 주세요. "
    "항상 한글로 답변하세요.\n\n"
)

conversation = ConversationChain(
    llm=bedrock, memory=memory, verbose=True
)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")
    if not user_message:
        return jsonify({"error": "메시지가 비어있습니다."}), 400

    try:
        response = conversation.run(input=SYSTEM_PREFIX + user_message)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
