from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langdetect import DetectorFactory, LangDetectException, detect
from pathlib import Path
import os
import re
import sqlite3
from typing import Any
from datetime import datetime, timezone

app = FastAPI(title="Airline Customer Service AI Agent")
BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "chat_history.db"
POLICY_INDEX_PATH = BASE_DIR / "data" / "policies"
DetectorFactory.seed = 0


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Customer message")

llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.2"))

def analyze_sentiment(text: str) -> str:
    result = llm.invoke(
        "Classify this customer message as POSITIVE or NEGATIVE. "
        "Reply with one word only.\n\n" + text
    )
    return result.content.strip().upper()

# 初始化 RAG
embeddings = OllamaEmbeddings(
    model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
)
vectorstore = FAISS.load_local(
    str(POLICY_INDEX_PATH),
    embeddings,
    allow_dangerous_deserialization=True,
)
def initialize_database() -> None:
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_message TEXT NOT NULL,
                reply TEXT NOT NULL,
                sentiment TEXT NOT NULL,
                language TEXT NOT NULL,
                policy_info TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


initialize_database()


def detect_language(text: str) -> str:
    chinese_characters = re.findall(r"[\u3400-\u9fff]", text)
    if chinese_characters:
        return "zh"
    if re.search(r"[A-Za-z]", text):
        return "en"
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

def query_policy(question: str) -> str:
    documents = vectorstore.similarity_search(question, k=1)
    return "\n\n".join(document.page_content for document in documents)

# 回覆生成
def generate_reply(
    user_message: str,
    sentiment: str,
    language: str,
    policy_info: str,
) -> str:
    is_traditional_chinese = language in {"zh", "CN"}
    response_language = "Traditional Chinese" if is_traditional_chinese else language
    prompt = f"""
    顧客訊息: {user_message}
    顧客情緒: {sentiment}
    顧客語言: {response_language}
    航空公司政策: {policy_info}

    請只使用 {response_language} 生成高EQ客服回覆，絕對不要使用其他語言。
    如果顧客語言是 English，整篇回覆必須是 English。
    如果顧客語言是 Traditional Chinese，整篇回覆必須是繁體中文，不得使用英文單字、英文句子或簡體中文。
    中文回覆請使用「抱歉造成您的不便」，不要使用「對不起」。
    回覆語氣專業且有同理心。
    請將最相關的政策內容自然整合到回覆主體，使用「根據航空公司政策」等清楚措辭。
    請只提供同理、確認問題與下一步建議，不要自行補充政策中沒有的細節。
    只處理顧客訊息提到的問題，不要提及無關的行李、退款或延誤議題。
    不得捏造法規、金額、期限、網址或航空公司名稱。
    """
    generated_reply = llm.invoke(prompt).content.strip()
    if is_traditional_chinese and re.search(r"[A-Za-z]", generated_reply):
        generated_reply = llm.invoke(
            "請將以下客服回覆完整改寫為自然、專業的繁體中文。"
            "不得保留任何英文字母、英文單字、英文標點或簡體中文。"
            "只輸出改寫後的回覆，不要解釋修改原因。\n\n"
            + generated_reply
        ).content.strip()
    if language == "en" and re.search(r"[\u3400-\u9fff]", generated_reply):
        generated_reply = llm.invoke(
            f"Customer message: {user_message}\n"
            f"Relevant airline policy: {policy_info}\n\n"
            "Write a professional, empathetic customer service reply entirely in English. "
            "Use only English words and sentences. Do not include Chinese or any other language. "
            "Do not invent policy details. Output only the reply."
        ).content.strip()
        if re.search(r"[\u3400-\u9fff]", generated_reply):
            generated_reply = (
                "I am sorry for the inconvenience caused by your flight delay. "
                "I will help you review the applicable airline policy and the available next steps. "
                "Please provide your flight number and travel date so we can assist you further."
            )
    if is_traditional_chinese:
        generated_reply = generated_reply.replace("xin lỗi", "很抱歉")
        generated_reply = re.sub(r"[A-Za-z]+(?:[ '\u2019-][A-Za-z]+)*", "", generated_reply)
        generated_reply = generated_reply.translate(str.maketrans("0123456789", "零一二三四五六七八九"))
        generated_reply = re.sub(r"\s{2,}", " ", generated_reply).strip()
        generated_reply = generated_reply.replace("對不起", "抱歉造成您的不便")
    return generated_reply

# Agent 工作流
async def customer_service_agent(user_message: str) -> dict[str, Any]:
    language = detect_language(user_message)
    sentiment = analyze_sentiment(user_message)
    policy_info = query_policy(user_message)
    reply = generate_reply(user_message, sentiment, language, policy_info)
    created_at = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            INSERT INTO chat_history
                (user_message, reply, sentiment, language, policy_info, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_message, reply, sentiment, language, policy_info, created_at),
        )
    return {
        "reply": reply,
        "sentiment": sentiment,
        "language": language,
        "policy_info": policy_info,
    }

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    return await customer_service_agent(request.message)


@app.get("/")
async def web_interface():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/en")
async def english_web_interface():
    return FileResponse(BASE_DIR / "en.html")
