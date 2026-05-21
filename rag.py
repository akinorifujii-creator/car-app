import streamlit as st
import pandas as pd
import anthropic
import numpy as np
import faiss
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

# データ読み込み
df = pd.read_csv('car data.csv')

# ドキュメント作成
documents = []
for _, row in df.iterrows():
    text = f"Car:{row['Car_Name']} Year:{row['Year']} Selling_Price:{row['Selling_Price']} Present_Price:{row['Present_Price']} Kms:{row['Kms_Driven']} Fuel:{row['Fuel_Type']} Seller:{row['Seller_Type']} Transmission:{row['Transmission']}"
    documents.append(text)

# ベクトル化とFAISSインデックス作成
@st.cache_resource
def load_rag():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(documents)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))
    return model, index

model, index = load_rag()

def translate_to_english(question: str, client: anthropic.Anthropic) -> str:
    """日本語の質問を英語の検索クエリに変換する"""
    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": f"以下の質問を英語の検索キーワード（5単語以内）に変換してください。キーワードだけ答えてください。\n質問:{question}"
        }]
    )
    return message.content[0].text.strip()

def search(query: str, top_k: int = 5) -> list:
    """FAISSで関連ドキュメントを検索する"""
    query_vector = model.encode([query])
    distances, indices = index.search(query_vector.astype(np.float32), top_k)
    return [documents[i] for i in indices[0]]

# タイトル
st.title("🚗 車両データ分析アシスタント")

# サイドバー
st.sidebar.header("データ概要")
st.sidebar.write(f"総台数: {len(df)}台")
st.sidebar.write(f"平均価格: {df['Selling_Price'].mean():.2f}ラク")
st.sidebar.write(f"最高価格: {df['Selling_Price'].max():.2f}ラク")

# タブ
tab1, tab2 = st.tabs(["💬 チャット", "📊 グラフ"])

# タブ1: チャット
with tab1:
    api_key = st.text_input("Anthropic APIキーを入力してください", type="password")

    if "history" not in st.session_state:
        st.session_state.history = []

    for msg in st.session_state.history:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").write(msg["content"])

    question = st.chat_input("質問を入力してください")

    if question and api_key:
        client = anthropic.Anthropic(api_key=api_key)

        st.chat_message("user").write(question)

        # 日本語→英語に変換
        english_query = translate_to_english(question, client)
        st.sidebar.write(f"検索クエリ: {english_query}")

        # RAGで関連データを検索
        related_docs = search(english_query)
        context = "\n".join(related_docs)

        # システムプロンプト（関連データだけ渡す）
        system_prompt =