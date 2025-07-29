import os
import requests
import tempfile
import logging
import re

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# === Load API Key ===
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# === FastAPI App ===
app = FastAPI()

# === Logger Setup ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Input Schema ===
class HackRxInput(BaseModel):
    documents: str  # Publicly accessible PDF URL
    questions: List[str]

# === Embedding and LLM Setup ===
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="gemma2-9b-it",
    temperature=0.2,
    model_kwargs={"max_completion_tokens": 200}
)


# === QA Prompt Template ===
qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a legal policy analyst specializing in Indian insurance documentation.

Using ONLY the provided policy context, answer the question with utmost precision. Your answer must:
- Be formal and use proper insurance terminology.
- Include only essential details: e.g., waiting periods, eligibility, limits, exclusions.
- Be structured and to-the-point: strictly 1–3 sentences only.
- Do NOT add disclaimers, assumptions, or filler phrases.

Format:
Answer: <your final concise legal answer here>

Context:
{context}

Question:
{question}
"""
)

# === Answer Trimming Function ===
def clean_and_trim_answer(text, max_sentences=3, max_words=60):
    import nltk
    nltk.download('punkt', quiet=True)
    from nltk.tokenize import sent_tokenize, word_tokenize

    sentences = sent_tokenize(text)
    trimmed = " ".join(sentences[:max_sentences])
    words = word_tokenize(trimmed)
    if len(words) > max_words:
        trimmed = " ".join(words[:max_words]) + "..."
    return trimmed.strip()

@app.post("/hackrx/run")
def handle_rag_request(payload: HackRxInput):
    pdf_path = None
    try:
        logger.info(f"📄 Downloading PDF from: {payload.documents}")
        response = requests.get(payload.documents)
        if response.status_code != 200 or not response.headers["Content-Type"].startswith("application/pdf"):
            return {"error": "❌ Invalid or inaccessible PDF URL."}

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            pdf_path = tmp_file.name

        logger.info("✅ PDF downloaded and saved temporarily.")

        # === Document Loading & Splitting ===
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(documents)
        logger.info(f"📑 Loaded and split {len(chunks)} chunks.")

        # === Embedding & FAISS VectorStore ===
        vectorstore = FAISS.from_documents(chunks, embedding=embedding_model)
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 6, "lambda_mult": 0.7})
        logger.info("🧠 FAISS vectorstore with MMR retriever created.")

        # === QA Chain Setup ===
        qa_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=qa_prompt)
        answer_list = []

        for q in payload.questions:
            logger.info(f"❓ Question: {q}")
            docs = retriever.get_relevant_documents(q)
            result = qa_chain.run({"input_documents": docs, "question": q})
            trimmed_answer = clean_and_trim_answer(result.strip())
            answer_list.append(trimmed_answer)

        logger.info("✅ All questions answered.")
        return {"answers": answer_list}

    except Exception as e:
        logger.error(f"⚠️ Error: {str(e)}")
        return {"error": str(e)}

    finally:
        if pdf_path and os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)
                logger.info("🧹 Temporary file cleaned.")
            except Exception as cleanup_error:
                logger.warning(f"⚠️ Cleanup failed: {cleanup_error}")
