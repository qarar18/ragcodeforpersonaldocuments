from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from bs4 import BeautifulSoup
import pdfplumber

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from fastapi.middleware.cors import CORSMiddleware

from playwright.sync_api import sync_playwright

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_name = "declare-lab/flan-alpaca-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
llm_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
local_llm = HuggingFacePipeline(pipeline=llm_pipeline)

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = None
qa_chain = None

def render_dynamic_page(url: str) -> str:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle")
        content = page.content()
        browser.close()
        return content

@app.post("/load_pdf")
async def load_pdf(file: UploadFile = File(...)):
    global vectorstore, qa_chain
    text = ""

    with pdfplumber.open(file.file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""

    if not text.strip():
        return {"error": "No extractable text found in PDF."}

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    chunks = splitter.split_text(text)

    vectorstore = Chroma.from_texts(chunks, embedding_model, persist_directory="./chroma")
    vectorstore.persist()

    qa_chain = RetrievalQA.from_chain_type(
        llm=local_llm,
        retriever=vectorstore.as_retriever()
    )

    return {"message": "✅ PDF uploaded and indexed."}

@app.post("/load_pdf_from_path")
def load_pdf_from_path():
    global vectorstore, qa_chain
    pdf_path = r"D:\OneDrive - Systems Limited\Desktop\RAG1\Phys320_L8.pdf"

    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""

    if not text.strip():
        return {"error": "No text found in PDF"}

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    chunks = splitter.split_text(text)

    vectorstore = Chroma.from_texts(chunks, embedding_model, persist_directory="./chroma")
    vectorstore.persist()

    qa_chain = RetrievalQA.from_chain_type(
        llm=local_llm,
        retriever=vectorstore.as_retriever()
    )

    return {"message": "✅ PDF from path loaded and indexed."}

class URLInput(BaseModel):
    url: str

@app.post("/load_url")
def load_url(data: URLInput):
    global vectorstore, qa_chain

    html_content = render_dynamic_page(data.url)
    soup = BeautifulSoup(html_content, "html.parser")
    main_content = soup.find("main") or soup.body
    text = main_content.get_text(strip=True) if main_content else soup.get_text(strip=True)

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    chunks = splitter.split_text(text)

    vectorstore = Chroma.from_texts(chunks, embedding_model, persist_directory="./chroma")
    vectorstore.persist()

    qa_chain = RetrievalQA.from_chain_type(
        llm=local_llm,
        retriever=vectorstore.as_retriever()
    )

    return {"message": "✅ Web page content loaded and indexed."}

# ❓ Ask question
class QuestionInput(BaseModel):
    question: str

@app.post("/ask")
def ask_question(data: QuestionInput):
    global qa_chain
    if not qa_chain:
        return {"error": "No document loaded yet."}

    answer = qa_chain.run(data.question)
    return {"answer": answer}
