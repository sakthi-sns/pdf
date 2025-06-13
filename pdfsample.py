import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import Runnable
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ---- Replace with your actual valid Gemini API key ----
GEMINI_API_KEY = "AIzaSyA7zjbbimSB2MiZm5xJxwHrTdpOoRrLBJc"

# ---- Initialize Gemini LLM ----
try:
    llm = ChatGoogleGenerativeAI(
        google_api_key="AIzaSyA7zjbbimSB2MiZm5xJxwHrTdpOoRrLBJc ",
        model="gemini-2.0-flash",
        temperature=0.2
    )
except Exception as e:
    st.error(f"❌ Error initializing Gemini model: {e}")
    st.stop()

# ---- Initialize Embeddings ----
try:
   embeddings = GoogleGenerativeAIEmbeddings(
   google_api_key=GEMINI_API_KEY,
   model="models/embedding-001"
)
except Exception as e:
    st.error(f"❌ Error initializing embeddings: {e}")
    st.stop()

# ---- Streamlit UI ----
st.set_page_config(page_title="📄 PDF RAG Q&A")
st.title("📄 PDF RAG (Retrieval-Augmented Generation) Q&A")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
question = st.text_input("Ask a question about the document:")

if uploaded_file and question and st.button("Ask"):
    try:
        # ---- Save uploaded file ----
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())
        
        # ---- Load and split PDF ----
        loader = PyPDFLoader("temp.pdf")
        pages = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        docs = text_splitter.split_documents(pages)

        # ---- Embed and index ----
        vectorstore = FAISS.from_documents(docs, embeddings)

        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        # ---- Retrieve relevant context ----
        relevant_docs = retriever.get_relevant_documents(question)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # ---- Build RAG prompt ----
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Use the following document context to answer the question."),
            ("human", "Document Context:\n{context}\n\nQuestion: {question}")
        ])

        rag_chain: Runnable = prompt | llm

        # ---- Invoke RAG chain ----
        result = rag_chain.invoke({"context": context, "question": question})

        answer = result.content if hasattr(result, "content") else str(result)

        # ---- Display result ----
        st.success("✅ Answer generated!")
        st.write(f"**Answer:** {answer}")
        
        with st.expander("🔍 Retrieved Context"):
            st.write(context)

    except Exception as e:
        st.error(f"⚠️ An error occurred: {e}")
    finally:
        if os.path.exists("temp.pdf"):
            os.remove("temp.pdf")
