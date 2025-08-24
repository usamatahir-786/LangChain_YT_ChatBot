import os
import streamlit as st
from dotenv import load_dotenv

# --- Load env (no secrets in code) ---
load_dotenv()  # reads .env if present

# Try both import styles for compatibility
try:
    from langchain_huggingface import (
        HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
    )
except ImportError:
    from langchain_huggingface.embeddings import HuggingFaceEmbeddings
    from langchain_huggingface.chat_models import ChatHuggingFace
    from langchain_huggingface.llms import HuggingFaceEndpoint

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from youtube_transcript_api._errors import NoTranscriptFound, VideoUnavailable
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel

# --------- Helpers ---------
def get_hf_token() -> str | None:
    # Prefer Streamlit secrets if running on Streamlit Cloud
    if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
        return st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    return os.getenv("HUGGINGFACEHUB_API_TOKEN")

def build_llm(repo_id: str = "openai/gpt-oss-20b") -> ChatHuggingFace:
    token = get_hf_token()
    if not token:
        st.error("Hugging Face token not found. Set HUGGINGFACEHUB_API_TOKEN in .env or Streamlit secrets.")
        st.stop()
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        task="text-generation",
        huggingfacehub_api_token=token,
        max_new_tokens=200,
        temperature=0.7,
    )
    return ChatHuggingFace(llm=llm)

def fetch_transcript_text(video_id: str) -> str:
    yt_api = YouTubeTranscriptApi()
    # Try English first; fall back to auto-captions if needed
    try:
        transcript_list = yt_api.fetch(video_id, languages=["en"])
    except NoTranscriptFound:
        transcript_list = yt_api.fetch(video_id, languages=["en-US", "en-GB", "ur", "hi"])
    # Elements are dicts: use ["text"]
    return " ".join(chunk["text"] for chunk in transcript_list)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --------- UI ---------
st.set_page_config(page_title="YouTube Video QA", page_icon="🎬", layout="wide")
st.title("🎥 YouTube Video Q&A with AI")
st.write("Enter a **YouTube Video ID**, watch the video, and ask questions about its transcript.")

with st.sidebar:
    st.header("⚙️ Settings")
    video_id = st.text_input("YouTube Video ID", placeholder="e.g. Gfr50f6ZBvo")
    process_btn = st.button("🔎 Process Video")

# Session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "model" not in st.session_state:
    st.session_state.model = None

# Process video
if process_btn and video_id:
    with st.spinner("📥 Fetching transcript..."):
        try:
            transcript = fetch_transcript_text(video_id)
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.create_documents([transcript])

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)
            st.session_state.model = build_llm()

            st.success("✅ Transcript processed successfully!")
            st.write(f"Total chunks created: **{len(chunks)}**")

            st.subheader("▶️ Watch Video")
            st.video(f"https://www.youtube.com/watch?v={video_id}")

        except TranscriptsDisabled:
            st.error("❌ No captions available for this video.")
        except (NoTranscriptFound, VideoUnavailable) as e:
            st.error(f"❌ Transcript not found or video unavailable: {e}")
        except Exception as e:
            st.error(f"⚠️ Error: {e}")

# Q&A
if st.session_state.vector_store and st.session_state.model:
    st.subheader("💬 Ask a Question about the Video")
    question = st.text_input("Your Question:", placeholder="e.g. What is the main topic of the video?")

    if st.button("🚀 Get Answer") and question:
        with st.spinner("🤔 Thinking..."):
            retriever = st.session_state.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 3, "lambda_mult": 0.2}
            )

            aug_chain = RunnableParallel({
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            })

            prompt_tmpl = PromptTemplate(
                template=(
                    "You are a helpful assistant.\n"
                    "Answer ONLY from the provided transcript context.\n"
                    "If context is insufficient, reply: \"I don't know about this\".\n\n"
                    "context: {context}\n"
                    "question: {question}\n"
                ),
                input_variables=["context", "question"]
            )

            parser = StrOutputParser()
            chain = aug_chain | prompt_tmpl | st.session_state.model | parser

            try:
                answer = chain.invoke(question)
                st.success("✅ Answer:")
                st.write(answer)
            except Exception as e:
                st.error(f"⚠️ Error while generating answer: {e}")
