import os
import streamlit as st
import google.generativeai as genai
import fitz  # PyMuPDF
import time
from textwrap import shorten
import google.generativeai as genai
print(dir(genai))


# --- Configuration -----------------------------------------------------------
st.set_page_config(page_title="AI Notes Assistant Agent", page_icon="📚")

# --- API Key Setup -----------------------------------------------------------
try:
    # Get the key from Streamlit secrets or environment variables
    api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
except Exception as e:
    st.error("Google API key not found or invalid. Please set it in your environment or Streamlit secrets.")
    st.stop()

# --- Model and App Settings --------------------------------------------------
MODEL_NAME = "gemini-1.5-flash-latest" # Use a fast and capable model
MAX_CHARS = 6000
CHUNK_SIZE = 2000


# --- Core Functions (Rewritten for Gemini) -----------------------------------
def call_gemini(prompt: str, retries=2):
    """Generic function to call the Gemini API with error handling."""
    model = genai.GenerativeModel(model_name=MODEL_NAME)
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except genai.types.generation_types.BlockedPromptException:
             st.error("The request was blocked due to safety concerns. Try rephrasing.")
             return "Request blocked."
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e): # Handle quota errors
                 st.error(f"Quota exceeded for Google API. Error: {e}")
                 return "Could not get an answer due to a quota error."
            if attempt < retries - 1:
                time.sleep(2)
                #time.sleep(2 ** attempt)  Exponential backoff
            else:
                st.error(f"An API error occurred after several retries: {e}")
                return "An unexpected API error occurred."
    return "Failed to get a response from the API."


@st.cache_data(show_spinner=False)
def summarise_long_text(text: str) -> str:
    """Summarises text using Gemini, handling long inputs by chunking."""
    chunks = [text[i : i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    partials = []

    for i, chunk in enumerate(chunks):
        with st.spinner(f"Summarising chunk {i+1}/{len(chunks)}…"):
            prompt = f"Summarise the following lecture notes in 3-5 concise bullet points:\n\n{chunk}"
            summary = call_gemini(prompt)
            partials.append(summary)

    combined_summaries = "\n".join(partials)
    final_prompt = f"Combine and condense these bullet-point summaries into a single, clear summary:\n\n{combined_summaries}"
    final_summary = call_gemini(final_prompt)
    return final_summary

def ask_question(context: str, question: str) -> str:
    """Answers a question using the provided context with Gemini."""
    prompt = f"Using ONLY the information in the following notes, answer the question clearly.\n\nNOTES:\n{context}\n\nQUESTION: {question}"
    return call_gemini(prompt)

def simplify_passage(passage: str) -> str:
    """Simplifies a passage using Gemini."""
    prompt = f"You rewrite text so that a first-year college student can understand it. Simplify the following passage:\n\n{passage}"
    return call_gemini(prompt)

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extracts text from a PDF file."""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return "".join(p.get_text() for p in doc)

# --- Streamlit UI ------------------------------------------------------------
st.title("📚 AI Notes Assistant Agent")

uploaded = st.file_uploader(
    "Upload your lecture notes (PDF or plain-text file)", type=["pdf", "txt"]
)

if uploaded:
    if uploaded.type == "application/pdf":
        raw_text = extract_text_from_pdf(uploaded.read())
    else:
        raw_text = uploaded.read().decode("utf-8", errors="ignore")

    if len(raw_text) > MAX_CHARS:
        st.warning(f"File is large. Only the first {MAX_CHARS:,} characters will be processed.")
        raw_text = raw_text[:MAX_CHARS]

    st.success("File uploaded. You can now use the features below.")

    # ---- SUMMARY ----
    if st.button("Generate Summary"):
        summary = summarise_long_text(raw_text)
        st.subheader("📝 Summary")
        st.markdown(summary)

    st.divider()

    # ---- QUESTION ANSWERING ----
    st.subheader("❓ Ask a Question about these Notes")
    question = st.text_input("Type your question here and press Enter")
    if question:
        with st.spinner("Thinking…"):
            answer = ask_question(raw_text, question)
        st.markdown(f"**Answer:** {answer}")

    st.divider()

    # ---- SIMPLIFY PASSAGE ----
    st.subheader("🔄 Simplify a Difficult Passage")
    passage = st.text_area("Paste a paragraph to rewrite in simpler words", height=150)
    if passage:
        with st.spinner("Rewriting…"):
            simpler = simplify_passage(passage)
        st.markdown("**Simplified version:**")
        st.markdown(simpler)
else:
    st.info("⬆️ Start by uploading a PDF or TXT file.")

st.caption("Powered by Google Gemini. Remember to set your `GOOGLE_API_KEY`.")