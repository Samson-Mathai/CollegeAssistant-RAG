import streamlit as st
import os
import tempfile
import time
import pymupdf4llm
from langchain_text_splitters import MarkdownTextSplitter
from langchain_core.documents import Document
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pymongo import MongoClient
import pdfplumber
import re as re_module

# --- Secret Keys Management ---
import sys

# 2. Try hardcoded local keys
hardcoded_mongo = ""
hardcoded_gemini = ""
try:
    import key_param
    hardcoded_mongo = key_param.MONGODB_URI
    hardcoded_gemini = key_param.GEMINI_API_KEY
except ImportError:
    pass

# 3. Try Streamlit Cloud Vault
try:
    hardcoded_mongo = hardcoded_mongo or st.secrets.get("MONGODB_URI", "")
    hardcoded_gemini = hardcoded_gemini or st.secrets.get("GEMINI_API_KEY", "")
except Exception:
    pass

# Lock the Database Connection globally
MONGODB_URI = hardcoded_mongo
GEMINI_API_KEY = hardcoded_gemini

if not MONGODB_URI or not GEMINI_API_KEY:
    st.error("🚨 Master Keys are missing from the Cloud Vault! The Database is locked.")
    st.stop()

#  UI Configuration 
st.set_page_config(page_title="Campus Affairs Navigator", layout="wide")
st.title("Campus Affairs Navigator")
st.markdown("Upload multiple PDFs (like Timetables, HELB Guidelines, Student Handbooks) and ask questions! The AI will answer and cite exactly which document the answer came from.")

import re

# --- Google OAuth Authentication ---
if not st.user.is_logged_in:
    st.markdown("Sign in with your Google account to access the AI assistant.")
    if st.button("🔐 Log in with Your School Email"):
        st.login("google")
    st.stop()

# User is authenticated 
user_email = st.user.email
user_name = st.user.name or user_email
ADMIN_EMAILS = ["samsonmathai77@gmail.com","mathaisamson6@gmail.com"]
role = "ADMIN" if user_email in ADMIN_EMAILS else "STUDENT"
active_user_id = "GLOBAL" if role == "ADMIN" else user_email

# Sidebar: User Info & FAQs
with st.sidebar:
    st.header("Campus Portal")
    
    if role == "ADMIN":
        st.success(f"Admin - {user_name}")
        st.caption("Documents you upload will be visible to everyone.")
    else:
        st.success(f"Student - {user_name}")
        st.caption(f"Logged in as {user_email}")
    
    if st.button("Log out"):
        st.logout()
    
    st.divider()
    
    st.header("Frequently Asked Questions")
    with st.expander("When are exams commencing?"):
        st.write("Upload your university's academic calendar or exam timetable, then type your question below. The AI will find the exact dates for you.")
    with st.expander("Why can't I print my examination card?"):
        st.write("This is usually due to pending fee balances or missing unit registrations. Ask the AI to check your student handbook for the exact clearance requirements.")
    with st.expander("Summarise my exam timetable"):
        st.write("Upload your PDF timetable and simply ask 'Summarise my exam timetable' below. The AI will extract all your specific units, dates, times, and venues into a neat list.")

#Database and AI Setup
@st.cache_resource(show_spinner=False)
def get_mongo_client():
    return MongoClient(MONGODB_URI)

# --- MongoDB Setup ---
client = get_mongo_client()
collection = client["book_mongodb_chunks"]["chunked_data"]

# --- AI Models Setup ---
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2", google_api_key=GEMINI_API_KEY)
vector_store = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embeddings,
    index_name="vector_index"
)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY, temperature=0)

# --- Timetable Detection & Extraction Helpers ---
def is_timetable_file(filename):
    """Returns True if the filename suggests it is a timetable/schedule document."""
    keywords = ["timetable", "time table", "schedule", "exam", "examination",
                "class", "lecture", "session", "semester", "calendar"]
    name_lower = filename.lower()
    return any(kw in name_lower for kw in keywords)


def load_timetable_pdf(tmp_file_path, filename):
    """Extract tables from a timetable PDF using pdfplumber and convert rows to
    natural-language sentences so the AI can answer scheduling queries accurately."""
    all_docs = []

    # --- Column-header mapping (flexible) ---
    DAY_KEYS = ["day"]
    TIME_KEYS = ["time", "period"]
    CODE_KEYS = ["unit code", "course code", "code"]
    NAME_KEYS = ["unit name", "course name", "subject"]
    LECTURER_KEYS = ["lecturer", "instructor"]
    VENUE_KEYS = ["venue", "room", "location"]

    def _match(header, candidates):
        h = header.strip().lower()
        return any(h == c for c in candidates)

    with pdfplumber.open(tmp_file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            for table in tables:
                if not table or len(table) < 2:
                    continue

                headers = [str(h).strip() if h else "" for h in table[0]]

                # Build index maps
                day_idx = next((i for i, h in enumerate(headers) if _match(h, DAY_KEYS)), None)
                time_idx = next((i for i, h in enumerate(headers) if _match(h, TIME_KEYS)), None)
                code_idx = next((i for i, h in enumerate(headers) if _match(h, CODE_KEYS)), None)
                name_idx = next((i for i, h in enumerate(headers) if _match(h, NAME_KEYS)), None)
                lect_idx = next((i for i, h in enumerate(headers) if _match(h, LECTURER_KEYS)), None)
                venue_idx = next((i for i, h in enumerate(headers) if _match(h, VENUE_KEYS)), None)

                mapped_indices = {day_idx, time_idx, code_idx, name_idx, lect_idx, venue_idx} - {None}

                for row in table[1:]:
                    cells = [str(c).strip() if c else "" for c in row]
                    if all(cell == "" for cell in cells):
                        continue

                    day = cells[day_idx] if day_idx is not None else ""
                    time_val = cells[time_idx] if time_idx is not None else ""
                    code = cells[code_idx] if code_idx is not None else ""
                    name = cells[name_idx] if name_idx is not None else ""
                    lecturer = cells[lect_idx] if lect_idx is not None else ""
                    venue = cells[venue_idx] if venue_idx is not None else ""

                    sentence = f"On {day} from {time_val}, {code} ({name}) is taught by {lecturer} in {venue}."

                    # Append any extra (unmapped) columns
                    extras = []
                    for idx, h in enumerate(headers):
                        if idx not in mapped_indices and cells[idx]:
                            extras.append(f"{h}: {cells[idx]}")
                    if extras:
                        sentence += " " + " ".join(extras)

                    doc = Document(page_content=sentence)
                    doc.metadata["hasCode"] = False
                    doc.metadata["source"] = filename
                    all_docs.append(doc)

            # Also capture any plain text on the page
            plain_text = page.extract_text()
            if plain_text:
                text_splitter = MarkdownTextSplitter(
                    chunk_size=1500,
                    chunk_overlap=200
                )
                text_doc = Document(page_content=plain_text)
                text_splits = text_splitter.split_documents([text_doc])
                for split in text_splits:
                    split.metadata["hasCode"] = False
                    split.metadata["source"] = filename
                all_docs.extend(text_splits)

    return all_docs


#  Document Ingestion 
with st.expander("Upload Campus Documents (Bulk Upload Supported)", expanded=True):
    st.info("You can drag and drop multiple PDFs here at the same time!")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        for f in uploaded_files:
            if is_timetable_file(f.name):
                st.caption(f"🗓️ {f.name} → Timetable (table-aware extraction)")
            else:
                st.caption(f"📃 {f.name} → Standard document")
        if st.button("Process Documents"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_files = len(uploaded_files)
            
            for i, uploaded_file in enumerate(uploaded_files):
                filename = uploaded_file.name
                status_text.text(f"Checking {filename} ({i+1}/{total_files})...")
                
                # Deduplication Check: Check if this file is already in MongoDB
                existing_doc = collection.find_one({"source": filename})
                if existing_doc:
                    st.warning(f"⏩ Skipped '{filename}'. It is already in your database!")
                    progress_bar.progress((i + 1) / total_files)
                    continue # Skip to the next file!
                
                status_text.text(f"Processing {filename} ({i+1}/{total_files})...")
                
                # Save the uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                try:
                    if is_timetable_file(filename):
                        status_text.text(f"🗓️ Timetable detected — using table-aware extraction for {filename}...")
                        splits = load_timetable_pdf(tmp_file_path, filename)
                    else:
                        # 1. Convert PDF to Markdown (Preserves Tables perfectly!)
                        md_text = pymupdf4llm.to_markdown(tmp_file_path)
                        
                        # Wrap the text in a Document object so LangChain can process it
                        doc = Document(page_content=md_text)
                        
                        # 2. Split the Markdown without destroying tables
                        text_splitter = MarkdownTextSplitter(
                            chunk_size=1500, # Increased chunk size to keep massive tables together
                            chunk_overlap=200
                        )
                        splits = text_splitter.split_documents([doc])
                        
                        # 3. Add metadata (CRUCIAL FOR SOURCE TRACKING)
                        for split in splits:
                            split.metadata["hasCode"] = False
                            split.metadata["source"] = filename # Track exactly which PDF this chunk came from
                    
                    if len(splits) == 0:
                        st.error(f"No text found in {filename}. It might be a scanned image.")
                        continue
                        
                    # 4. Upload to MongoDB in smaller batches to avoid rate limits
                    import time
                    batch_size = 5
                    for batch_start in range(0, len(splits), batch_size):
                        batch = splits[batch_start:batch_start+batch_size]
                        status_text.text(f"Uploading batch {batch_start//batch_size + 1} of {(len(splits)-1)//batch_size + 1} for {filename}...")
                        try:
                            MongoDBAtlasVectorSearch.from_documents(
                                documents=batch,
                                embedding=embeddings,
                                collection=collection,
                                index_name="vector_index"
                            )
                            time.sleep(8) # Much longer sleep to respect Free Tier API rate limits!
                        except Exception as e:
                            status_text.text(f"Rate limit hit! Cooling down for 30 seconds...")
                            time.sleep(30)
                            MongoDBAtlasVectorSearch.from_documents(batch, embeddings, collection=collection, index_name="vector_index")
                            
                    st.success(f"✅ Successfully embedded {filename} into MongoDB.")
                    
                except Exception as e:
                    import traceback
                    st.error(f"❌ Error processing {filename}:\n\n{traceback.format_exc()}")
                    st.stop() # Stop the loop so they can fix the error/API key
                finally:
                    os.remove(tmp_file_path)
                
                progress_bar.progress((i + 1) / total_files)
                
            status_text.text("All documents processed successfully!")

# --- Main Chat Interface ---
st.divider()
st.subheader("Ask Anything About Your Uploaded Documents")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("E.g. How do I appeal my HELB Band categorization?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching through your documents..."):

            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 5,
                    "pre_filter": {"hasCode": {"$eq": False}}
                }
            )

            template = """
You are an expert assistant helping Kenyan university students navigate 
academic and funding documents.

RULE 1 — SOURCE GROUNDING:
Answer using ONLY the provided context. Cite the source ONCE at the 
very bottom of your response as: Source: [filename]
If the context does not contain the answer, begin with:
"⚠️ Note: This information is based on general knowledge, 
not your uploaded documents."

RULE 2 — TIMETABLE FORMAT (apply ONLY when the question asks for a 
timetable, schedule, or class list):

❌ DO NOT use bullet points
❌ DO NOT organise by subject name  
❌ DO NOT repeat the source on every entry
❌ DO NOT add text like "closest match for..."
❌ DO NOT list Day / Time / Venue / Lecturer as separate lines

✅ Organise by DAY, chronologically (Monday → Friday)
✅ Use a Markdown table under each day heading
✅ Sort entries within each day by start time
✅ Cite the source ONCE at the very bottom only
✅ Deduplicate — if the same class appears twice in context, list it once

USE EXACTLY THIS FORMAT for timetable responses:

📅 **MONDAY, 22 JUNE 2026**
| Time          | Venue   | Subject                           | Lecturer              |
|---------------|---------|-----------------------------------|-----------------------|
| 10:00 – 12:00 | R2      | Personal Computer Software Support| Mr. Mwangi / Dennis   |

📅 **TUESDAY, 23 JUNE 2026**
| Time          | Venue   | Subject                           | Lecturer              |
|---------------|---------|-----------------------------------|-----------------------|
| 12:30 – 14:30 | ATS A   | IT Virtualisation                 | Mr. Wesley / Harrison |
| 15:00 – 17:00 | ICT LAB | Info Systems Analysis & Design    | Mr. Wesley / Harrison |

Source: ICT-TIMETABLE.pdf

RULE 3 — EXAM CLASHES:
Check every retrieved entry carefully before concluding whether 
a clash exists. Never fabricate timetable data.

CONTEXT:
{context}

QUESTION: {question}

OUTPUT:"""

            custom_rag_prompt = PromptTemplate.from_template(template)

            # FIXED: Source appears ONCE in context, not on every chunk
            def format_docs(docs):
                sources = set()
                chunks = []

                for doc in docs:
                    source_name = doc.metadata.get('source', 'Unknown Document')
                    sources.add(source_name)
                    # No source tag on individual chunks anymore
                    chunks.append(doc.page_content)

                context_text = "\n\n".join(chunks)

                # Append all unique sources once at the bottom of context
                if sources:
                    source_list = ", ".join(sorted(sources))
                    context_text += f"\n\nAvailable sources: {source_list}"

                return context_text

            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | custom_rag_prompt
                | llm
                | StrOutputParser()
            )

            try:
                response = rag_chain.invoke(prompt)
                st.markdown(response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
            except Exception as e:
                st.error(
                    f"An error occurred: {e}. "
                    f"If you ran out of tokens, swap your API key in the sidebar!"
                )


