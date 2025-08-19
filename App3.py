import streamlit as st
import os
import asyncio
from dotenv import load_dotenv

# LangChain imports - Updated for Google AI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI # <-- CHANGED
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever

asyncio.set_event_loop(asyncio.new_event_loop())

# --- st.set_page_config() 
st.set_page_config(page_title="CV Chatbot with Google AI", layout="wide") # <-- CHANGED

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)


# --- Google AI Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Please set the GOOGLE_API_KEY environment variable in the .env file.")
    st.stop()

# --- Global Components (initialized once) ---
@st.cache_resource
def get_llm():
    """Initializes and caches the Google AI LLM."""
    # CHANGED: Using ChatGoogleGenerativeAI instead of AzureChatOpenAI
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", # يمكنك اختيار نموذج آخر مثل "gemini-pro"
        google_api_key=GOOGLE_API_KEY,
        temperature=0.1,
        convert_system_message_to_human=True # بعض نماذج Gemini تتطلب هذا
    )

@st.cache_resource
def get_embeddings():
    """Initializes and caches the GoogleGenerativeAIEmbeddings."""
    # CHANGED: Using GoogleGenerativeAIEmbeddings instead of AzureOpenAIEmbeddings
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", # أحدث نموذج للتضمين من Google
        google_api_key=GOOGLE_API_KEY
    )

llm = get_llm()
embeddings = get_embeddings()

# --- Functions for CV Processing and QA (No changes needed in this section) ---

def process_cv_files(uploaded_files):
    """
    Loads, splits, embeds, and stores CV documents into a Chroma vector store.
    Each PDF is processed and chunked individually to maintain CV context
    and add metadata indicating the source CV file.
    """
    all_chunks = []
    temp_files_to_clean = []

    st.info(f"Processing {len(uploaded_files)} CVs...")

    for uploaded_file in uploaded_files:
        # Save file to a temporary location to be read by PyPDFLoader
        temp_file_path = f"./temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        temp_files_to_clean.append(temp_file_path)
        
        loader = PyPDFLoader(temp_file_path)
        
        pages_from_this_cv = loader.load()
        
        if not pages_from_this_cv:
            st.warning(f"No content found in {uploaded_file.name}. Skipping.")
            continue

        # Add metadata (cv_name and source) to each page before splitting
        for page in pages_from_this_cv:
            page.metadata['cv_name'] = uploaded_file.name
            page.metadata['source'] = uploaded_file.name

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True,
        )
        chunks_for_this_cv = text_splitter.split_documents(pages_from_this_cv)
        all_chunks.extend(chunks_for_this_cv)

    # Clean up temporary files
    for file_path in temp_files_to_clean:
        if os.path.exists(file_path):
            os.remove(file_path)

    if not all_chunks:
        st.warning("No documents loaded or processed from the uploaded files.")
        return None

    st.info(f"Total chunks processed from all CVs: {len(all_chunks)}.")

    vector_store = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        collection_name="cv_documents"
    )
    return vector_store

def get_qa_chain(vector_store, chat_history_for_llm=""):
    """
    Creates and returns a RetrievalQA chain using a MultiQueryRetriever for better context.
    """
    if not vector_store:
        return None

    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=vector_store.as_retriever(search_kwargs={"k": 15}),
        llm=llm
    )

    document_prompt = ChatPromptTemplate.from_template(
        "--- CV Snippet from: {cv_name} ---\n{page_content}\n"
    )

    # --- START: CORRECTED AND FINAL SYSTEM TEMPLATE ---
    system_template = f"""
                        You are an AI HR Analyst with a specialized cognitive module for deep data extraction. Your primary directive is to provide clean, professional, and highly accurate answers based ONLY on the provided CV context.

                        --- CORE DIRECTIVES ---
                        1.  **Strict Context Adherence:** You MUST ONLY use the information explicitly stated in the provided CV context. Do not make up or infer information *from the CVs*.
                        2.  **Intelligent Query Interpretation (For User's Question ONLY):**
                            *   You MUST be tolerant of user typos, abbreviations, and common synonyms to understand their *intent*.
                            *   **Example (Skills):** 'HTM' must match 'HTML'. 'ML' must match 'Machine Learning'.
                            *   **Example (Names):** 'Johan' should match 'John'. 'Hasan' should match 'Hassan'.
                        3.  **Strict "AND" Logic (Default Behavior):**
                            *   For all queries that are NOT comparisons, you must treat multiple requirements as a strict logical AND. Return only candidates who explicitly meet ALL specified requirements.

                        --- USER INTENT SCENARIOS ---

                        *   **IF the user explicitly asks to "compare" two or more named individuals...**
                            *   **Your Action:** This is a special scenario. Your primary goal is to find the named individuals first (tolerating minor typos in their names).
                            *   **Override Default Logic:** For this task, you will temporarily ignore the strict "AND" logic regarding the job role.
                            *   **Generate a Comparison Table:** After finding the individuals, create a "Detailed Comparison" table. In the table, you MUST evaluate each person against the requested criteria (e.g., "Suitability for AI Engineer").
                            *   **Acknowledge Mismatches:** It is expected that one candidate may be a poor fit. Your job is to show *why* they are a good or bad fit, based on the evidence in their CV.

                        *   **IF the user asks to "rank all," "list all," or "who are the candidates for"...**
                            *   **Your Action:** Find every relevant candidate (who meets ALL criteria as per the default "AND" logic) and present them as a **numbered "Suitability Ranking" list**, starting from 1., from best to worst.

                        --- OUTPUT FORMATS ---
                        *   **For Ranking/Single Candidate:** Use the numbered format starting with "1.".
                            **1. [Candidate's Full Name] - Suitability Score: [Score]/10**
                            *   **Justification:** [A concise, evidence-based summary.]
                            **Contact Information**
                            *   **Email:** [email@example.com or N/A]
                            *   **Phone:** [+123456789 or N/A]

                        *   **For Comparisons:** Use a markdown table. Example:
                            **Detailed Comparison: [Person A] vs. [Person B] for [Job Role]**
                            | Criteria                  | [Person A's Name]                                       | [Person B's Name]                                       |
                            | ------------------------- | ------------------------------------------------------- | ------------------------------------------------------- |
                            | **Overall Suitability**   | [e.g., High / Medium / Low]                             | [e.g., High / Medium / Low]                             |
                            | **Relevant Experience**   | [Summary of their experience relevant to the role]      | [Summary of their experience, or 'N/A' if none]         |

                        <Previous_Conversation_History>
                        {chat_history_for_llm}
                        </Previous_Conversation_History>

                        **Context from CVs:**
                        {{context}}
                 """
    # --- END: CORRECTED AND FINAL SYSTEM TEMPLATE ---
    
    human_template = "{question}"
    
    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template),
    ])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever_from_llm,
        return_source_documents=False,
        chain_type_kwargs={
            "prompt": chat_prompt,
            "document_prompt": document_prompt,
            "document_separator": "\n\n==== NEXT CV SNIPPET ====\n\n"
        }
    )
    return qa_chain
    

# --- Streamlit UI (Main Content) ---
st.markdown("<h1 style='text-align: center; font-size: 48px;'>🤖 CHATBOT_WITH_YOUR_CVs</h1>", unsafe_allow_html=True)
# CHANGED: Updated text to reflect Google AI
st.markdown("<h3 style='text-align: center;'>Upload multiple CVs (PDFs), process them, and then ask questions about their content.</h2>", unsafe_allow_html=True)
# Initialize session state variables
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "display_chat_history" not in st.session_state:
    st.session_state.display_chat_history = []
if "llm_chat_history" not in st.session_state:
    st.session_state.llm_chat_history = []
    
# --- CV Upload and Processing Section ---
st.header("1. Upload & Process CVs")
uploaded_files = st.file_uploader(
    "Upload multiple CVs (PDF files only)",
    type=["pdf"],
    accept_multiple_files=True,
    key="cv_uploader",
    help="Select one or more PDF CV files to upload."
)

if st.button("Process CVs", key="process_button"):
    if uploaded_files:
        with st.spinner("Processing CVs... This may take a moment."):
            st.session_state.vector_store = process_cv_files(uploaded_files)
            st.session_state.display_chat_history = [] 
            st.session_state.llm_chat_history = []
        if st.session_state.vector_store:
            st.success("CVs processed and ready for questions! You can now start asking questions below.")
            st.rerun()
        else:
            st.error("Failed to process CVs. Please check the uploaded files.")
    else:
        st.warning("Please upload at least one PDF CV to process.")

# --- Chat Interface Section ---
st.header("2. Ask Questions")

GREETINGS = {"hello", "hi", "good morning", "good afternoon", "good evening", "hey", "hola"}
GREETING_RESPONSE = "Hello! I am an AI assistant specialized in filtering and answering questions about Resumes (CVs). Please upload CVs and ask me questions about their content."

for chat_entry in st.session_state.display_chat_history:
    with st.chat_message("user"):
        st.markdown(f"<span style='color:#007bff;'>{chat_entry['question']}</span>", unsafe_allow_html=True) 
    with st.chat_message("assistant"):
        st.markdown(f"<span style='color:#28a745;'>{chat_entry['answer']}</span>", unsafe_allow_html=True) 

prompt = st.chat_input(
    "Enter your question about the CVs:",
    disabled=not st.session_state.vector_store
)

if prompt:
    with st.chat_message("user"):
        st.markdown(f"<span style='color:#007bff;'>{prompt}</span>", unsafe_allow_html=True)

    normalized_prompt = prompt.strip().lower()

    if normalized_prompt in GREETINGS:
        answer = GREETING_RESPONSE
        with st.chat_message("assistant"):
            st.markdown(f"<span style='color:#28a745;'>{answer}</span>", unsafe_allow_html=True)
        st.session_state.display_chat_history.append({"question": prompt, "answer": answer})
    else:
        history_for_llm_prompt = ""
        if st.session_state.llm_chat_history:
            history_for_llm_prompt = "Previous conversation:\n"
            for chat_turn in st.session_state.llm_chat_history[-5:]:
                history_for_llm_prompt += f"User: {chat_turn['question']}\n"
                history_for_llm_prompt += f"AI: {chat_turn['answer']}\n"
            history_for_llm_prompt += "\n"

        qa_chain = get_qa_chain(st.session_state.vector_store, chat_history_for_llm=history_for_llm_prompt)
        
        if qa_chain:
            with st.chat_message("assistant"):
                with st.spinner("Searching and generating answer..."):
                    try:
                        result = qa_chain.invoke({"query": prompt})
                        answer = result.get("result", "No answer found.")
                        
                        st.markdown(f"<span style='color:#28a745;'>{answer}</span>", unsafe_allow_html=True)
                        st.session_state.display_chat_history.append({"question": prompt, "answer": answer})
                        st.session_state.llm_chat_history.append({"question": prompt, "answer": answer})

                    except Exception as e:
                        error_message = f"An error occurred while getting the answer: {e}"
                        st.error(error_message)
                        st.session_state.display_chat_history.append({"question": prompt, "answer": error_message})
        else:
            with st.chat_message("assistant"):
                warning_message = "QA chain not initialized. Please re-process CVs."
                st.warning(warning_message)
                st.session_state.display_chat_history.append({"question": prompt, "answer": warning_message})

else:
    if not st.session_state.vector_store:
        st.info("Please upload and process CVs first to enable the chat interface.")

# --- Reset Button ---
if st.button("Reset Chatbot", key="reset_button"):
    st.session_state.vector_store = None
    st.session_state.display_chat_history = []
    st.session_state.llm_chat_history = []
    st.success("Chatbot reset. Please upload new CVs to begin.")
    st.rerun()