import streamlit as st
import os
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# --- PLACE st.set_page_config() HERE (MUST BE THE ABSOLUTE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="CV Chatbot with Azure GPT-4.0-mini", layout="wide")

# Load environment variables from .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)


# --- Azure OpenAI Configuration ---
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY" )
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION") # Default if not specified
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME" )

if not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_CHAT_DEPLOYMENT_NAME, AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME]):
    st.error("Please set all required Azure OpenAI environment variables in the .env file.")
    st.stop()

# --- Global Components (initialized once) ---
@st.cache_resource
def get_llm():
    """Initializes and caches the AzureChatOpenAI LLM."""
    return AzureChatOpenAI(
        openai_api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_key=AZURE_OPENAI_API_KEY,
        azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
        temperature=0.1 
    )

@st.cache_resource
def get_embeddings():
    """Initializes and caches the AzureOpenAIEmbeddings."""
    return AzureOpenAIEmbeddings(
        openai_api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_key=AZURE_OPENAI_API_KEY,
        azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME
    )

llm = get_llm()
embeddings = get_embeddings()

# --- Functions for CV Processing and QA ---

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
        
        st.info(f"Loading and processing: {uploaded_file.name}")
        loader = PyPDFLoader(temp_file_path)
        
        pages_from_this_cv = loader.load()
        
        if not pages_from_this_cv:
            st.warning(f"No content found in {uploaded_file.name}. Skipping.")
            continue

        # Add metadata (cv_name and source) to each page before splitting
        for page in pages_from_this_cv:
            # Ensure cv_name and source are set correctly for each page.
            # This metadata will be propagated to chunks.
            page.metadata['cv_name'] = uploaded_file.name  # Unique identifier for each CV
            page.metadata['source'] = uploaded_file.name    # Also store as 'source' for LangChain conventions

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True,
        )
        chunks_for_this_cv = text_splitter.split_documents(pages_from_this_cv)
        all_chunks.extend(chunks_for_this_cv)
        
    #    st.info(f"Processed '{uploaded_file.name}' into {len(chunks_for_this_cv)} chunks.")

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
    Creates and returns a RetrievalQA chain with a dynamic prompt including chat history.
    `chat_history_for_llm` is a formatted string of previous conversation turns.
    """
    if not vector_store:
        return None

    # Define how each retrieved document (chunk) should be formatted in the prompt context.
    # This explicitly includes the 'cv_name' metadata within the text seen by the LLM.
    document_prompt = ChatPromptTemplate.from_template(
        "--- CV Name: {cv_name} ---\n{page_content}\n"
    )

    system_template = f"""
                You are an expert AI assistant specialized *ONLY* in extracting information and answering questions about Resumes (CVs).
                Your goal is to extract specific details and summarize information from the provided CVs.

                Your strict instructions are:
                1.  **Information Source:** Only use the context provided from the CVs to answer questions. Each retrieved document chunk is clearly delineated by "--- CV Name: [filename] ---" at its beginning. This [filename] is the unique identifier for each CV.
                2.  **No Fabrication:** Do not make up any information. If you cannot find the exact answer within the provided CV context, clearly state: 'The requested information is not available in the provided data.'
                3.  **Concise and Clear Answers:** Provide the shortest and clearest answer that directly addresses the question.
                4.  **Strict Individual Focus and No Mixing:** Understand that all sections of a CV (education, experience, contact, skills, projects, etc.) pertain exclusively to the *same individual*.
                    *   When extracting details for a *specific named individual* (e.g., "John Doe's experience"), you MUST extract information *only* from the document sections clearly marked with that person's CV name (e.g., '--- CV Name: JohnDoe.pdf ---').
                    *   **CRITICAL:** Do NOT mix or conflate information, experiences, or skills from different CVs. For example, if you list skills for 'John Doe', ensure those skills are *only* from the document sections belonging to 'John Doe's' CV, even if similar skills appear in other CVs.
                    *   When answering follow-up questions, always maintain context about the person previously discussed, linking related information across different parts of *their* CV only.
                5.  **Understanding Query Intent & Flexibility for Roles:**
                    *   **Role Validation and Response Hierarchy:** When a user asks about a job title or role:
                        *   **a) Invented or Nonsensical Roles:** If the requested title (e.g., "Chief Coffee Maker of the Universe") or a valid title with clearly nonsensical or invalidating additions (e.g., "Software Engineer with magical powers") is clearly not a standard, recognized professional role, you MUST respond in English:
                            "**No, it doesn't exist.** The requested job title is not a recognized professional role in the provided data, and therefore no relevant information can be extracted."
                            *   **Then, you MUST immediately ask the user after this answer, in English:** "If you want to know about roles similar to [the most likely standard role inferred from the user's input, e.g., 'Software Engineer'] in the CVs, please answer **Yes**."
                        *   **b) Standard Role - Direct Query (Exact/Close Match):** If the query is for a standard, recognized professional role without explicit ranking or suitability implied (e.g., "Who is the 'Project Manager'?", "What experience does the 'Data Scientist' have?"), look for direct matches or highly similar job titles in the CVs. Be tolerant of minor spelling variations and 
                               common synonyms (e.g., "engineer", "fullstack" vs. "full stack", "ML" vs. "Machine Learning"). If no direct or highly similar match is found in the provided CV context, state: 'The requested information is not available in the provided data.'
                        *   **c) Suitability/Matching Queries & Ranking:** If a user asks for a person "suitable for," "closest to," "best for," "find someone who can do X," "recommend someone for X role," or explicitly asks to 'rank' or 'sort' individuals for a role/skills (e.g., "Find me someone suitable for a Machine Learning Engineer role", "Who is the most suitable Full-Stack Developer?")
                            *   **Identify & Rank:** Interpret this as a request to identify and, if ranking is implied, sort individuals whose skills, experience, or previous roles align with the requested function.
                            *   **Output Format:** Provide a ranked list from most suitable to least. For each individual, include their full name, a subjective suitability score (e.g., on a scale of 1-10 or a percentage), and a brief justification of why they are suitable based on their CV. (e.g., "John Doe - Score: 9/10 - Extensive experience in Python..."). Do NOT include file names here.
                            *   **No Subjective Recommendations (unless ranked):** Other than the requested ranking, do not provide subjective "recommendations" or simple rankings without justification/score. Simply identify those who match based on the provided CV content.
                            *   If no individuals are suitable, state that clearly.
                        *   **d) Handling "Yes" Response for Similar Roles:** If the user's direct next response is "Yes", and your last answer ended with "No, it doesn't exist." and the prompt for similar roles, then interpret "Yes" as a request to identify and display individuals with roles *similar* to the core job title extracted from the user's original query.
                            *   **In this case, you MUST search for similar roles and present them in a list:** List the person's full name and their most relevant job title from their CV. Do NOT include file names here.
                            *   **Required Format for "Yes" response:**
                                "Candidates for Similar Roles:
                                - [Person's Full Name 1] - [Their Job Title in CV 1]
                                - [Person's Full Name 2] - [Their Job Title in CV 2]
                                ..."
                            *   If no similar roles are found, state: "No similar roles found for this title in the available CVs."

                6.  **Specific Information Extraction:** You are expected to extract details about any CV section, such as: education history, work experience, skills (technical, soft, languages), contact information (email, phone, LinkedIn), projects, certifications, awards, personal details relevant to a CV, etc.
                7.  **Identifying Multiple Individuals (Crucial for Skills):** If a query asks to identify individuals for a role, skill, or other criteria (e.g., "Who are the Software Engineers?", "List all candidates with Python skills"), you MUST list *all* individuals found in the provided CVs that match the criteria.
                    *   **Output Format:** For each identified person, list their full name followed by the specific skill(s) or relevant detail(s) found in *their* CV. Present each person's information on a new line or as a distinct bullet point.
                    *   **Example for skills:**
                        "Persons with Python skills:
                        - John Doe: Advanced Python, Django, Flask, Pandas
                        - Jane Smith: Python scripting, Data analysis with Python (NumPy, SciPy)
                        - Ali Khan: Basic Python for web development
                        "
                    *   Do NOT include file names here. If no individuals match, state: 'No individuals matching this criteria were found in the provided data.'
                8.  **Comparisons:** If a user requests a comparison between two or more individuals based on specific CV details (e.g., "Compare John's experience with Jane's education", "Show me the skills of both Ali and Omar"), you MUST present the comparison in a clear and concise table format. The table should have 
                      columns for the specific detail(s) being compared (e.g., "Experience", "Education", "Skills") and rows for each person. Use markdown table syntax. If information for a specific comparison point is not available for a person, indicate 'N/A' or 'Not available' in the table cell.
                9.  **Continuous Conversation:** You are capable of answering multiple questions simultaneously if they are all relevant to the available data.
                10. **Off-Topic Questions:** If any questions unrelated to CV filtering or their content are asked (e.g., general inquiries, personal questions), respond by stating that you are an assistant specialized ONLY in 'CV filtering' and cannot answer such questions. (Simple greetings are handled separately.)

                <Previous_Conversation_History>
                {chat_history_for_llm}
                </Previous_Conversation_History>

                **Context:** {{context}}
                """
    human_template = "{question}"
    
    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template),
    ])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # 'stuff' puts all retrieved docs into the prompt
        retriever=vector_store.as_retriever(search_kwargs={"k": 7}), # Increased k to retrieve more potential candidates
        return_source_documents=False, # This ensures raw source documents are not returned by LangChain
        chain_type_kwargs={
            "prompt": chat_prompt,
            "document_prompt": document_prompt, # <--- IMPORTANT ADDITION: Formats each retrieved document
            "document_separator": "\n\n==== NEXT CV SECTION ====\n\n" # <--- IMPORTANT ADDITION: Clear separator between documents
        }
    )
    return qa_chain

# --- Streamlit UI (Main Content) ---
st.title("📄 CV Filtering Chatbot")
st.markdown("Upload multiple CVs (PDFs), process them, and then ask questions about their content using Azure GPT-4.0-mini.")

# Initialize session state variables
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "display_chat_history" not in st.session_state: # This stores ALL chat for display
    st.session_state.display_chat_history = []
if "llm_chat_history" not in st.session_state: # This stores only relevant chat for LLM memory (last 5 turns)
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
            # Clear both histories on new CV processing
            st.session_state.display_chat_history = [] 
            st.session_state.llm_chat_history = [] # Important: clear LLM memory too
        if st.session_state.vector_store:
            st.success("CVs processed and ready for questions! You can now start asking questions below.")
            st.rerun() # Rerun to update the UI state and enable the chat input
        else:
            st.error("Failed to process CVs. Please check the uploaded files.")
    else:
        st.warning("Please upload at least one PDF CV to process.")

# --- Chat Interface Section ---
st.header("2. Ask Questions")

# Define greeting phrases and the fixed response (handled outside LLM for efficiency)
GREETINGS = {"hello", "hi", "good morning", "good afternoon", "good evening", "hey", "hola"}
GREETING_RESPONSE = "Hello! I am an AI assistant specialized in filtering and answering questions about Resumes (CVs). Please upload CVs and ask me questions about their content."


# Display chat messages from history on app rerun with custom colors
for chat_entry in st.session_state.display_chat_history:
    with st.chat_message("user"):
        # Blue color for user's question
        st.markdown(f"<span style='color:#007bff;'>{chat_entry['question']}</span>", unsafe_allow_html=True) 
    with st.chat_message("assistant"):
        # Green color for assistant's answer
        st.markdown(f"<span style='color:#28a745;'>{chat_entry['answer']}</span>", unsafe_allow_html=True) 

# THIS IS THE CRUCIAL LINE: The st.chat_input() must be called
# to define the 'prompt' variable on each Streamlit rerun.
prompt = st.chat_input(
    "Enter your question about the CVs:",
    disabled=not st.session_state.vector_store
)

if prompt:
    # Display the user's question immediately in the chat history with blue color
    with st.chat_message("user"):
        st.markdown(f"<span style='color:#007bff;'>{prompt}</span>", unsafe_allow_html=True)

    # Normalize prompt for greeting check
    normalized_prompt = prompt.strip().lower()

    # Check if the prompt is a greeting (handle outside the LLM chain for efficiency)
    if normalized_prompt in GREETINGS:
        answer = GREETING_RESPONSE
        with st.chat_message("assistant"):
            st.markdown(f"<span style='color:#28a745;'>{answer}</span>", unsafe_allow_html=True)
        # Add greeting to display history, but NOT to LLM memory history
        st.session_state.display_chat_history.append({"question": prompt, "answer": answer})
    else:
        # Prepare the conversation history for the LLM's prompt
        # This history will only contain relevant Q&A pairs, up to the last 5.
        history_for_llm_prompt = ""
        if st.session_state.llm_chat_history:
            history_for_llm_prompt = "Previous conversation:\n"
            # Loop through the last 5 entries of llm_chat_history
            for chat_turn in st.session_state.llm_chat_history[-5:]: # Use negative indexing to get last N
                history_for_llm_prompt += f"User: {chat_turn['question']}\n"
                history_for_llm_prompt += f"AI: {chat_turn['answer']}\n"
            history_for_llm_prompt += "\n" # Add a newline for separation

        # Process the query using the QA chain
        # Pass the formatted history string to get_qa_chain
        qa_chain = get_qa_chain(st.session_state.vector_store, chat_history_for_llm=history_for_llm_prompt)
        
        if qa_chain:
            with st.chat_message("assistant"):
                with st.spinner("Searching and generating answer..."):
                    try:
                        # Invoke the RetrievalQA chain
                        result = qa_chain.invoke({"query": prompt})
                        answer = result.get("result", "No answer found.")
                        
                        # Display assistant's answer with green color
                        st.markdown(f"<span style='color:#28a745;'>{answer}</span>", unsafe_allow_html=True)
                        
                        # Add to display history (always add for display)
                        st.session_state.display_chat_history.append({"question": prompt, "answer": answer})
                        
                        # Add to LLM memory history (only if not a greeting and successful)
                        st.session_state.llm_chat_history.append({"question": prompt, "answer": answer})

                    except Exception as e:
                        error_message = f"An error occurred while getting the answer: {e}"
                        st.error(error_message)
                        # Add error to display history, but NOT to LLM memory history
                        st.session_state.display_chat_history.append({"question": prompt, "answer": error_message})
        else:
            with st.chat_message("assistant"):
                warning_message = "QA chain not initialized. Please re-process CVs."
                st.warning(warning_message)
                # Add warning to display history, but NOT to LLM memory history
                st.session_state.display_chat_history.append({"question": prompt, "answer": warning_message})

else:
    # This block handles the case where prompt is None (no input yet)
    # or when the vector_store is not initialized.
    if not st.session_state.vector_store:
        st.info("Please upload and process CVs first to enable the chat interface.")

# --- Reset Button ---
if st.button("Reset Chatbot", key="reset_button"):
    st.session_state.vector_store = None
    st.session_state.display_chat_history = []
    st.session_state.llm_chat_history = [] # Reset LLM memory as well
    st.success("Chatbot reset. Please upload new CVs to begin.")
    st.rerun()