📄 CV Filtering Chatbot using Azure OpenAI & LangChain
This project is a powerful and interactive CV (Resume) filtering assistant built with 
Streamlit, Azure OpenAI GPT-4.0-mini, and LangChain. It enables HR professionals, 
recruiters, and hiring managers to upload multiple CVs in PDF format, process them 
into vector embeddings, and ask natural language questions to extract, summarize, 
compare, and rank candidates based on roles, skills, experience, and more.

********************************************************************
🚀 Features: 
************
🔍 Upload & Process Multiple CVs (PDFs) with automated chunking and metadata tagging

💬 Conversational QA Interface powered by Azure GPT-4.0-mini for querying CVs

📌 Advanced Prompt Engineering with strict rules to:

Avoid hallucinations

Keep CVs strictly separate per individual

Validate job roles and suggest similar ones

🧠 Smart Ranking: Identify and rank the most suitable candidates for specific roles

📊 Side-by-Side Comparison between individuals' experiences, skills, or education

♻️ Session Memory for contextual multi-turn conversations

🌐 Built with LangChain, Chroma Vector DB, and Azure OpenAI

********************************************************************
🧠 Use Case Examples: 
*********************
“Who has Python and machine learning skills?”

“Compare Ahmed and Salma’s education.”

“Find someone suitable for a Data Analyst role.”

“Who is the most experienced project manager?”

“Show all candidates with AWS certification.”

********************************************************************
🛠 Tech Stack: 
**************

Frontend: Streamlit

LLM: Azure OpenAI GPT-4.0-mini

Embeddings: Azure OpenAI Embeddings

Vector Store: Chroma

Document Parsing: PyPDFLoader from LangChain

Prompt Engineering: ChatPromptTemplate, SystemMessagePromptTemplate

********************************************************************
📁 How to Run: 
**************
    1)     Clone the repo

      2)   Create a .env file with your Azure OpenAI credentials

        3) Run the app:
                streamlit run app.py