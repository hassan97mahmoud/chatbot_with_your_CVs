code
Markdown
# 📄 🤖 CHATBOT_WITH_YOUR_CVs

<div align="center">

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-8E75B9?style=for-the-badge&logo=google&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-008653?style=for-the-badge&logo=langchain&logoColor=white)

</div>

This project is a powerful and interactive CV (Resume) filtering assistant built with Streamlit, Google Gemini (via the Gemini 1.5 Flash model), and LangChain.

It enables HR professionals, recruiters, and hiring managers to upload multiple CVs in PDF format, process them into vector embeddings, and ask natural language questions to extract, summarize, compare, and rank candidates based on roles, skills, experience, and more.

---

## 🎥 Live Demo

Click the badge below to watch a video demonstration of the project in action.

[![Watch the Demo Video](https://img.shields.io/badge/Watch-Demo%20Video-red?style=for-the-badge&logo=youtube)](https://github.com/hassan97mahmoud/chatbot_with_your_CVs/blob/main/CHATBOT_WITH_YOUR_CVs-VEED.mp4)

---

## 🚀 Key Features

-   **Upload & Process Multiple CVs:** Supports PDF format with automated chunking and metadata tagging.
-   **Conversational QA Interface:** Powered by Google Gemini for querying CVs in natural language.
-   **Advanced Prompt Engineering:** Features a highly detailed system prompt with strict rules to:
    -   Avoid hallucinations and provide answers based only on CV context.
    -   Keep CV data strictly separate per individual.
    -   Extract specific information like contact details reliably.
-   **Smart Ranking & Comparison:**
    -   Identify and rank the most suitable candidates for specific roles.
    -   Provide side-by-side comparisons of individuals' experiences, skills, or education.
-   **Enhanced Retrieval with Multi-Query:** Uses `MultiQueryRetriever` to generate multiple perspectives on a user's question, improving the quality of retrieved information.
-   **Session Memory:** Maintains context for more natural multi-turn conversations.

## 🧠 Use Case Examples

-   *“Who has Python and machine learning skills?”*
-   *“Compare Ahmed and Salma’s education.”*
-   *“Find someone suitable for a Data Analyst role and give me their contact info.”*
-   *“Who is the most experienced project manager?”*
-   *“Rank all candidates with AWS certification.”*

## 🛠 Tech Stack

| Layer            | Technology                                                                                                                                                                                                |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Frontend**     | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)                                                                                             |
| **LLM**          | ![Google Gemini](https://img.shields.io/badge/Google%20Gemini-8E75B9?style=for-the-badge&logo=google&logoColor=white) (`gemini-1.5-flash`)                                                                 |
| **Embeddings**   | ![Google Generative AI Embeddings](https://img.shields.io/badge/Google%20Embeddings-4285F4?style=for-the-badge&logo=google&logoColor=white) (`text-embedding-004`)                                          |
| **Vector Store** | ![ChromaDB](https://img.shields.io/badge/ChromaDB-5B3AAB?style=for-the-badge&logo=chromadb&logoColor=white)                                                                                                 |
| **Framework**    | ![LangChain](https://img.shields.io/badge/LangChain-008653?style=for-the-badge&logo=langchain&logoColor=white)                                                                                             |
| **Document Parsing** | `PyPDFLoader` (LangChain)                                                                                                                                                                                 |

## 📁 Getting Started

Follow these steps to run the project locally:

### 1. Clone the repository
```bash
git clone https://github.com/hassan97mahmoud/chatbot_with_your_CVs.git
cd chatbot_with_your_CVs/cv_chatbot
