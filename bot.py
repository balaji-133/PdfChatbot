import os
import streamlit as st
from dotenv import load_dotenv

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv()

# Get HF token
hf_token = os.getenv('HF_TOKEN')
if not hf_token:
    st.error("HF_TOKEN not found. Make sure it's set in your .env file.")
    st.stop()

# Set HF token
os.environ['HF_TOKEN'] = hf_token

# Initialize HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Streamlit UI
st.title("Conversational RAG with PDF and Chat History")
st.write("Upload PDF(s) and ask questions based on their content.")

# Groq API key input
api_key = st.text_input("Enter Groq API Key", type="password")

if api_key:
    # Initialize LLM
    llm = ChatGroq(api_key=api_key, model="Gemma2-9b-It")

    # Session ID input
    session_id = st.text_input("Enter Session ID", value="default_session")

    # Set up session state
    if 'store' not in st.session_state:
        st.session_state.store = {}

    # PDF file uploader
    uploaded_files = st.file_uploader("Choose PDF(s)", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        documents = []

        with st.spinner("Loading and processing PDFs..."):
            for uploaded_file in uploaded_files:
                # Save PDF to a unique temp file
                temp_pdf_path = f"./temp_{uploaded_file.name.replace(' ', '_')}"
                with open(temp_pdf_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                # Load and extract text
                loader = PyPDFLoader(temp_pdf_path)
                docs = loader.load()
                documents.extend(docs)

            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)

            # Create vector store
            vector_store = Chroma.from_documents(
                documents=splits,
                embedding=embeddings
                # Optional: add persist_directory="./chroma_db"
            )
            retriever = vector_store.as_retriever()

        # Prompt for contextual question reformulation
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do not answer the question — "
            "just reformulate it if needed and otherwise return it as it is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        history_aware_retriever = create_history_aware_retriever(
            llm=llm,
            retriever=retriever,
            prompt=contextualize_q_prompt
        )

        # Prompt for question answering
        system_prompt = (
            "You are an assistant in question answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you don't know. "
            "Give answers in both 2 marks and 8 marks formats. "
            "Explain the answer briefly.\n\n{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        # Final RAG chain
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Chat history retrieval function
        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # User query
        user_input = st.text_input("Ask a question about the PDFs")

        if user_input:
            session_history = get_session_history(session_id)

            with st.spinner("Thinking..."):
                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={
                        "configurable": {"session_id": session_id}
                    }
                )

            # Output
            st.success("Assistant:")
            st.markdown(response["answer"])

            # Optional: show chat history
            with st.expander("Chat History"):
                for message in session_history.messages:
                    st.write(message)

    else:
        st.warning("Please upload at least one PDF file.")
else:
    st.warning("Please enter your Groq API key to proceed.")
