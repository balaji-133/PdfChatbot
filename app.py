# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.llms import Ollama
# import streamlit as st
# import os

# from dotenv import load_dotenv

# load_dotenv()

# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# os.environ["LANGCHAIN_TRACING_V2"]="true"
# os.environ["LANGCHAIN_PROJECT"]="Simple Q&A chatbot with ollama"

# pmpt=ChatPromptTemplate.from_messages(
#     [
#     ("system","You are a helpful assistant that helps people find information.please give interactive responses for given query"),
#     ("user","Question: {question}")
# ])

# def gen_ans(question,engine,temp,max_tokens):
#     llm=Ollama(model=engine)
#     opsr=StrOutputParser()
#     chain=pmpt|llm|opsr
#     ans=chain.invoke({"question":question})
#     return ans

# engine=st.sidebar.selectbox("Select Model",["gemma2","mistral"])
# temperature=st.sidebar.slider("Select Temperature",min_value=0.0,max_value=1.0,value=0.7)
# max_tokens=st.sidebar.slider("Select Max Tokens",min_value=50,max_value=300,value=150)

# st.write("# Q&A Chatbot with Ollama")
# user_input=st.text_input("Enter your question here")

# if user_input:
#     response=gen_ans(user_input,engine,temperature,max_tokens)
#     st.write(response)
# else:
#     st.write("Please enter a question to get an answer.")
# app.py

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A chatbot with ollama"

# Define the prompt
pmpt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that helps people find information. Please give interactive responses for the given query."),
    ("user", "Question: {question}")
])

# Define the answer generation function
def gen_ans(question, engine, temp, max_tokens):
    llm = OllamaLLM(model=engine, temperature=temp, max_tokens=max_tokens)
    output_parser = StrOutputParser()
    chain = pmpt | llm | output_parser
    answer = chain.invoke({"question": question})
    return answer

# Streamlit UI
st.set_page_config(page_title="Q&A Chatbot with Ollama")
st.title("🤖 Q&A Chatbot with Ollama")

# Sidebar controls
engine = st.sidebar.selectbox("Select Model", ["gemma:2b", "mistral"])
temperature = st.sidebar.slider("Select Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Select Max Tokens", min_value=50, max_value=300, value=150)

# User input
user_input = st.text_input("Enter your question here")

# Display response
if user_input:
    with st.spinner("Generating answer..."):
        try:
            response = gen_ans(user_input, engine, temperature, max_tokens)
            st.success("Response:")
            st.write(response)
        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.info("Please enter a question to get an answer.")
