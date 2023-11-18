"""
TODO:
1. remove uploaded files -> clear button
2. reorganize script
3. use more session state 
4. write readme
"""

import os
import tempfile

import streamlit as st
from langchain.chat_models import ChatOllama
from langchain.document_loaders import PDFMinerLoader, WebBaseLoader
from langchain.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

# set page configs
st.set_page_config(
    page_title="Private RAG with LLama2 Model", layout="wide", page_icon="⚡"
)

retriever = None
vectorstore = None
context = ""


# Define a function to clear the cache
@st.cache_resource
def clear_cache():
    st.cache_resource.clear()


def generate_prompt(query, chat_history, context=""):
    template = """[INST] <<SYS>>You are a helpful assistant,
A user is going to ask a question. Refer to the Related Documents below 
when answering their question. Use them as much as possible
when answering the question. If you do not know the answer, say so.

Do not answer questions not related.

When answering questions, take into consideration the history of the 
chat converastion, which is listed below under Chat History. The chat history 
is in chronological order, so the most recent exhange is at the bottom.

Related Documents:
==================
{}
Chat History:
=============
{}
<</SYS>>

{} [/INST]
"""
    prompt = template.format(context, chat_history, query)
    return prompt


def split_documents(loaded_doc):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=0, encoding_name="cl100k_base"
    )
    splits = splitter.split_documents(loaded_doc)
    return splits


# Function to handle the reset action
def reset_inputs():
    # st.session_state["uploaded_files"] = None
    st.session_state["inserted_urls"] = None
    st.session_state.messages = []


# Initialize session state variables for uploaded files and text input
if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = []
if "inserted_urls" not in st.session_state:
    st.session_state["inserted_urls"] = ""

with st.sidebar:
    loaded_docs = list()
    st.write("You can chat with your PDFs and URLs. Just provide them")

    text_input = st.text_input(
        "Enter your URLs here:",
        help="You can write multiple URLs here. Seperate using comma (,).",
        key="inserted_urls",
    )
    uploaded_files = st.file_uploader(
        "Upload your PDFs here:",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can upload multiple PDFs up to 10 here. ",
        key="uploaded_files",
    )
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # save uploaded files to temporary files
            temp_dir = tempfile.mkdtemp()
            path = os.path.join(temp_dir, uploaded_file.name)
            with open(path, "wb") as f:
                f.write(uploaded_file.getvalue())
            # load and appent to chunks' list
            loader = PDFMinerLoader(path)
            data = loader.load()
            loaded_docs.append(data)
    if text_input:
        if "," in text_input:
            text_input = text_input.split(",")
        else:
            text_input = [text_input]

        for url in text_input:
            loader = WebBaseLoader(url)
            data = loader.load()
            loaded_docs.append(data)
if loaded_docs:
    loaded_docs = [split_documents(loaded_doc) for loaded_doc in loaded_docs]
    loaded_docs = [item for sublist in loaded_docs for item in sublist]
    if loaded_docs:
        # Initialize vectorstore and add data to vectorstore
        vectorstore = Chroma.from_documents(
            documents=loaded_docs,
            collection_name="rag-private",
            embedding=GPT4AllEmbeddings(),  # type: ignore
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})


def ask_llm(chat_model, query):
    response = chat_model.invoke(query)
    return response.content


ollama_llm = "llama2:7b-chat"
chat_llm = ChatOllama(model=ollama_llm)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_input)

    if retriever is not None:
        context = retriever.get_relevant_documents(user_input)

    def format_conversation(conversation):
        formatted_output = []
        for item in conversation:
            speaker = "human" if item["role"] == "user" else "ai"
            formatted_output.append(f"{speaker}: {item['content']}")
        return "\n".join(formatted_output)

    last3_chat = st.session_state.messages[-3:]
    last3_chat = format_conversation(last3_chat)

    prompt = generate_prompt(context=context, query=user_input, chat_history=last3_chat)

    response = ask_llm(chat_llm, prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)

with st.sidebar:
    # Adding a Reset Chat button
    if st.button("Reset Chat"):
        if vectorstore:
            vectorstore.delete_collection()
        clear_cache()
        reset_inputs()
        st.experimental_rerun()
