"""
TODO:
1. remove uploaded files -> clear button
4. write readme
"""

import os
import tempfile

import streamlit as st

from modules.chat_module import ChatHandler
from modules.loader_module import LoaderHandler
from modules.vectorstore_module import VectorStoreHandler
from utils import others

chat_llm = ChatHandler()
data_loader = LoaderHandler()
vectorstore_loader = VectorStoreHandler()
retriever, vectorstore, context = None, None, ""

# set page configs
st.set_page_config(
    page_title="Private RAG with LLama2 Model", layout="wide", page_icon="⚡"
)

# Initialize session state variables for uploaded files and chat history
if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = []
if "file_uploader_key" not in st.session_state:
    st.session_state["file_uploader_key"] = 0
if "messages" not in st.session_state:
    st.session_state.messages = []
# current_convo = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


with st.sidebar:
    loaded_docs = list()
    st.write("You can chat with your PDFs and URLs. Just provide them")

    text_input = st.text_input(
        "Enter your URL here:",
        help="Enter the URL you want to chat with.",
        # key="input_text",
        # value=st.session_state.input_text,
    )
    files = st.file_uploader(
        "Upload your PDFs here:",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can upload multiple PDFs up to 10 here. ",
        key=st.session_state["file_uploader_key"],
    )

    if st.button("LOAD DATA"):
        if files:
            st.session_state["uploaded_files"] = files
            for uploaded_file in files:
                path = others.save_temp_file(uploaded_file)
                data = data_loader.load_pdf(path)
                loaded_docs.append(data)

        if text_input:
            if "," in text_input:
                text_input = ",".split(text_input)
            else:
                text_input = [text_input]

            for each_input in text_input:
                data = data_loader.load_url(each_input)
                loaded_docs.append(data)

    with st.spinner("Loading data. Please wait!"):
        if loaded_docs:
            loaded_docs = [
                others.split_documents(loaded_doc) for loaded_doc in loaded_docs
            ]
            loaded_docs = [item for sublist in loaded_docs for item in sublist]
            retriever = vectorstore_loader.load_vectorstore(loaded_docs)
            st.success("Data loaded! You can chat now.")
    if st.button("Reset Chat"):
        st.session_state.messages = []

        st.session_state["file_uploader_key"] += 1
        st.experimental_rerun()
        if vectorstore:
            vectorstore.delete_collection()
        st.experimental_rerun()

if user_input := st.chat_input("Hi!"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    # current_convo.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    if retriever is not None:
        context = retriever.get_relevant_documents(user_input)

    chat_history = others.format_conversation(st.session_state.messages)
    prompt = others.generate_prompt(
        context=context, query=user_input, chat_history=chat_history
    )

    response = chat_llm.ask_llm(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
