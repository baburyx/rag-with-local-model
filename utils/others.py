import os
import tempfile

from langchain.text_splitter import RecursiveCharacterTextSplitter


def generate_prompt(query, chat_history, context=""):
    """
    Generate a prompt for the chat model.
    :param query: The user query.
    :param chat_history: The history of the chat conversation.
    :param context: Additional context for the query.
    :return: A formatted prompt string.
    """

    template = """[INST] <<SYS>>\nYou are a helpful assistant,
A user is going to ask a question.Answer in maximum of 3 sentences. Refer to the Related Documents below 
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
    """
    Splits a loaded document into smaller chunks.

    Uses RecursiveCharacterTextSplitter with specified chunk size and overlap. Ideal for large documents to enable easier processing.

    Parameters:
    - loaded_doc: The document to be split, expected to be a string or a compatible format.

    Returns:
    - List of document chunks.
    """
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=0, encoding_name="cl100k_base"
    )
    splits = splitter.split_documents(loaded_doc)
    return splits


def format_conversation(conversation):
    """
    Formats a list of conversation entries into a string.

    Parameters:
    - conversation: List of dict with 'role' and 'content' keys.

    Returns:
    - String of formatted conversation.
    """
    conversation = conversation[-3:]
    formatted_output = []
    for item in conversation:
        if item["role"] == "user":
            formatted_output.append(f"[INST]{item['content']}[/INST]")
        else:
            formatted_output.append(f"{item['content']}")
    return "\n".join(formatted_output)


def save_temp_file(file):
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, file.name)
    with open(path, "wb") as f:
        f.write(file.getvalue())
    return path
