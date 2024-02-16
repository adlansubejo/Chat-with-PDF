import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import os
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_chat import message
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain

# Sidebar contents
with st.sidebar:
    st.title("Chat PDF")
    st.markdown(
        """
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)

    """
    )
    add_vertical_space(5)
    # st.write('')


def load_llm():
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-xxl",
        model_kwargs={"temperature": 0.5, "max_length": 512},
    )
    return llm


# Define the path for generated embeddings
DB_FAISS_PATH = "vectorstore/db_faiss"


def main():
    load_dotenv()
    st.header("Chat with PDF")

    # upload a PDF file
    pdf = st.file_uploader("Upload a PDF file", type="pdf")

    # read the PDF file
    if pdf is None:
        return

    pdf_reader = PdfReader(pdf)

    pages = pdf_reader.pages

    # document splitting
    text_pages = []
    for pages in pages:
        text_pages.append(pages.extract_text())

    chunk_size = 300
    chunk_overlap = 10

    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunks = []
    for page in text_pages:
        chunks.extend(r_splitter.split_text(page))

    # embeddings
    store_name = pdf.name[:-4]

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    # Create or load the FAISS vector store
    if os.path.exists(f"{DB_FAISS_PATH}/{store_name}"):
        db = FAISS.load_local(f"{DB_FAISS_PATH}/{store_name}", embeddings)
    else:
        db = FAISS.from_texts(chunks, embedding=embeddings)
        db.save_local(f"{DB_FAISS_PATH}/{store_name}")

    # Load the language model
    llm = load_llm()

    # Create a conversational chain
    dbRetriever = db.as_retriever()

    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=dbRetriever)

    # Function for conversational chat
    def conversational_chat(query):
        result = chain({"question": query, "chat_history": st.session_state["history"]})
        st.session_state["history"].append((query, result["answer"]))
        return result["answer"]

    # Initialize chat history
    if "history" not in st.session_state:
        st.session_state["history"] = []

    # Initialize messages
    if "generated" not in st.session_state:
        st.session_state["generated"] = ["Hello ! Ask me about " + store_name + " 🤗"]

    if "past" not in st.session_state:
        st.session_state["past"] = ["Hey ! 👋"]

    # Create containers for chat history and user input
    response_container = st.container()
    container = st.container()

    # User input form
    with container:
        with st.form(key="my_form", clear_on_submit=True):
            user_input = st.text_input(
                "Query:", placeholder="Talk to pdf file 👉 (:", key="input"
            )
            submit_button = st.form_submit_button(label="Send")

        if submit_button and user_input:
            output = conversational_chat(user_input)
            st.session_state["past"].append(user_input)
            st.session_state["generated"].append(output)

    # Display chat history
    if st.session_state["generated"]:
        with response_container:
            for i in range(len(st.session_state["generated"])):
                message(
                    st.session_state["past"][i],
                    is_user=True,
                    key=str(i) + "_user",
                    avatar_style="big-smile",
                )
                message(
                    st.session_state["generated"][i], key=str(i), avatar_style="thumbs"
                )


if __name__ == "__main__":
    main()
