import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Replicate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain import PromptTemplate
from htmlTemplates import bot_template, user_template, css

# Define the path for generated embeddings
DB_FAISS_PATH = "vectorstore/db_faiss"

custom_prompt_template = """Use the following information to answer the users question, if you dont know the answer
just say " I don't know the answer". DO NOT make up answers that are not based on facts. Explain with detailed answers
that are easy to understand

Context: {context}
Question: {question}

Only return the useful aspects of the answer below and nothing else.
Helpful answer:
"""


## Code for prompt, retrieval and bot
def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vector store, we also pass in context and question.
    """
    prompt = PromptTemplate(
        template=custom_prompt_template, input_variables=["context", "question"]
    )
    return prompt


# get the text from the PDF
def get_pdf_text(pdf_file):
    text = ""
    pdf_name = pdf_file.name[:-4]
    if pdf_file is not None:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text, pdf_name


def get_chunk_text(text):

    text_splitter = CharacterTextSplitter(
        chunk_size=300, chunk_overlap=10, length_function=len
    )

    chunks = text_splitter.split_text(text)

    return chunks


def get_vector_store(text_chunks, pdf_name):

    # For OpenAI Embeddings

    # embeddings = OpenAIEmbeddings()

    # For Huggingface Embeddings

    embeddings = HuggingFaceHubEmbeddings()

    # Create or load the FAISS vector store
    if os.path.exists(f"{DB_FAISS_PATH}/{pdf_name}"):
        db = FAISS.load_local(f"{DB_FAISS_PATH}/{pdf_name}", embeddings)
    else:
        db = FAISS.from_texts(text_chunks, embedding=embeddings)
        db.save_local(f"{DB_FAISS_PATH}/{pdf_name}")

    return db


def get_conversation_chain(vector_store, prompt):

    # OpenAI Model

    # llm = ChatOpenAI()

    # HuggingFace Model

    # repo_id = "mistralai/Mistral-7B-v0.1"
    # llm = HuggingFaceHub(
    #     repo_id=repo_id,

    # )

    # Initialize Replicate Llama2 Model
    llm = Replicate(
        model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
        input={
            "temperature": 0.75,
            "max_length": 3000,
        },
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        # chain_type="stuff",
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
        memory=memory,
        # return_source_documents = True,
        # chain_type_kwargs = {'prompt':prompt}
    )

    return conversation_chain


def handle_user_input(question):
    if st.session_state.conversation is None:
        st.write("Conversation not initialized. Please upload a PDF file first.")
        return

    response = st.session_state.conversation({"question": question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(
                user_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True,
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True
            )


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with Your own PDFs", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with Your own PDFs :books:")
    question = st.text_input("Ask anything to your PDF: ")

    if question:
        handle_user_input(question)

    with st.sidebar:
        st.subheader("Upload your Documents Here: ")
        pdf_file = st.file_uploader(
            "Choose your PDF Files and Press OK",
            type="pdf",
            accept_multiple_files=False,
        )

        if st.button("OK"):
            with st.spinner("Processing your PDFs..."):

                if pdf_file is None:
                    st.write("Please upload a PDF file")
                    return

                # Get PDF Text
                raw_text, pdf_name = get_pdf_text(pdf_file)

                # Get Text Chunks
                text_chunks = get_chunk_text(raw_text)

                # Create Vector Store
                vector_store = get_vector_store(text_chunks, pdf_name)
                st.write("DONE")

                # Create conversation chain
                qa_prompt = set_custom_prompt()
                st.session_state.conversation = get_conversation_chain(vector_store, qa_prompt)

        st.markdown(
            """
        ## About
        This app is an LLM-powered chatbot built using:
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://python.langchain.com/)

        based on:
        [github](https://github.com/Abdullahw72/langchain-chatbot-multiple-PDF)

        """
        )


if __name__ == "__main__":
    main()
