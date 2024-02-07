import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import pickle
import os
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory

# Sidebar contents
with st.sidebar:
    st.title("Chat PDF")
    st.markdown(
        """
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model

    """
    )
    add_vertical_space(5)
    # st.write('')


def handle_userinput(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        st.write(message.content)

def get_conversation_chain(vectorstore):
    # llm = ChatOpenAI()
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


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
    # st.write(f'{store_name}')

    if os.path.exists(f"{store_name}.pkl"):
        with open(f"{store_name}.pkl", "rb") as f:
            vectorStore = pickle.load(f)
        # st.write('Embeddings Loaded from the Disk')s
    else:
        embeddings = HuggingFaceEmbeddings()
        vectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        with open(f"{store_name}.pkl", "wb") as f:
            pickle.dump(vectorStore, f)

    # Accept user questions/query
    query = st.text_input("Ask questions about your PDF file:")

    buttonProcess = st.button("Process")

    if query and buttonProcess:
        with st.spinner("Processing..."):
            # st.write('Processing...')

            # doc = VectorStore.similarity_search(query=query, k=3)

            llm = HuggingFaceHub(
                repo_id="google/flan-t5-xxl",
                model_kwargs={"temperature": 0.5, "max_length": 512},
            )

            qa = RetrievalQA.from_chain_type(
                chain_type="stuff",
                retriever=vectorStore.as_retriever(),
                llm=llm,
                memory=ConversationBufferMemory(memory_key="qa_history"),
            )

            response = qa(query)

            st.write(response)



if __name__ == "__main__":
    main()
