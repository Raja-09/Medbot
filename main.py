from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from prompt import *
from datetime import datetime


import streamlit as st
from watsonxlangchain import LangChainInterface

creds = {
    "apikey": "OygblaOFT_3U3UGDDVPZFDE3kdmn54wxhuEICVvzx904",
    "url": "https://us-south.ml.cloud.ibm.com",
}
st.title("Ask our MEDBOT🤖")

st.markdown(
    """
***Disclaimer:*** 
* This application is intended for experimental use.
* The creators assume no responsibility for any outcomes related to its use.
* The advise given by the bot is not a substitute for professional medical advice.
"""
)

def initialize_lang_chain(temperature):
    llm = LangChainInterface(
        credentials=creds,
        model="meta-llama/llama-3-8b-instruct",
        params={"decoding_method": "sample", "max_new_tokens": 300, "temperature": temperature},
        project_id="111b2e25-a82c-45dc-ab80-e48629f202fd",
    )
    return llm


@st.cache_resource
def load_pdf():
    pdf_name = "Medical_book.pdf"
    loaders = [PyPDFLoader(pdf_name)]
    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2"),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0),
    ).from_loaders(loaders)
    return index


def create_retrieval_qa_chain(llm, index):
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=index.vectorstore.as_retriever(),
        input_key="symptoms",
        chain_type_kwargs=chain_type_kwargs,
    )
    return chain


def chat_interface(chain):

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        content = message["content"]
        st.chat_message(message["role"]).markdown(content)

    prompt = st.chat_input("Pass Your Prompt here")

    if prompt:
        st.chat_message("user").markdown(prompt)
        current_time = datetime.now()
        st.session_state.messages.append({"role": "user", "content": prompt})
        response = chain.run(prompt)
        new_time = datetime.now()
        print(response [:10]+" generated in "+str(new_time - current_time),len(prompt.split()))

        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


def main():
   
    start_time = datetime.now()
    index = load_pdf()
    index_loaded_time = datetime.now()
    print("Loaded pdf in:",index_loaded_time - start_time)
    llm = initialize_lang_chain(0.3)
    chain = create_retrieval_qa_chain(llm, index)
    chat_interface(chain)


if __name__ == "__main__":
    main()
