from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from prompt import *


import streamlit as st
from watsonxlangchain import LangChainInterface

creds = {
    "apikey": "OygblaOFT_3U3UGDDVPZFDE3kdmn54wxhuEICVvzx904",
    "url": "https://us-south.ml.cloud.ibm.com",
}
llm = LangChainInterface(
    credentials=creds,
    model="meta-llama/llama-2-70b-chat",
    params={"decoding_method": "sample", "max_new_tokens": 500, "temperature": 0.5},
    project_id="111b2e25-a82c-45dc-ab80-e48629f202fd",
)


@st.cache_resource
def load_pdf():
    pdf_name = "Medical_book.pdf"
    loaders = [PyPDFLoader(pdf_name)]

    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2"),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0),
    ).from_loaders(loaders)

    return index


PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": PROMPT}
index = load_pdf()

chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=index.vectorstore.as_retriever(),
    input_key="symptoms",
    chain_type_kwargs=chain_type_kwargs,
)

st.title("Ask our MEDBOT🤖")

st.markdown(
    """
***Disclaimer:*** 
* This application is intended for experimental use.
* The creators assume no responsibility for any outcomes related to its use.
"""
)
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

prompt = st.chat_input("Pass Your Prompt here")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = chain.run(prompt)

    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
