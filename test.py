from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from prompt import *

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

def test_cases(prompt,chain):
    response = chain.run(prompt)
    return response.lower()

assert "cataracts" in test_cases("decrease in clarity of vision, not fully correctable with glasses. loss of contrast sensitivity.Disturbing glare in light",chain)
assert "jaundice" in test_cases("Dark urine, Pale or clay-colored stools, Yellow color inside the mouth, and Itching what am i suffering from ", chain)
assert "pneumonia" in test_cases("I have a fever, cough, and shortness of breath. What could be the cause?", chain)

