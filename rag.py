from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder
import os

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

db_path = "./chroma_db"
model_path = "./my_model_cache"

#Initialize BGE-M3
model_name = "BAAI/bge-m3"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings' : True}

embeddings = HuggingFaceEmbeddings(
    model_name = model_name,
    model_kwargs = model_kwargs,
    encode_kwargs = encode_kwargs,
    cache_folder = model_path
)

#Initialize llm 
llm = ChatOllama(
    model = "qwen2.5:3b",
    temperature=0,
    num_predict=512
)

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    max_length=512
)


def load_data():
    loader = PyPDFLoader("data/doc.pdf")
    data = loader.load()

    return data

def chunk_data():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 512,
        chunk_overlap = 50,
        length_function = len,
        is_separator_regex=False,
    )

    doc = load_data()
    chunks = text_splitter.split_documents(doc)

    return chunks

def store_vector():
    chunks = chunk_data()

    if os.path.exists(db_path):
        vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    else:
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
    return vector_db

def rerank_documents(query: str, documents: list, top_n: int = 5) -> list:
    pairs = [(query, doc.page_content) for doc in documents]

    scores = reranker.predict(pairs)
    
    scored_docs = sorted(zip(scores, documents), key=lambda x:x[0], reverse=True)
    return [doc for _, doc in scored_docs[:top_n]]


def retrieve_data(userQuery: str):
    retriever = store_vector().as_retriever(search_kwargs={"k":20})
    candidate_docs = retriever.invoke(userQuery)

    reranked_docs = rerank_documents(userQuery, candidate_docs, top_n=5)

    context = "\n\n".join(doc.page_content for doc in reranked_docs)

    template = """Answer the question based on the following context:
    {context}
    
    Question: {question}"""

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    return chain.invoke({"context": context, "question": userQuery})


while True:
    user_query = input("\nYou: ")

    if user_query.lower() in ['exit','q','quit']:
        print("Goodbye")
        break;

    try:
        llm_response = retrieve_data(user_query)
        print(f"\nQWEN: {llm_response}")
    except Exception as e:
        print(f"An error occured: {e}")