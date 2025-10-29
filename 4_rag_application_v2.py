


import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI , GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from langsmith import traceable

load_dotenv()   

# Set Project name for LangSmith tracking
os.environ["LANGSMITH_PROJECT"] = "RAGChatbot"
PDF_PATH = "Books.pdf"   

# 1) Load PDF

@traceable(name="load_pdf")
def load_pdf(path: str):
    loader = PyPDFLoader(path)
    return loader.load()  # one Document per page

# 2) Chunk
@traceable(name="chunk_documents")
def chunk_documents(docs, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

# 3) Embed + index
@traceable(name="embed_and_index")
def embed_and_index(splits):
    emb =  HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vs = FAISS.from_documents(splits, emb)
    return vs


# Setup the entire pipeline with tracing

@traceable(name="setup_pipeline")
def setup_pipeline(pdf_path: str):
    docs = load_pdf(pdf_path)
    splits = chunk_documents(docs)
    retriever = embed_and_index(splits)
    return retriever

# 4) Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

# 5) Chain
api_key = os.getenv("GEMINI_API_KEY")
llm =  ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0,
    api_key=api_key
)
def format_docs(docs): return "\n\n".join(d.page_content for d in docs)



# Build the index under traced setup

vector_store = setup_pipeline(PDF_PATH)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})



parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

chain = parallel | prompt | llm | StrOutputParser()

# 6) Ask questions
print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
q = input("\nQ: ")


config  = {
    "run_name": "RAG Chatbot Run",
}


ans = chain.invoke(q.strip())
print("\nA:", ans)





# @traceable decorators have been added to key functions to enable LangSmith tracing for the entire RAG pipeline, including PDF loading, document chunking, embedding, and indexing.

# Here is one issues with this implementation:
# We are tracing setup_pipeline and chain separately, which means the linkage between them is not captured in LangSmith. It would be better to have a single trace that encompasses both the setup and the querying process to provide a complete picture of the RAG workflow.