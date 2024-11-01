from langchain_community.vectorstores import Qdrant
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
import qdrant_client
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")

# Initialize Qdrant client and vector store
client = qdrant_client.QdrantClient(
    url=QDRANT_HOST,
    api_key=QDRANT_API_KEY
)
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
vectorstore = Qdrant(
    client=client,
    collection_name=QDRANT_COLLECTION_NAME,
    embeddings=embeddings
)

# Set up the retrieval QA chain
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(api_key=OPENAI_API_KEY),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Query the chain
query = "What is the role of Arjun here ?"
response = qa.invoke(query)  # Changed from .run() to .invoke()
print(response)
