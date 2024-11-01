from langchain_community.vectorstores import Qdrant  # Updated import
from langchain_community.embeddings import OpenAIEmbeddings  # Updated import
import qdrant_client
import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

# Access environment variables
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")

# Initialize Qdrant client
client = qdrant_client.QdrantClient(
    url=QDRANT_HOST,
    api_key=QDRANT_API_KEY
)

# Define collection name and configuration
collection_config = qdrant_client.http.models.VectorParams(
    size=1536,  # Set the vector size according to embedding model
    distance=qdrant_client.http.models.Distance.COSINE
)

# Check if the collection exists, delete if it does, then create a new one
if client.collection_exists(QDRANT_COLLECTION_NAME):
    client.delete_collection(QDRANT_COLLECTION_NAME)

client.create_collection(
    collection_name=QDRANT_COLLECTION_NAME,
    vectors_config=collection_config
)

print("Collection setup complete.")

# Create your vector store with OpenAI embeddings
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
vectorstore = Qdrant(
    client=client,
    collection_name=QDRANT_COLLECTION_NAME,
    embeddings=embeddings
)

print("Vector store created successfully.")

# Add documents to the vector database
# Load and split text into chunks
def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200, # feeding the context
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

with open("story.txt") as f:
    raw_text = f.read()

texts = get_chunks(raw_text)

# Add texts to the vector store
vectorstore.add_texts(texts)

print("Documents added to the vector database.")

# Set up the retrieval QA chain
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(api_key=OPENAI_API_KEY),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Query the chain
query = "How many friends are there and what are their names?"
response = qa.run(query)
print(response)
