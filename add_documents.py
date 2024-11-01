from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import qdrant_client
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")

# Initialize Qdrant client and vector store with OpenAI embeddings
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

# Function to split text into chunks
def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Load and split text
with open("story.txt") as f:
    raw_text = f.read()

print("add_documents.py : Raw Text =", raw_text)

texts = get_chunks(raw_text)

print("add_documents.py : texts in chucks =", texts)

# Add texts to the vector store
vectorstore.add_texts(texts)

print("add_documents.py : Documents added to the vector database.")
