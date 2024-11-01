from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import OpenAIEmbeddings
import qdrant_client
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")

# Initialize Qdrant client
client = qdrant_client.QdrantClient(
    url=QDRANT_HOST,
    api_key=QDRANT_API_KEY
)

# Define collection configuration
collection_config = qdrant_client.http.models.VectorParams(
    size=1536,
    distance=qdrant_client.http.models.Distance.COSINE
)

# Create or recreate the collection
if client.collection_exists(QDRANT_COLLECTION_NAME):
    client.delete_collection(QDRANT_COLLECTION_NAME)

client.create_collection(
    collection_name=QDRANT_COLLECTION_NAME,
    vectors_config=collection_config
)

print("Collection setup complete.")
