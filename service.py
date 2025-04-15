import os
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone import Pinecone
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List
from pinecone import Pinecone, ServerlessSpec
import logging
import openai
import json

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = "quickstart"
SIMILARITY_THRESHOLD = 0.40
DATA_FILE = "scraped_data_Dutch.json"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o")

def initialize_pinecone_index():
    existing_indexes = pc.list_indexes().names()
    if INDEX_NAME not in existing_indexes:
        logger.info("Creating Pinecone index...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        logger.info("Index created successfully!")
    else:
        logger.info("Index already exists!")


def check_index_data():
    index = pc.Index(INDEX_NAME)
    stats = index.describe_index_stats()
    return stats['total_vector_count'] > 0


def create_embeddings():
    try:
        if check_index_data():
            print("Index already has data. Skipping embedding creation.")
            return

        if not os.path.exists(DATA_FILE):
            print("Error: Data file not found.")
            return

        with open(DATA_FILE, "r", encoding="utf-8") as f:
            genk_data = json.load(f)

        index = pc.Index(INDEX_NAME)
        vectors = []
        count = 0

        for item in genk_data:
            if "AllDetails" not in item or not item["AllDetails"]:
                continue
            details = item["AllDetails"]
            det = str(hash(details))  # unique id

            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=details
            )
            embedding = response.data[0].embedding

            vectors.append((
                det,
                embedding,
                {
                    "details": details,
                    "url": item.get("URL", ""),
                    "title": item.get("Title", ""),
                }
            ))
            count += 1

            # Batch insert every 100 vectors
            if len(vectors) >= 100:
                index.upsert(vectors)
                vectors = []

        if vectors:
            index.upsert(vectors)

        print(f"Successfully stored {count} embeddings in Pinecone.")

    except Exception as e:
        logger.error(f"Embedding creation failed: {e}")

class PineconeRetrieverWithThreshold(BaseRetriever):
    def _get_relevant_documents(self, query: str) -> List[Document]:
        index = pc.Index(INDEX_NAME)
        results = index.query(
            vector=embeddings.embed_query(query),
            top_k=3,
            include_metadata=True,
            include_values=False
        )
        
        documents = []
        for match in results.matches:
            if match.score >= SIMILARITY_THRESHOLD:
                metadata = match.metadata or {}
                documents.append(Document(
                    page_content=metadata.get("details", ""),
                    metadata={
                        "title": metadata.get("title", ""),
                        "url": metadata.get("url", "")
                    }
                ))
        return documents

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)

# Initialize components
retriever = PineconeRetrieverWithThreshold()

# Contextualize question prompt
contextualize_q_system_prompt = (
    "Gegeven een chatgeschiedenis en de laatste gebruikersvraag, "
    "formuleer een zelfstandige vraag. Herformuleer alleen indien nodig."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# History-aware retriever
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Answer prompt
qa_system_prompt = (
    "Beantwoord vragen strikt gebaseerd op de context, altijd in het Nederlands. "
    "Gebruik opsommingstekens (•) als de informatie baat heeft bij een lijstvorm."
    "Vermeld relevante URLs en titels. Als informatie ontbreekt, zeg 'Ik weet het niet'.\n\n"
    "Context:\n{context}"
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Document formatting
document_prompt = PromptTemplate.from_template(
    "Titel: {title}\nInhoud: {page_content}\nBron: {url}\n"
)

# Create chains
question_answer_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=qa_prompt,
    document_prompt=document_prompt,
)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

def continual_chat():
    print("Start chatten! Typ 'exit' om te stoppen.")
    chat_history = []
    while True:
        query = input("Jij: ")
        if query.lower() == 'exit':
            break
        result = rag_chain.invoke({
            "input": query,
            "chat_history": chat_history
        })
        print(f"\nAI: {result['answer']}\n")
        chat_history.extend([
            HumanMessage(content=query),
            AIMessage(content=result["answer"])
        ])

if __name__ == "__main__":
    print("Initizializing data\n")
    initialize_pinecone_index()
    if not pc.Index(INDEX_NAME).describe_index_stats()['total_vector_count'] > 0:
        create_embeddings()
    continual_chat()