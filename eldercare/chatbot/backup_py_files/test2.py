import os
import logging
from typing import Optional
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from transformers import pipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalQAKnowledgeBase:
    def __init__(self, docs_dir="./knowledge_base"):
        self.docs_dir = docs_dir
        self.documents_text = ""
        self.qa_pipeline = None
        self.initialize()

    def initialize(self):
        try:
            if not os.path.exists(self.docs_dir):
                logger.error(f"Knowledge base directory {self.docs_dir} not found.")
                return

            logger.info(f"Loading documents from {self.docs_dir}")
            loader = DirectoryLoader(self.docs_dir, glob="**/*.txt", loader_cls=TextLoader)
            raw_documents = loader.load()

            if not raw_documents:
                logger.warning("No documents found in the knowledge base directory.")
                return

            logger.info(f"Found {len(raw_documents)} documents. Splitting and combining content...")
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.split_documents(raw_documents)

            self.documents_text = "\n".join([doc.page_content for doc in docs])

            logger.info("Loading HuggingFace QA pipeline...")
            self.qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
            logger.info("Knowledge base and QA model initialized successfully!")

        except Exception as e:
            logger.error(f"Initialization error: {str(e)}")

    def query(self, question: str) -> str:
        if not self.qa_pipeline or not self.documents_text:
            return "Knowledge base is not ready. Please check initialization."

        try:
            result = self.qa_pipeline({
                'question': question,
                'context': self.documents_text
            })

            answer = result.get('answer', '')
            if not answer.strip():
                return "No answer could be found in the knowledge base."

            return answer

        except Exception as e:
            logger.error(f"Query error: {str(e)}")
            return "Error while processing your question."

def main():
    kb = LocalQAKnowledgeBase(docs_dir="./knowledge_base")

    print("\n===== Local QA Knowledge Base =====")
    print("Ask a question based on your documents.")
    print("Type 'quit' to exit.")
    print("=====================================\n")

    while True:
        query = input("Enter your question: ").strip()
        if query.lower() in ["quit", "exit", "q"]:
            break

        print("\nSearching knowledge base for:", query)
        print("-" * 40)

        answer = kb.query(query)
        print("Answer:", answer)
        print("-" * 40)

if __name__ == "__main__":
    main()
