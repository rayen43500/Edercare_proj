import os
import json
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import asyncio
from dotenv import load_dotenv
import unittest
from unittest.mock import MagicMock, patch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Try to import the real dependencies first
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import TextLoader, DirectoryLoader
    REAL_IMPORTS = True
except ImportError:
    # If imports fail, we'll use mocks
    REAL_IMPORTS = False
    logger.warning("LangChain imports failed, using mock objects for testing")


class MockEmbeddings:
    """Mock embeddings class for testing without API"""
    def embed_documents(self, texts):
        # Return a simple mock embedding for each text (dimensionality=5)
        return [[hash(text) % 100 / 100, 
                (hash(text) >> 8) % 100 / 100,
                (hash(text) >> 16) % 100 / 100,
                (hash(text) >> 24) % 100 / 100,
                (hash(text) >> 32) % 100 / 100] for text in texts]
    
    def embed_query(self, text):
        # Return a simple mock embedding for the query
        return [hash(text) % 100 / 100, 
                (hash(text) >> 8) % 100 / 100,
                (hash(text) >> 16) % 100 / 100,
                (hash(text) >> 24) % 100 / 100,
                (hash(text) >> 32) % 100 / 100]


class MockFAISS:
    """Mock FAISS vector store for testing"""
    @classmethod
    def from_documents(cls, documents, embedding):
        instance = cls()
        instance.docs = documents
        instance.embedding = embedding
        return instance
    
    def as_retriever(self, search_type=None, search_kwargs=None):
        return MockRetriever(self.docs)


class MockRetriever:
    """Mock retriever for testing"""
    def __init__(self, docs):
        self.docs = docs
    
    def get_relevant_documents(self, query):
        # For testing, just return 2 docs that contain parts of the query
        results = []
        for doc in self.docs[:min(2, len(self.docs))]:
            results.append(doc)
        return results


class MockDocument:
    """Mock document for testing"""
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


class MockTextSplitter:
    """Mock text splitter for testing"""
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_documents(self, documents):
        results = []
        for doc in documents:
            # Simple split by paragraphs for testing
            content = doc.page_content
            paragraphs = content.split('\n\n')
            for para in paragraphs:
                if para.strip():
                    results.append(MockDocument(para.strip(), doc.metadata))
        return results


class MockDirectoryLoader:
    """Mock directory loader for testing"""
    def __init__(self, path, glob=None):
        self.path = path
        self.glob = glob
    
    def load(self):
        # Create mock documents if directory doesn't exist
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            return []

        results = []
        for filename in os.listdir(self.path):
            if filename.endswith('.txt'):
                with open(os.path.join(self.path, filename), 'r') as f:
                    content = f.read()
                    results.append(MockDocument(content, {"source": filename}))
        return results


class ElderCareKnowledgeBase:
    def __init__(self, knowledge_dir: str = "knowledge_base"):
        """
        Initialize the knowledge base.
        
        Args:
            knowledge_dir: Directory containing knowledge base documents
        """
        self.knowledge_dir = knowledge_dir
        self.vector_store = None
        self.embeddings = None
        self.retriever = None
    
    async def initialize(self, use_local_embeddings: bool = True):
        """
        Initialize the knowledge base with documents from the knowledge directory.
        
        Args:
            use_local_embeddings: Whether to use local embeddings model instead of OpenAI
        """
        # Create knowledge base directory if it doesn't exist
        if not os.path.exists(self.knowledge_dir):
            os.makedirs(self.knowledge_dir)
            # Create sample documents if directory is empty
            self._create_sample_documents()
        
        # Load dependencies based on what's available
        if REAL_IMPORTS:
            # Use real libraries if available
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Load documents from knowledge base directory
            try:
                loader = DirectoryLoader(self.knowledge_dir, glob="**/*.txt")
                documents = loader.load()
                
                if not documents:
                    logger.warning("No documents found in knowledge base. Creating samples.")
                    self._create_sample_documents()
                    documents = loader.load()
                    
                # Split documents into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                splits = text_splitter.split_documents(documents)
                
                # Create vector store
                self.vector_store = FAISS.from_documents(splits, self.embeddings)
                self.retriever = self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3}
                )
                
                logger.info(f"Knowledge base initialized with {len(splits)} document chunks")
                
            except Exception as e:
                logger.error(f"Error initializing knowledge base: {str(e)}")
                # Fallback to simple retriever
                if self.vector_store:
                    self.retriever = self.vector_store.as_retriever()
        else:
            # Use mock objects for testing
            logger.info("Using mock objects for testing (no LangChain imports)")
            self.embeddings = MockEmbeddings()
            
            loader = MockDirectoryLoader(self.knowledge_dir, glob="**/*.txt")
            documents = loader.load()
            
            if not documents:
                logger.warning("No documents found in knowledge base. Creating samples.")
                self._create_sample_documents()
                documents = loader.load()
            
            text_splitter = MockTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)
            
            self.vector_store = MockFAISS.from_documents(splits, self.embeddings)
            self.retriever = self.vector_store.as_retriever()
            
            logger.info(f"Mock knowledge base initialized with {len(splits)} document chunks")

    def _create_sample_documents(self):
        """Create sample knowledge base documents if none exist."""
        sample_docs = {
            "tech_tips.txt": """
            ## Smartphone Basics
            - To turn on your smartphone, press and hold the power button on the side or top.
            - Increase text size: Go to Settings > Display > Font Size.
            - Make the screen brighter: Pull down from the top of the screen and adjust the brightness slider.
            
            ## Common Apps
            ### WhatsApp
            1. Find the green WhatsApp icon on your home screen
            2. Tap on it to open
            3. To send a message, tap on a contact name
            4. Type your message at the bottom and press the arrow to send
            5. To make a call, tap on the phone icon in the top right
            
            ### Facebook
            1. Find the blue Facebook icon on your home screen
            2. Tap on it to open
            3. Scroll down to see posts from friends and family
            4. To like a post, tap the thumbs up button
            5. To comment, tap the comment button and type your message
            """,
            
            "health_tips.txt": """
            ## Arthritis Management
            - Gentle exercises like swimming and walking can help reduce joint pain
            - Apply warm compresses to stiff joints in the morning
            - Cold packs can help reduce inflammation after activity
            - Consult your doctor if pain suddenly increases or new symptoms appear
            
            ## Maintaining Good Sleep
            - Try to go to bed and wake up at the same time each day
            - Create a relaxing bedtime routine
            - Avoid screens (TV, phone) at least 30 minutes before bed
            - Keep your bedroom cool, dark, and quiet
            
            ## Hearing Loss
            - Face the person speaking and watch their lips
            - Ask people to speak clearly, not loudly
            - Reduce background noise when having conversations
            - Consider a hearing evaluation if you often ask people to repeat themselves
            """,
            
            "entertainment.txt": """
            ## Books for Mystery Lovers
            - "The Thursday Murder Club" by Richard Osman - A lighthearted mystery solved by four friends in a retirement community
            - "Still Life" by Louise Penny - First in the Inspector Gamache series, gentle mysteries with rich characters
            - "The No. 1 Ladies' Detective Agency" by Alexander McCall Smith - Charming mysteries set in Botswana
            
            ## Classical Music Collections
            - "50 Greatest Pieces of Classical Music" by London Philharmonic Orchestra
            - "Mozart for Morning Coffee" - Gentle classical pieces to start the day
            - "Relaxing Classical" compilation - Soothing pieces for relaxation
            
            ## Gardening Shows and Resources
            - "Gardeners' World" - Long-running BBC gardening show
            - "Monty Don's Italian Gardens" - Exploration of beautiful Italian gardens
            - "Growing Gracefully" - Gardening techniques adapted for seniors
            """
        }
        
        # Create the files
        for filename, content in sample_docs.items():
            with open(os.path.join(self.knowledge_dir, filename), "w") as f:
                f.write(content)
    
    async def query(self, question: str) -> str:
        """
        Query the knowledge base with a question.
        
        Args:
            question: The question to ask
            
        Returns:
            str: Answer to the question
        """
        if not self.retriever:
            return "Knowledge base has not been initialized yet."
        
        try:
            # Get relevant documents
            docs = self.retriever.get_relevant_documents(question)
            
            if not docs:
                return "I couldn't find any relevant information for your question."
            
            # Extract the content from the docs
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # For testing, just return the relevant context
            return f"Based on the available information:\n{context}"
            
        except Exception as e:
            logger.error(f"Error querying knowledge base: {str(e)}")
            return f"Error querying knowledge base: {str(e)}"


class TestElderCareKnowledgeBase(unittest.TestCase):
    """Test class for ElderCareKnowledgeBase"""
    
    def setUp(self):
        # Use a test-specific directory
        self.test_dir = "test_knowledge_base"
        if os.path.exists(self.test_dir):
            # Clean up any existing files
            for file in os.listdir(self.test_dir):
                os.remove(os.path.join(self.test_dir, file))
        else:
            os.makedirs(self.test_dir)
    
    def tearDown(self):
        # Clean up test files
        if os.path.exists(self.test_dir):
            for file in os.listdir(self.test_dir):
                os.remove(os.path.join(self.test_dir, file))
            os.rmdir(self.test_dir)
    
    async def test_initialization(self):
        """Test if the knowledge base initializes correctly"""
        kb = ElderCareKnowledgeBase(knowledge_dir=self.test_dir)
        await kb.initialize(use_local_embeddings=True)
        
        # Verify sample documents were created
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "tech_tips.txt")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "health_tips.txt")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "entertainment.txt")))
        
        # Verify retriever was created
        self.assertIsNotNone(kb.retriever)
    
    async def test_query(self):
        """Test if queries return relevant information"""
        kb = ElderCareKnowledgeBase(knowledge_dir=self.test_dir)
        await kb.initialize(use_local_embeddings=True)
        
        # Test query about tech
        result = await kb.query("How do I use WhatsApp?")
        self.assertIn("WhatsApp", result)
        
        # Test query about health
        result = await kb.query("What can help with arthritis?")
        self.assertIn("Arthritis", result)
        
        # Test query about entertainment
        result = await kb.query("Recommend some mystery books")
        self.assertIn("Mystery", result)


async def main():
    """Main function to run tests"""
    # Create test instance
    test_loader = unittest.TestLoader()
    test_suite = test_loader.loadTestsFromTestCase(TestElderCareKnowledgeBase)
    
    # Run the tests
    runner = unittest.TextTestRunner()
    print("Running tests...")
    test_result = runner.run(test_suite)
    
    # If tests pass, demonstrate a manual query
    if not test_result.failures and not test_result.errors:
        print("\nTests passed! Demonstrating manual query...")
        kb = ElderCareKnowledgeBase()
        await kb.initialize(use_local_embeddings=True)
        
        while True:
            query = input("\nEnter a question (or 'exit' to quit): ")
            if query.lower() == 'exit':
                break
            
            answer = await kb.query(query)
            print(f"\nAnswer: {answer}")


if __name__ == "__main__":
    asyncio.run(main())