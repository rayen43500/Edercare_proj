import os
from typing import List, Dict, Tuple
import logging
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedKnowledgeBase:
    def __init__(self, knowledge_dir: str = "knowledge_base"):
        """Initialize the enhanced knowledge base with embeddings and FAISS."""
        self.knowledge_dir = knowledge_dir
        self.documents = {}
        self.sections = []
        self.section_sources = []
        
        # Initialize the embedding model
        logger.info("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize FAISS index
        self.dimension = 384  # Dimension of the embeddings
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Load documents and create embeddings
        self._load_documents()
        self._create_embeddings()
    
    def _load_documents(self):
        """Load all documents from the knowledge base directories."""
        directories = ['technology', 'health', 'entertainment', 'companion']
        
        for directory in directories:
            dir_path = os.path.join(self.knowledge_dir, directory)
            if os.path.exists(dir_path):
                for filename in os.listdir(dir_path):
                    if filename.endswith('.txt'):
                        file_path = os.path.join(dir_path, filename)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                key = f"{directory}/{filename}"
                                self.documents[key] = f.read()
                                logger.info(f"Loaded document: {key}")
                        except Exception as e:
                            logger.error(f"Error loading {file_path}: {str(e)}")
        
        if not self.documents:
            logger.warning("No documents found in knowledge base!")
        else:
            logger.info(f"Loaded {len(self.documents)} documents from knowledge base")
    
    def _create_embeddings(self):
        """Create embeddings for all sections and build FAISS index."""
        for filename, content in self.documents.items():
            # Split content into sections
            sections = self._split_into_sections(content)
            
            for section in sections:
                if section.strip():
                    self.sections.append(section)
                    self.section_sources.append(filename)
        
        if self.sections:
            # Create embeddings for all sections
            logger.info("Creating embeddings...")
            embeddings = self.embedding_model.encode(self.sections, show_progress_bar=True)
            
            # Add embeddings to FAISS index
            self.index.add(np.array(embeddings).astype('float32'))
            logger.info(f"Created {len(self.sections)} embeddings")
    
    def _split_into_sections(self, content: str) -> List[str]:
        """Split content into meaningful sections."""
        sections = []
        current_section = []
        
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                if current_section:
                    sections.append('\n'.join(current_section))
                    current_section = []
                continue
            
            if line[0].isdigit() or (line.isupper() and len(line) > 3):
                if current_section:
                    sections.append('\n'.join(current_section))
                    current_section = []
            current_section.append(line)
        
        if current_section:
            sections.append('\n'.join(current_section))
        
        return sections
    
    def find_relevant_info(self, query: str, k: int = 2) -> str:
        """Find relevant information using FAISS similarity search."""
        # Create query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Search in FAISS index
        distances, indices = self.index.search(
            np.array([query_embedding]).astype('float32'), k
        )
        
        # Get relevant sections
        relevant_sections = []
        for idx in indices[0]:
            if idx < len(self.sections):  # Ensure index is valid
                section = self.sections[idx]
                source = self.section_sources[idx]
                relevant_sections.append(f"[From {source}]\n{section}")
        
        if relevant_sections:
            return "\n\n".join(relevant_sections)
        
        return "I couldn't find specific information about that. Please try asking something else."

class EnhancedChatbot:
    def __init__(self):
        """Initialize the enhanced chatbot."""
        self.knowledge_base = EnhancedKnowledgeBase()
        
        # Initialize the question-answering pipeline
        logger.info("Loading question-answering model...")
        self.qa_pipeline = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            device=0 if torch.cuda.is_available() else -1
        )
    
    def generate_response(self, context: str, query: str) -> str:
        """Generate a response using the question-answering pipeline."""
        try:
            # Use the QA pipeline to get an answer
            result = self.qa_pipeline(
                question=query,
                context=context,
                max_answer_len=100,
                handle_impossible_answer=True
            )
            
            if result['score'] > 0.5:  # Only use answers with good confidence
                return result['answer']
            else:
                return "I found some information but I'm not confident enough to give a specific answer."
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I'm having trouble processing that question. Could you try rephrasing it?"
    
    def get_response(self, user_input: str) -> str:
        """Get a response for the user's input."""
        # Get relevant information
        relevant_info = self.knowledge_base.find_relevant_info(user_input)
        
        if relevant_info and "I couldn't find" not in relevant_info:
            # Generate response using the QA pipeline
            response = self.generate_response(relevant_info, user_input)
            return f"Here's what I found:\n\n{relevant_info}\n\nMy response:\n{response}"
        
        return "I'm not sure how to help with that. Could you try asking something else?"

def main():
    """Run the enhanced chatbot."""
    try:
        chatbot = EnhancedChatbot()
        
        print("Enhanced Chatbot initialized! Type 'quit' to exit.")
        print("You can ask questions about technology, health, entertainment, or companionship.")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\nGoodbye!")
                    break
                
                response = chatbot.get_response(user_input)
                print(f"\nAssistant: {response}")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
                print("Please try again.")
    
    except Exception as e:
        print(f"Error initializing chatbot: {str(e)}")
        print("Please make sure all dependencies are installed and try again.")

if __name__ == "__main__":
    main() 