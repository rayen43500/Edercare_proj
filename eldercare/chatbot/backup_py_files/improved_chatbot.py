import os
import json
import logging
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
import asyncio
import aiohttp
import google.generativeai as genai
import hashlib
from datetime import datetime, timedelta
from dotenv import load_dotenv
from functools import wraps
import backoff
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.chains import RetrievalQA
from langchain_community.llms import OpenAI as LangchainOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def async_retry(
    max_attempts: int = 3,
    initial_wait: float = 1.0,
    max_wait: float = 10.0,
    exponential_base: float = 2.0
):
    """Decorator for async functions to implement exponential backoff retry logic."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            attempt = 0
            while attempt < max_attempts:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    if attempt == max_attempts:
                        logger.error(f"Final attempt failed: {str(e)}")
                        raise
                    wait_time = min(
                        initial_wait * (exponential_base ** (attempt - 1)),
                        max_wait
                    )
                    logger.warning(
                        f"Attempt {attempt} failed: {str(e)}. "
                        f"Retrying in {wait_time:.2f} seconds..."
                    )
                    await asyncio.sleep(wait_time)
        return wrapper
    return decorator

class CacheError(Exception):
    """Custom exception for cache-related errors."""
    pass

class ResponseCache:
    """
    Enhanced cache for storing LLM responses with improved performance and memory management.
    """
    
    def __init__(
        self,
        cache_dir: str = "response_cache",
        ttl_hours: int = 24,
        max_cache_size: int = 1000,
        cleanup_interval: int = 3600  # 1 hour in seconds
    ):
        """
        Initialize the cache with improved configuration options.
        
        Args:
            cache_dir: Directory to store cache files
            ttl_hours: Time-to-live for cache entries in hours
            max_cache_size: Maximum number of cache entries
            cleanup_interval: Interval in seconds for cache cleanup
        """
        self.cache_dir = cache_dir
        self.ttl_hours = ttl_hours
        self.max_cache_size = max_cache_size
        self.cleanup_interval = cleanup_interval
        self._last_cleanup = datetime.now()
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize cache metadata
        self.metadata_file = os.path.join(self.cache_dir, "cache_metadata.json")
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """Load cache metadata from disk."""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, "r") as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = {"entries": {}, "last_cleanup": datetime.now().isoformat()}
        except Exception as e:
            logger.error(f"Error loading cache metadata: {str(e)}")
            self.metadata = {"entries": {}, "last_cleanup": datetime.now().isoformat()}
    
    def _save_metadata(self) -> None:
        """Save cache metadata to disk."""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self.metadata, f)
        except Exception as e:
            logger.error(f"Error saving cache metadata: {str(e)}")
            raise CacheError(f"Failed to save cache metadata: {str(e)}")
    
    def _get_cache_key(self, messages: List[Dict[str, str]]) -> str:
        """Generate a unique cache key for the messages."""
        message_str = json.dumps(messages, sort_keys=True)
        return hashlib.md5(message_str.encode()).hexdigest()
    
    def _cleanup_if_needed(self) -> None:
        """Clean up expired cache entries if needed."""
        now = datetime.now()
        if (now - self._last_cleanup).total_seconds() < self.cleanup_interval:
            return
            
        try:
            expired_keys = []
            for key, entry in self.metadata["entries"].items():
                timestamp = datetime.fromisoformat(entry["timestamp"])
                if now - timestamp > timedelta(hours=self.ttl_hours):
                    expired_keys.append(key)
                    cache_file = os.path.join(self.cache_dir, f"{key}.json")
                    if os.path.exists(cache_file):
                        os.remove(cache_file)
            
            # Remove expired entries from metadata
            for key in expired_keys:
                del self.metadata["entries"][key]
            
            self._last_cleanup = now
            self.metadata["last_cleanup"] = now.isoformat()
            self._save_metadata()
            
            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
                
        except Exception as e:
            logger.error(f"Error during cache cleanup: {str(e)}")
    
    def get(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """
        Retrieve a response from cache if it exists and is not expired.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Optional[str]: Cached response or None if not found
        """
        try:
            self._cleanup_if_needed()
            cache_key = self._get_cache_key(messages)
            
            if cache_key not in self.metadata["entries"]:
                return None
            
            entry = self.metadata["entries"][cache_key]
            timestamp = datetime.fromisoformat(entry["timestamp"])
            
            if datetime.now() - timestamp > timedelta(hours=self.ttl_hours):
                return None
            
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
            if not os.path.exists(cache_file):
                return None
            
            with open(cache_file, "r") as f:
                cache_data = json.load(f)
                return cache_data["response"]
                
        except Exception as e:
            logger.warning(f"Error reading from cache: {str(e)}")
            return None
    
    def set(self, messages: List[Dict[str, str]], response: str) -> None:
        """
        Store a response in the cache with size management.
        
        Args:
            messages: List of message dictionaries
            response: The response text to cache
        """
        try:
            self._cleanup_if_needed()
            cache_key = self._get_cache_key(messages)
            
            # Check if we need to remove oldest entries
            if len(self.metadata["entries"]) >= self.max_cache_size:
                oldest_key = min(
                    self.metadata["entries"].keys(),
                    key=lambda k: datetime.fromisoformat(self.metadata["entries"][k]["timestamp"])
                )
                del self.metadata["entries"][oldest_key]
                oldest_file = os.path.join(self.cache_dir, f"{oldest_key}.json")
                if os.path.exists(oldest_file):
                    os.remove(oldest_file)
            
            # Store new entry
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
            cache_data = {
                "timestamp": datetime.now().isoformat(),
                "response": response
            }
            
            with open(cache_file, "w") as f:
                json.dump(cache_data, f)
            
            # Update metadata
            self.metadata["entries"][cache_key] = {
                "timestamp": cache_data["timestamp"],
                "size": len(response)
            }
            self._save_metadata()
            
        except Exception as e:
            logger.error(f"Error writing to cache: {str(e)}")
            raise CacheError(f"Failed to write to cache: {str(e)}")

class ElderCareKnowledgeBase:
    """
    Enhanced knowledge base for the ElderCare Assistant using RAG (Retrieval-Augmented Generation).
    Implements improved document processing, caching, and retrieval mechanisms.
    """
    
    def __init__(
        self,
        knowledge_dir: str = "knowledge_base",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        max_retrieval_docs: int = 5
    ):
        """
        Initialize the knowledge base with improved configuration.
        
        Args:
            knowledge_dir: Directory containing knowledge base documents
            chunk_size: Size of document chunks for processing
            chunk_overlap: Overlap between chunks
            max_retrieval_docs: Maximum number of documents to retrieve
        """
        self.knowledge_dir = knowledge_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_retrieval_docs = max_retrieval_docs
        self.vector_store = None
        self.embeddings = None
        self.retriever = None
        self.raw_documents = {}
        self._initialized = False
        
    @async_retry(max_attempts=3)
    async def initialize(self, use_local_embeddings: bool = True) -> None:
        """
        Initialize the knowledge base with improved error handling and retry logic.
        
        Args:
            use_local_embeddings: Whether to use local embeddings model
        """
        if self._initialized:
            logger.info("Knowledge base already initialized")
            return
            
        try:
            # Create knowledge base directory if it doesn't exist
            os.makedirs(self.knowledge_dir, exist_ok=True)
            
            # Initialize embeddings model
            if use_local_embeddings:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                logger.info("Using local HuggingFace embeddings model")
            else:
                raise ValueError("OpenAI embeddings are not supported in this version")
            
            # Load and process documents
            await self._process_documents()
            
            self._initialized = True
            logger.info("Knowledge base initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Error initializing knowledge base: {str(e)}")
            raise
    
    async def _process_documents(self) -> None:
        """Process documents with improved error handling and performance."""
        try:
            # Load raw documents first
            self._load_raw_documents()
            
            # Load and process documents for vector store
            loader = DirectoryLoader(
                self.knowledge_dir,
                glob="**/*.txt",
                show_progress=True
            )
            documents = loader.load()
            
            if not documents:
                logger.warning("No documents found in knowledge base. Creating samples.")
                self._create_sample_documents()
                documents = loader.load()
            
            # Split documents with optimized parameters
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            splits = text_splitter.split_documents(documents)
            
            # Create vector store with optimized parameters
            self.vector_store = FAISS.from_documents(
                splits,
                self.embeddings,
                normalize_L2=True
            )
            
            # Create optimized retriever
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": self.max_retrieval_docs,
                    "score_threshold": 0.5
                }
            )
            
            logger.info(f"Processed {len(splits)} document chunks")
            
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            raise
    
    def _load_raw_documents(self) -> None:
        """Load raw documents with improved error handling."""
        try:
            for root, _, files in os.walk(self.knowledge_dir):
                for file in files:
                    if file.endswith('.txt'):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            
                            self.raw_documents[file] = {
                                "content": content,
                                "path": file_path,
                                "category": os.path.basename(root),
                                "last_modified": os.path.getmtime(file_path)
                            }
                        except Exception as e:
                            logger.warning(f"Error loading document {file}: {str(e)}")
                            continue
            
            logger.info(f"Loaded {len(self.raw_documents)} raw documents")
            
        except Exception as e:
            logger.error(f"Error loading raw documents: {str(e)}")
            raise
    
    @async_retry(max_attempts=3)
    async def retrieve_relevant_context(
        self,
        query: str,
        limit: int = 3,
        use_local_embeddings: bool = True
    ) -> str:
        """
        Retrieve relevant context with improved error handling and retry logic.
        
        Args:
            query: The search query
            limit: Maximum number of relevant documents to retrieve
            use_local_embeddings: Whether to use local embeddings
            
        Returns:
            str: Retrieved context
        """
        if not self._initialized:
            raise RuntimeError("Knowledge base not initialized")
            
        try:
            # Try vector search first
            if self.retriever:
                docs = await self.retriever.aget_relevant_documents(query)
                if docs:
                    return "\n\n".join(doc.page_content for doc in docs[:limit])
            
            # Fallback to keyword search
            return self._keyword_retrieval(query)
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            # Fallback to keyword search
            return self._keyword_retrieval(query)
    
    def _keyword_retrieval(self, query: str) -> str:
        """
        Enhanced keyword-based retrieval with improved relevance scoring.
        
        Args:
            query: The search query
            
        Returns:
            str: Retrieved content
        """
        try:
            query_words = set(query.lower().split())
            relevant_sections = []
            scores = []
            
            for doc_info in self.raw_documents.values():
                content = doc_info["content"]
                sections = content.split("\n##")
                
                for section in sections:
                    section = section.strip()
                    if not section:
                        continue
                        
                    # Calculate relevance score
                    section_words = set(section.lower().split())
                    word_matches = len(query_words.intersection(section_words))
                    if word_matches > 0:
                        score = word_matches / len(query_words)
                        relevant_sections.append(section)
                        scores.append(score)
            
            # Sort by relevance score
            if relevant_sections:
                sorted_sections = [s for _, s in sorted(zip(scores, relevant_sections), reverse=True)]
                return "\n\n".join(sorted_sections[:3])
            
            return "No relevant information found."
            
        except Exception as e:
            logger.error(f"Error in keyword retrieval: {str(e)}")
            return "Error retrieving information."

    def _create_sample_documents(self):
        """Create sample knowledge base documents if none exist."""
        sample_docs = {
            "tech_tips.txt": """
            # Technology Tips for Seniors
            
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
            # Health and Wellness for Seniors
            
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
            # Entertainment Recommendations for Seniors
            
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
        
        for filename, content in sample_docs.items():
            with open(os.path.join(self.knowledge_dir, filename), "w") as f:
                f.write(content)

class LLMProvider:
    """
    Enhanced LLM provider with improved error handling, caching, and performance optimizations.
    Supports both local HuggingFace models and external API backends with fallback mechanisms.
    """
    
    def __init__(
        self,
        cache_dir: str = "response_cache",
        max_retries: int = 3,
        timeout: int = 30,
        use_local_model: bool = True
    ):
        """
        Initialize the LLM provider with improved configuration.
        
        Args:
            cache_dir: Directory for response caching
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
            use_local_model: Whether to use local HuggingFace model by default
        """
        self.cache = ResponseCache(cache_dir=cache_dir)
        self.max_retries = max_retries
        self.timeout = timeout
        self._session = None
        self._model = None
        self.use_local_model = use_local_model
        self._local_model = None
        
    async def _ensure_session(self) -> None:
        """Ensure aiohttp session is initialized."""
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
    
    async def _ensure_model(self) -> None:
        """Ensure appropriate model is initialized."""
        if self.use_local_model:
            if self._local_model is None:
                try:
                    from transformers import pipeline
                    self._local_model = pipeline(
                        "text-generation",
                        model="gpt2",  # You can change this to any other model
                        device="cpu"
                    )
                except Exception as e:
                    logger.error(f"Error initializing local model: {str(e)}")
                    raise
        else:
            if self._model is None:
                try:
                    self._model = genai.GenerativeModel('gemini-pro')
                except Exception as e:
                    logger.error(f"Error initializing Gemini model: {str(e)}")
                    raise
    
    async def close(self) -> None:
        """Clean up resources."""
        if self._session:
            await self._session.close()
            self._session = None
    
    @async_retry(max_attempts=3)
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 300,
        use_cache: bool = True,
        context: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Generate response with improved error handling and caching.
        
        Args:
            messages: List of message dictionaries
            temperature: Response temperature
            max_tokens: Maximum tokens in response
            use_cache: Whether to use response caching
            context: Optional context from knowledge base
            
        Returns:
            Tuple[str, str]: Generated response and source
        """
        try:
            # Check cache first
            if use_cache:
                cached_response = self.cache.get(messages)
                if cached_response:
                    logger.info("Using cached response")
                    return cached_response, "cache"
            
            # Ensure resources are initialized
            await self._ensure_session()
            await self._ensure_model()
            
            if self.use_local_model:
                # Use local model with context
                if context:
                    prompt = f"Context: {context}\n\nUser: {messages[-1]['content']}\n\nAssistant:"
                else:
                    prompt = f"User: {messages[-1]['content']}\n\nAssistant:"
                
                response = self._local_model(
                    prompt,
                    max_length=max_tokens,
                    temperature=temperature,
                    num_return_sequences=1
                )[0]['generated_text']
                
                # Extract only the assistant's response
                response_text = response.split("Assistant:")[-1].strip()
                source = "local"
            else:
                # Use external API
                conversation = self._model.start_chat(history=[])
                
                # Add messages to conversation
                for msg in messages:
                    if msg["role"] == "user":
                        conversation.send_message(msg["content"])
                    else:
                        conversation.send_message(msg["content"], role="assistant")
                
                # Generate response
                response = await conversation.send_message(
                    messages[-1]["content"],
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                        "top_p": 0.8,
                        "top_k": 40
                    }
                )
                
                response_text = response.text
                source = "gemini"
            
            # Cache the response
            if use_cache:
                self.cache.set(messages, response_text)
            
            return response_text, source
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            # Try to fallback to the other model if one fails
            if self.use_local_model:
                logger.info("Falling back to external API...")
                self.use_local_model = False
                return await self.generate_response(
                    messages, temperature, max_tokens, use_cache, context
                )
            else:
                logger.info("Falling back to local model...")
                self.use_local_model = True
                return await self.generate_response(
                    messages, temperature, max_tokens, use_cache, context
                )
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Format messages for logging and debugging.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            str: Formatted message string
        """
        formatted = []
        for msg in messages:
            role = msg["role"].upper()
            content = msg["content"]
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted)

class ElderCareAssistant:
    """
    Enhanced ElderCare Assistant with improved error handling, memory management,
    and performance optimizations.
    """
    
    def __init__(
        self,
        user_profile: Optional[Dict[str, Any]] = None,
        knowledge_dir: str = "knowledge_base",
        cache_dir: str = "response_cache",
        max_history: int = 10,
        use_local_model: bool = True
    ):
        """
        Initialize the assistant with improved configuration.
        
        Args:
            user_profile: User profile information
            knowledge_dir: Directory for knowledge base
            cache_dir: Directory for response caching
            max_history: Maximum number of conversation turns to remember
            use_local_model: Whether to use local model by default
        """
        self.user_profile = user_profile or {
            "id": "default",
            "name": "User",
            "age": 65,
            "interests": [],
            "health_conditions": [],
            "tech_comfort": "beginner"
        }
        self.knowledge_base = ElderCareKnowledgeBase(knowledge_dir=knowledge_dir)
        self.llm_provider = LLMProvider(
            cache_dir=cache_dir,
            use_local_model=use_local_model
        )
        self.max_history = max_history
        self.conversation_history: List[Dict[str, str]] = []
        self._initialized = False
        self.use_local_model = use_local_model
    
    async def initialize(self) -> None:
        """Initialize the assistant with improved error handling."""
        if self._initialized:
            logger.info("Assistant already initialized")
            return
            
        try:
            await self.knowledge_base.initialize()
            self._initialized = True
            logger.info("Assistant initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing assistant: {str(e)}")
            raise
    
    def _get_personalized_system_prompt(self) -> str:
        """
        Generate a personalized system prompt based on user profile.
        
        Returns:
            str: Personalized system prompt
        """
        name = self.user_profile.get("name", "User")
        age = self.user_profile.get("age", 65)
        interests = ", ".join(self.user_profile.get("interests", []))
        health_conditions = ", ".join(self.user_profile.get("health_conditions", []))
        tech_comfort = self.user_profile.get("tech_comfort", "beginner")
        
        return f"""You are an ElderCare Assistant, a helpful and empathetic AI companion for elderly users.
Your current user is {name}, who is {age} years old.
Their interests include: {interests}
Health conditions to be aware of: {health_conditions}
Their technical comfort level is: {tech_comfort}

Please:
1. Be patient and clear in your responses
2. Use simple language appropriate for their technical comfort level
3. Show empathy and understanding
4. Provide helpful information about their interests
5. Be mindful of their health conditions
6. Offer assistance with technology when needed
7. Maintain a friendly and supportive tone

Remember to:
- Break down complex information into simple steps
- Provide clear explanations
- Be encouraging and positive
- Respect their privacy and dignity
- Focus on their well-being and comfort"""
    
    def _update_conversation_history(self, role: str, content: str) -> None:
        """
        Update conversation history with size management.
        
        Args:
            role: Message role (user/assistant)
            content: Message content
        """
        self.conversation_history.append({"role": role, "content": content})
        if len(self.conversation_history) > self.max_history * 2:  # *2 because each turn has 2 messages
            self.conversation_history = self.conversation_history[-self.max_history * 2:]
    
    @async_retry(max_attempts=3)
    async def process_message(self, user_message: str) -> str:
        """
        Process user message with improved error handling and context management.
        
        Args:
            user_message: User's message
            
        Returns:
            str: Assistant's response
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Update conversation history
            self._update_conversation_history("user", user_message)
            
            # Get relevant context from knowledge base
            context = None
            if self.use_local_model:
                context = await self.knowledge_base.retrieve_relevant_context(user_message)
            
            # Prepare messages for LLM
            messages = [
                {"role": "system", "content": self._get_personalized_system_prompt()}
            ]
            
            if context:
                messages.append({
                    "role": "system",
                    "content": f"Relevant information:\n{context}"
                })
            
            messages.extend(self.conversation_history)
            
            # Generate response
            response, source = await self.llm_provider.generate_response(
                messages=messages,
                temperature=0.7,
                max_tokens=300,
                use_cache=True,
                context=context
            )
            
            # Update conversation history
            self._update_conversation_history("assistant", response)
            
            # Log the response source
            logger.info(f"Response generated using {source} model")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return (
                "I apologize, but I'm having trouble processing your message right now. "
                "Please try again in a moment."
            )
    
    def switch_model(self, use_local: bool) -> None:
        """
        Switch between local and external API models.
        
        Args:
            use_local: Whether to use local model
        """
        self.use_local_model = use_local
        self.llm_provider.use_local_model = use_local
        logger.info(f"Switched to {'local' if use_local else 'external'} model")
    
    async def get_tech_help(self, app_name: str) -> str:
        """
        Get technology help with improved error handling.
        
        Args:
            app_name: Name of the application
            
        Returns:
            str: Help information
        """
        try:
            context = await self.knowledge_base.retrieve_relevant_context(
                f"how to use {app_name} for elderly users"
            )
            
            messages = [
                {"role": "system", "content": self._get_personalized_system_prompt()},
                {"role": "system", "content": f"Context about {app_name}:\n{context}"},
                {"role": "user", "content": f"Can you help me understand how to use {app_name}?"}
            ]
            
            response, _ = await self.llm_provider.generate_response(
                messages=messages,
                temperature=0.5,  # Lower temperature for more focused responses
                max_tokens=400
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error getting tech help: {str(e)}")
            return (
                f"I apologize, but I'm having trouble finding information about {app_name}. "
                "Please try asking about a different application or try again later."
            )
    
    async def recommend_entertainment(
        self,
        category: str,
        preferences: Optional[List[str]] = None
    ) -> str:
        """
        Recommend entertainment with improved personalization.
        
        Args:
            category: Entertainment category
            preferences: User preferences
            
        Returns:
            str: Entertainment recommendations
        """
        try:
            # Get user interests from profile
            user_interests = self.user_profile.get("interests", [])
            
            # Combine preferences
            all_preferences = list(set(user_interests + (preferences or [])))
            
            # Get relevant context
            context = await self.knowledge_base.retrieve_relevant_context(
                f"{category} recommendations for elderly users with interests in {', '.join(all_preferences)}"
            )
            
            messages = [
                {"role": "system", "content": self._get_personalized_system_prompt()},
                {"role": "system", "content": f"Context about {category}:\n{context}"},
                {"role": "user", "content": f"Can you recommend some {category} based on my interests?"}
            ]
            
            response, _ = await self.llm_provider.generate_response(
                messages=messages,
                temperature=0.8,  # Higher temperature for more creative recommendations
                max_tokens=400
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error recommending entertainment: {str(e)}")
            return (
                f"I apologize, but I'm having trouble finding {category} recommendations right now. "
                "Please try again later or ask about a different category."
            )
    
    async def provide_health_tip(self, concern: Optional[str] = None) -> str:
        """
        Provide health tips with improved context awareness.
        
        Args:
            concern: Specific health concern
            
        Returns:
            str: Health tip
        """
        try:
            # Get user's health conditions
            health_conditions = self.user_profile.get("health_conditions", [])
            
            # Prepare query
            query = concern if concern else "general health tips"
            if health_conditions:
                query += f" for someone with {', '.join(health_conditions)}"
            
            # Get relevant context
            context = await self.knowledge_base.retrieve_relevant_context(query)
            
            messages = [
                {"role": "system", "content": self._get_personalized_system_prompt()},
                {"role": "system", "content": f"Health information:\n{context}"},
                {"role": "user", "content": f"Can you provide a health tip about {query}?"}
            ]
            
            response, _ = await self.llm_provider.generate_response(
                messages=messages,
                temperature=0.6,
                max_tokens=300
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error providing health tip: {str(e)}")
            return (
                "I apologize, but I'm having trouble providing health information right now. "
                "Please consult with your healthcare provider for medical advice."
            )
    
    def update_user_profile(self, new_information: Dict[str, Any]) -> None:
        """
        Update user profile with improved validation.
        
        Args:
            new_information: New user information
        """
        try:
            # Validate and update each field
            for key, value in new_information.items():
                if key in self.user_profile:
                    if isinstance(value, type(self.user_profile[key])):
                        self.user_profile[key] = value
                    else:
                        logger.warning(f"Invalid type for {key}: expected {type(self.user_profile[key])}, got {type(value)}")
                else:
                    logger.warning(f"Unknown profile field: {key}")
            
            logger.info("User profile updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating user profile: {str(e)}")
            raise
    
    def extract_intent(self, user_message: str) -> Dict[str, Any]:
        """
        Extract user intent with improved accuracy.
        
        Args:
            user_message: User's message
            
        Returns:
            Dict[str, Any]: Extracted intent information
        """
        try:
            message_lower = user_message.lower()
            
            # Define intent patterns
            intents = {
                "tech_help": ["how to use", "help with", "trouble with", "can't use"],
                "entertainment": ["recommend", "suggestion", "what to watch", "what to read"],
                "health": ["health", "exercise", "diet", "sleep", "pain", "symptom"],
                "greeting": ["hello", "hi", "hey", "good morning", "good afternoon"],
                "farewell": ["bye", "goodbye", "see you", "talk later"]
            }
            
            # Check for intent matches
            detected_intents = {}
            for intent, patterns in intents.items():
                if any(pattern in message_lower for pattern in patterns):
                    detected_intents[intent] = True
            
            # Extract entities
            entities = {
                "app_name": None,
                "category": None,
                "health_concern": None
            }
            
            # Extract app name
            if "tech_help" in detected_intents:
                for word in message_lower.split():
                    if word in ["app", "application", "program", "software"]:
                        idx = message_lower.split().index(word)
                        if idx > 0:
                            entities["app_name"] = message_lower.split()[idx-1]
            
            # Extract category
            if "entertainment" in detected_intents:
                categories = ["movie", "book", "music", "game", "show"]
                for category in categories:
                    if category in message_lower:
                        entities["category"] = category
                        break
            
            # Extract health concern
            if "health" in detected_intents:
                health_terms = ["pain", "sleep", "diet", "exercise", "symptom"]
                for term in health_terms:
                    if term in message_lower:
                        entities["health_concern"] = term
                        break
            
            return {
                "intents": detected_intents,
                "entities": entities,
                "confidence": 0.8 if detected_intents else 0.5
            }
            
        except Exception as e:
            logger.error(f"Error extracting intent: {str(e)}")
            return {
                "intents": {},
                "entities": {},
                "confidence": 0.0
            }
    
    async def add_to_knowledge_base(self, category: str, content: str) -> bool:
        """
        Add information to knowledge base with improved validation.
        
        Args:
            category: Information category
            content: Information content
            
        Returns:
            bool: Success status
        """
        try:
            # Validate category
            if not category or not isinstance(category, str):
                raise ValueError("Invalid category")
            
            # Validate content
            if not content or not isinstance(content, str):
                raise ValueError("Invalid content")
            
            # Create category directory if it doesn't exist
            category_dir = os.path.join(self.knowledge_base.knowledge_dir, category)
            os.makedirs(category_dir, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{category}_{timestamp}.txt"
            filepath = os.path.join(category_dir, filename)
            
            # Write content to file
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            
            # Reinitialize knowledge base to include new content
            await self.knowledge_base.initialize()
            
            logger.info(f"Added new content to knowledge base: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding to knowledge base: {str(e)}")
            return False

async def main():
    """
    Main function with improved error handling and cleanup.
    """
    assistant = None
    try:
        # Initialize with a sample user profile
        user_profile = {
            "id": "user1",
            "name": "User",
            "age": 65,
            "interests": ["reading", "music", "gardening"],
            "health_conditions": ["arthritis"],
            "tech_comfort": "beginner"
        }
        
        logger.info("Creating ElderCare Assistant...")
        assistant = ElderCareAssistant(
            user_profile=user_profile,
            knowledge_dir="knowledge_base",
            cache_dir="response_cache",
            max_history=10,
            use_local_model=True
        )
        
        logger.info("Initializing assistant...")
        await assistant.initialize()
        
        logger.info("Processing greeting...")
        response = await assistant.process_message("Hello! How can you help me today?")
        print(f"\nAssistant: {response}\n")
        
        # Keep the chat running
        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\nGoodbye! Have a wonderful day!")
                    break
                
                response = await assistant.process_message(user_input)
                print(f"\nAssistant: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! Have a wonderful day!")
                break
            except Exception as e:
                logger.error(f"Error in chat loop: {str(e)}")
                print("\nI apologize, but I encountered an error. Please try again.")
                continue
    
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        print("\nI apologize, but I encountered a serious error. Please restart the application.")
    
    finally:
        # Cleanup
        if assistant:
            try:
                await assistant.llm_provider.close()
            except Exception as e:
                logger.error(f"Error during cleanup: {str(e)}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nGoodbye! Have a wonderful day!")
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        print("\nI apologize, but I encountered a serious error. Please restart the application.") 