import os
import json
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import asyncio
import aiohttp
import openai
import hashlib
from datetime import datetime, timedelta
from dotenv import load_dotenv
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class ResponseCache:
    """
    Simple cache for storing LLM responses to reduce API calls and costs.
    """
    
    def __init__(self, cache_dir: str = "response_cache", ttl_hours: int = 24):
        """
        Initialize the cache.
        
        Args:
            cache_dir: Directory to store cache files
            ttl_hours: Time-to-live for cache entries in hours
        """
        self.cache_dir = cache_dir
        self.ttl_hours = ttl_hours
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def _get_cache_key(self, messages: List[Dict[str, str]]) -> str:
        """Generate a unique cache key for the messages."""
        # Serialize messages to string and hash
        message_str = json.dumps(messages, sort_keys=True)
        return hashlib.md5(message_str.encode()).hexdigest()
    
    def get(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """
        Retrieve a response from cache if it exists and is not expired.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Optional[str]: Cached response or None if not found
        """
        cache_key = self._get_cache_key(messages)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if not os.path.exists(cache_file):
            return None
            
        try:
            with open(cache_file, "r") as f:
                cache_data = json.load(f)
                
            # Check if cache is expired
            timestamp = datetime.fromisoformat(cache_data["timestamp"])
            if datetime.now() - timestamp > timedelta(hours=self.ttl_hours):
                # Cache expired
                return None
                
            return cache_data["response"]
            
        except Exception as e:
            logger.warning(f"Error reading from cache: {str(e)}")
            return None
    
    def set(self, messages: List[Dict[str, str]], response: str) -> None:
        """
        Store a response in the cache.
        
        Args:
            messages: List of message dictionaries
            response: The response text to cache
        """
        cache_key = self._get_cache_key(messages)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        try:
            cache_data = {
                "timestamp": datetime.now().isoformat(),
                "response": response
            }
            
            with open(cache_file, "w") as f:
                json.dump(cache_data, f)
                
        except Exception as e:
            logger.warning(f"Error writing to cache: {str(e)}")

class ElderCareKnowledgeBase:
    """
    Knowledge base for the ElderCare Assistant using RAG (Retrieval-Augmented Generation).
    Stores and retrieves relevant information for elder care.
    """
    
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
        # Dictionary to store raw documents for fallback
        self.raw_documents = {}
        
    async def initialize(self, use_local_embeddings: bool = False):
        """
        Initialize the knowledge base with documents from the knowledge directory.
        
        Args:
            use_local_embeddings: Whether to use local embeddings model instead of OpenAI
        """
        # Create embeddings model
        if use_local_embeddings:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            logger.info("Using local HuggingFace embeddings model")
        else:
            self.embeddings = OpenAIEmbeddings()
            logger.info("Using OpenAI embeddings model")
            
        # Create knowledge base directory if it doesn't exist
        if not os.path.exists(self.knowledge_dir):
            os.makedirs(self.knowledge_dir)
            # Create sample documents if directory is empty
            self._create_sample_documents()
            
        # Load documents from knowledge base directory
        try:
            # First load all documents and store raw content for fallback
            self._load_raw_documents()
            
            # Now process for vector store
            loader = DirectoryLoader(self.knowledge_dir, glob="**/*.txt")
            documents = loader.load()
            
            if not documents:
                logger.warning("No documents found in knowledge base. Creating samples.")
                self._create_sample_documents()
                documents = loader.load()
                
            # Split documents into smaller chunks for better retrieval
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,  # Reduced chunk size for more precise retrieval
                chunk_overlap=100
            )
            splits = text_splitter.split_documents(documents)
            
            # Create vector store
            self.vector_store = FAISS.from_documents(splits, self.embeddings)
            
            # Create base retriever with increased k value
            base_retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}  # Increased from 3 to 5 for better recall
            )
            
            # Enhance retriever with LLM-based compression for better context selection
            if not use_local_embeddings:  # Only use compressor with OpenAI
                llm = ChatOpenAI(temperature=0)
                compressor = LLMChainExtractor.from_llm(llm)
                self.retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=base_retriever
                )
            else:
                # For local embeddings, just use the base retriever
                self.retriever = base_retriever
            
            logger.info(f"Knowledge base initialized with {len(splits)} document chunks")
            
        except Exception as e:
            logger.error(f"Error initializing knowledge base: {str(e)}")
            # Fallback to simple retriever
            if self.vector_store:
                self.retriever = self.vector_store.as_retriever()

    def _load_raw_documents(self):
        """Load and store raw documents for keyword fallback retrieval."""
        try:
            for root, dirs, files in os.walk(self.knowledge_dir):
                for file in files:
                    if file.endswith('.txt'):
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r') as f:
                            content = f.read()
                            
                        # Store in dictionary with filename as key
                        self.raw_documents[file] = {
                            "content": content,
                            "path": file_path,
                            "category": os.path.basename(root)
                        }
            
            logger.info(f"Loaded {len(self.raw_documents)} raw documents for fallback retrieval")
        
        except Exception as e:
            logger.error(f"Error loading raw documents: {str(e)}")
                
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
    
    def _keyword_retrieval(self, query: str) -> List[str]:
        """
        Simple keyword-based retrieval as fallback method.
        
        Args:
            query: The query string
            
        Returns:
            List[str]: List of relevant document contents
        """
        # Extract keywords (simplified approach)
        keywords = [word.lower() for word in query.split() if len(word) > 3]
        
        relevant_docs = []
        for filename, doc_info in self.raw_documents.items():
            content = doc_info["content"].lower()
            
            # Check if any keywords appear in the document
            relevance_score = sum(1 for keyword in keywords if keyword in content)
            
            if relevance_score > 0:
                relevant_docs.append((doc_info["content"], relevance_score))
        
        # Sort by relevance score and return contents
        relevant_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc[0] for doc in relevant_docs[:2]]  # Return top 2 most relevant
                
    async def retrieve_relevant_context(self, query: str, limit: int = 3, use_local_embeddings: bool = False) -> str:
        """
        Retrieve relevant information from the knowledge base.
        
        Args:
            query: The query to search for
            limit: Maximum number of documents to retrieve
            use_local_embeddings: Whether to use local embeddings
            
        Returns:
            str: Combined relevant information
        """
        if not self.retriever:
            logger.warning("Knowledge base not initialized. Initializing now.")
            await self.initialize(use_local_embeddings=use_local_embeddings)
        
        try:
            # First try vector retrieval
            documents = self.retriever.get_relevant_documents(query)
            documents = documents[:limit]  # Limit number of documents
            
            if not documents:
                # Fall back to keyword search if vector retrieval fails or returns empty
                logger.info("Vector retrieval returned no results, falling back to keyword search")
                raw_docs = self._keyword_retrieval(query)
                if raw_docs:
                    return "\n\n-----\n\n".join(raw_docs)
                return ""
            
            # Process retrieved documents to extract the most relevant sections
            extracted_contexts = []
            
            for doc in documents:
                content = doc.page_content
                
                # If using local embeddings, try to extract more specific paragraphs
                if use_local_embeddings:
                    # Split content into paragraphs
                    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
                    
                    # Look for paragraphs containing keywords from the query
                    keywords = [word.lower() for word in query.split() if len(word) > 3]
                    
                    relevant_paragraphs = []
                    for para in paragraphs:
                        if any(keyword in para.lower() for keyword in keywords):
                            relevant_paragraphs.append(para)
                    
                    # If found relevant paragraphs, use them instead of full content
                    if relevant_paragraphs:
                        content = '\n\n'.join(relevant_paragraphs)
                
                extracted_contexts.append(content)
            
            # Combine document contents with separators for clarity
            context = "\n\n-----\n\n".join(extracted_contexts)
            return context
            
        except Exception as e:
            logger.error(f"Error retrieving from knowledge base: {str(e)}")
            
            # Attempt keyword fallback on exception
            try:
                raw_docs = self._keyword_retrieval(query)
                if raw_docs:
                    return "\n\n-----\n\n".join(raw_docs)
            except Exception as inner_e:
                logger.error(f"Keyword fallback also failed: {str(inner_e)}")
                
            return ""

class LLMProvider:
    """
    Provides access to different LLM options with fallback capabilities and caching.
    """
    
    def __init__(self):
        """Initialize the LLM provider with configuration."""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
        self.local_model_url = os.getenv("LOCAL_MODEL_URL", "http://localhost:8000/v1")
        
        # Default model settings
        self.default_model = os.getenv("DEFAULT_LLM_MODEL", "gpt-3.5-turbo")  # Changed to 3.5-turbo to reduce costs
        self.fallback_model = os.getenv("FALLBACK_LLM_MODEL", "gpt-3.5-turbo-16k")
        self.local_model_enabled = os.getenv("USE_LOCAL_MODEL", "false").lower() == "true"
        
        # Create response cache
        self.cache = ResponseCache()
        
        # Track token usage
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        
    async def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.7, 
        max_tokens: int = 300,
        use_cache: bool = True
    ) -> Tuple[str, str]:
        """
        Generate a response using the preferred LLM with fallback options and caching.
        
        Args:
            messages: List of message dictionaries
            temperature: Temperature for response generation
            max_tokens: Maximum tokens in response
            use_cache: Whether to use response cache
            
        Returns:
            Tuple[str, str]: (response text, model used)
        """
        # Check cache first if enabled
        if use_cache:
            cached_response = self.cache.get(messages)
            if cached_response:
                logger.info("Using cached response")
                return cached_response, "cache"
        
        # Try primary model (OpenAI preferred)
        try:
            if self.openai_api_key:
                logger.info(f"Generating response with {self.default_model}")
                response = await openai.ChatCompletion.acreate(
                    model=self.default_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )
                
                # Update token usage
                if hasattr(response, 'usage'):
                    self.token_usage["prompt_tokens"] += response.usage.prompt_tokens
                    self.token_usage["completion_tokens"] += response.usage.completion_tokens
                    self.token_usage["total_tokens"] += response.usage.total_tokens
                    
                    # Log token usage
                    logger.info(f"Token usage: {response.usage.total_tokens} tokens")
                
                result = response.choices[0].message.content
                
                # Cache the result
                if use_cache:
                    self.cache.set(messages, result)
                    
                return result, self.default_model
                
        except Exception as e:
            logger.warning(f"Error with primary model: {str(e)}")
            
        # Try fallback OpenAI model
        try:
            if self.openai_api_key:
                logger.info(f"Falling back to {self.fallback_model}")
                response = await openai.ChatCompletion.acreate(
                    model=self.fallback_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )
                
                # Update token usage
                if hasattr(response, 'usage'):
                    self.token_usage["prompt_tokens"] += response.usage.prompt_tokens
                    self.token_usage["completion_tokens"] += response.usage.completion_tokens
                    self.token_usage["total_tokens"] += response.usage.total_tokens
                
                result = response.choices[0].message.content
                
                # Cache the result
                if use_cache:
                    self.cache.set(messages, result)
                    
                return result, self.fallback_model
                
        except Exception as e:
            logger.warning(f"Error with fallback model: {str(e)}")
            
        # Try local model if enabled
        if self.local_model_enabled:
            try:
                logger.info("Falling back to local model")
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.local_model_url}/chat/completions",
                        json={
                            "model": "local-model",
                            "messages": messages,
                            "temperature": temperature,
                            "max_tokens": max_tokens
                        },
                        headers={"Content-Type": "application/json"}
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            response_text = result["choices"][0]["message"]["content"]
                            
                            # Cache the result
                            if use_cache:
                                self.cache.set(messages, response_text)
                                
                            return response_text, "local-model"
            except Exception as e:
                logger.warning(f"Error with local model: {str(e)}")
        
        # Use basic template-based responses as final fallback
        try:
            fallback_response = self._generate_template_response(messages[-1]["content"])
            return fallback_response, "template-fallback"
        except Exception as e:
            logger.error(f"All response generation methods failed: {str(e)}")
                
        # Last resort - return error message
        return "I'm sorry, I'm having trouble generating a response right now. Could you try again in a moment?", "none"
    
    def _generate_template_response(self, user_message: str) -> str:
        """Generate a simple template-based response as ultimate fallback."""
        message = user_message.lower()
        
        # Basic templates
        if any(word in message for word in ["hello", "hi", "hey", "greetings"]):
            return "Hello! It's nice to talk with you. How can I help you today?"
            
        if any(word in message for word in ["thank", "thanks"]):
            return "You're very welcome! Is there anything else I can help you with?"
            
        if any(word in message for word in ["bye", "goodbye"]):
            return "Goodbye! It was nice chatting with you. Have a wonderful day!"
            
        if any(word in message for word in ["help", "assist"]):
            return "I'd be happy to help you. Could you please tell me a bit more about what you need assistance with?"
            
        if any(word in message for word in ["phone", "smartphone", "app"]):
            return "Technology can sometimes be tricky. Would you like me to explain how to use a specific app or function on your phone?"
            
        if any(word in message for word in ["health", "pain", "doctor"]):
            return "Your health is important. While I can offer general wellness tips, it's always best to consult with your healthcare provider for specific medical advice."
            
        # Default response
        return "I'm here to help with technology questions, provide entertainment recommendations, or just chat. What would you like to talk about today?"

class ElderCareAssistant:
    """
    Core NLP service for the ElderCare Assistant.
    Handles conversation management, personalization, and domain-specific responses.
    Enhanced with RAG (Retrieval-Augmented Generation) capabilities.
    """
    
    def __init__(self, user_profile: Optional[Dict[str, Any]] = None):
        """
        Initialize the ElderCare Assistant.
        
        Args:
            user_profile: Optional dictionary containing user profile information
        """
        self.user_profile = user_profile or {}
        self.conversation_history = []
        self.max_history_length = 10
        self.knowledge_base = ElderCareKnowledgeBase()
        self.llm_provider = LLMProvider()
        self.use_local_embeddings = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"
        
        # Define the base system prompt
        self.base_system_prompt = """
        You are ElderCare, a friendly and patient assistant designed specifically for elderly users.
        
        CORE PERSONALITY TRAITS:
        - Speak in a warm, friendly, and respectful manner
        - Use clear and simple language, avoiding technical jargon
        - Be patient and offer to repeat or clarify when needed
        - Show empathy and emotional intelligence
        - Maintain a conversational, natural tone like a trusted friend
        
        GUIDANCE AREAS:
        1. Technology assistance: Help navigate phones/apps with simple step-by-step instructions
        2. Companionship: Engage in friendly conversation, remember personal details
        3. Entertainment: Recommend books, music, movies based on preferences
        4. Wellbeing: Provide basic health advice while encouraging professional consultation
        
        COMMUNICATION GUIDELINES:
        - Keep responses concise (2-3 sentences when possible)
        - Use slightly larger than average font equivalent in speech
        - Speak at a measured pace with clear articulation
        - Avoid complex sentence structures
        - Use familiar terminology and examples
        
        SAFETY GUIDELINES:
        - For health concerns, provide general wellness advice but encourage consulting healthcare professionals
        - Never advise on medication dosages or specific medical treatments
        - For emergency situations, suggest contacting emergency services or a caregiver
        
        IMPORTANT INSTRUCTION:
        When given information from a knowledge base, extract only the SPECIFIC relevant information that answers
        the user's question. Do NOT repeat the entire document content. Focus on giving a concise, helpful answer
        based on the retrieved information.
        """
        
    async def initialize(self):
        """Initialize the assistant's components."""
        await self.knowledge_base.initialize(use_local_embeddings=self.use_local_embeddings)

    def _get_personalized_system_prompt(self) -> str:
        """
        Create a personalized system prompt based on user profile.
        
        Returns:
            str: The personalized system prompt
        """
        system_prompt = self.base_system_prompt
        
        # Add personalization if user profile exists
        if self.user_profile:
            personalization = "\n\nUSER INFORMATION:\n"
            
            if self.user_profile.get("name"):
                personalization += f"- Name: {self.user_profile['name']}\n"
            
            if self.user_profile.get("age"):
                personalization += f"- Age: {self.user_profile['age']}\n"
                
            if self.user_profile.get("interests"):
                interests = ", ".join(self.user_profile["interests"])
                personalization += f"- Interests: {interests}\n"
                
            if self.user_profile.get("health_conditions"):
                conditions = ", ".join(self.user_profile["health_conditions"])
                personalization += f"- Health considerations: {conditions}\n"
                
            if self.user_profile.get("tech_comfort"):
                personalization += f"- Technology comfort level: {self.user_profile['tech_comfort']}\n"
                
            system_prompt += personalization
            
        return system_prompt
        
    async def process_message(self, user_message: str) -> str:
        """
        Process a user message and generate a response using RAG.
        
        Args:
            user_message: The message from the user
            
        Returns:
            str: The assistant's response
        """
        # Extract intent to determine if knowledge retrieval is needed
        intent_info = self.extract_intent(user_message)
        
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        
        # Trim history if needed
        if len(self.conversation_history) > self.max_history_length * 2:
            # Keep first message for context and the most recent messages
            self.conversation_history = [self.conversation_history[0]] + self.conversation_history[-self.max_history_length*2+1:]
        
        # Prepare base system prompt
        system_prompt = self._get_personalized_system_prompt()
        
        # Retrieve relevant context based on intent
        relevant_context = ""
        if intent_info["intent"] in ["tech_help", "entertainment", "health_advice"]:
            relevant_context = await self.knowledge_base.retrieve_relevant_context(
                user_message,
                use_local_embeddings=self.use_local_embeddings
            )
            
        # Add context to system prompt if available
        if relevant_context:
            system_prompt += f"\n\nRELEVANT INFORMATION:\n{relevant_context}\n\nUse the above information to help answer the user's question when relevant. Extract only the specific information needed, not the entire document."
        
        # Determine if we should use full OpenAI capabilities or conserve tokens
        use_full_context = intent_info["intent"] in ["tech_help", "health_advice"]
        
        # Prepare messages for the API call
        if use_full_context:
            # Use full conversation history for important topics
            messages = [
                {"role": "system", "content": system_prompt}
            ] + self.conversation_history[-5:]  # Limit to last 5 messages
        else:
            # Use just the current query for less critical topics
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        
        try:
            # Use cached responses for common queries
            use_cache = intent_info["intent"] in ["greeting", "conversation", "entertainment"]
            
            # Generate response using LLM provider with fallback capabilities
            response, model_used = await self.llm_provider.generate_response(
                messages,
                temperature=0.7,
                max_tokens=200,  # Reduced from 300 to conserve tokens
                use_cache=use_cache
            )
            
            # Log which model was used
            logger.info(f"Response generated using model: {model_used}")
            
            # Post-process response for elderly users (simplify if needed)
            if model_used != "template-fallback" and len(response) > 500:
                # If response is too long, try to shorten it
                shorten_messages = [
                    {"role": "system", "content": "You are an assistant that makes text more concise while maintaining the key information. Keep your response under 3-4 sentences."},
                    {"role": "user", "content": f"Please shorten this response while keeping the essential information: {response}"}
                ]
                
                try:
                    shortened_response, _ = await self.llm_provider.generate_response(
                        shorten_messages,
                        temperature=0.3,
                        max_tokens=150,
                        use_cache=True
                    )
                    response = shortened_response
                except Exception as e:
                    logger.warning(f"Failed to shorten response: {str(e)}")
            
            # Add the response to the conversation history
            self.conversation_history.append({"role": "assistant", "content": response})
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I'm sorry, I'm having trouble responding right now. Could you try again in a moment?"
    
    async def get_tech_help(self, app_name: str) -> str:
        """
        Get technology help for a specific application using RAG.
        
        Args:
            app_name: Name of the application
            
        Returns:
            str: Step-by-step guidance for using the
        """