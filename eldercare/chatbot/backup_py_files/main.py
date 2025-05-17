import os
import json
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import asyncio
import aiohttp
import openai
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
        else:
            self.embeddings = OpenAIEmbeddings()
            
        # Create knowledge base directory if it doesn't exist
        if not os.path.exists(self.knowledge_dir):
            os.makedirs(self.knowledge_dir)
            # Create sample documents if directory is empty
            self._create_sample_documents()
            
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
            
            # Create base retriever
            base_retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            
            # Enhance retriever with LLM-based compression for better context selection
            llm = ChatOpenAI(temperature=0)
            compressor = LLMChainExtractor.from_llm(llm)
            self.retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever
            )
            
            logger.info(f"Knowledge base initialized with {len(splits)} document chunks")
            
        except Exception as e:
            logger.error(f"Error initializing knowledge base: {str(e)}")
            # Fallback to simple retriever
            if self.vector_store:
                self.retriever = self.vector_store.as_retriever()
                
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
                
    async def retrieve_relevant_context(self, query: str, limit: int = 3) -> str:
        """
        Retrieve relevant information from the knowledge base.
        
        Args:
            query: The query to search for
            limit: Maximum number of documents to retrieve
            
        Returns:
            str: Combined relevant information
        """
        if not self.retriever:
            logger.warning("Knowledge base not initialized. Initializing now.")
            await self.initialize()
            
        try:
            documents = self.retriever.get_relevant_documents(query)
            documents = documents[:limit]  # Limit number of documents
            
            if not documents:
                return ""
                
            # Combine document contents
            context = "\n\n".join([doc.page_content for doc in documents])
            return context
            
        except Exception as e:
            logger.error(f"Error retrieving from knowledge base: {str(e)}")
            return ""

class LLMProvider:
    """
    Provides access to different LLM options with fallback capabilities.
    """
    
    def __init__(self):
        """Initialize the LLM provider with configuration."""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
        self.local_model_url = os.getenv("LOCAL_MODEL_URL", "http://localhost:8000/v1")
        
        # Default model settings
        self.default_model = os.getenv("DEFAULT_LLM_MODEL", "gpt-4")
        self.fallback_model = os.getenv("FALLBACK_LLM_MODEL", "gpt-3.5-turbo")
        self.local_model_enabled = os.getenv("USE_LOCAL_MODEL", "false").lower() == "true"
        
    async def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.7, 
        max_tokens: int = 300
    ) -> Tuple[str, str]:
        """
        Generate a response using the preferred LLM with fallback options.
        
        Args:
            messages: List of message dictionaries
            temperature: Temperature for response generation
            max_tokens: Maximum tokens in response
            
        Returns:
            Tuple[str, str]: (response text, model used)
        """
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
                return response.choices[0].message.content, self.default_model
                
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
                return response.choices[0].message.content, self.fallback_model
                
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
                            return result["choices"][0]["message"]["content"], "local-model"
            except Exception as e:
                logger.warning(f"Error with local model: {str(e)}")
                
        # Last resort - return error message
        return "I'm sorry, I'm having trouble generating a response right now. Could you try again in a moment?", "none"

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
        """
        
    async def initialize(self):
        """Initialize the assistant's components."""
        await self.knowledge_base.initialize()

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
            relevant_context = await self.knowledge_base.retrieve_relevant_context(user_message)
            
        # Add context to system prompt if available
        if relevant_context:
            system_prompt += f"\n\nRELEVANT INFORMATION:\n{relevant_context}\n\nUse the above information to help answer the user's question when relevant."
        
        # Prepare messages for the API call
        messages = [
            {"role": "system", "content": system_prompt}
        ] + self.conversation_history
        
        try:
            # Generate response using LLM provider with fallback capabilities
            response, model_used = await self.llm_provider.generate_response(messages)
            
            # Log which model was used
            logger.info(f"Response generated using model: {model_used}")
            
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
            str: Step-by-step guidance for using the app
        """
        # Retrieve relevant tech help information
        search_query = f"how to use {app_name} app for seniors"
        relevant_context = await self.knowledge_base.retrieve_relevant_context(search_query)
        
        prompt = f"""
        Please provide simple step-by-step instructions for an elderly person to use the {app_name} app on their smartphone.
        Instructions should be very clear, simple, and assume minimal technical knowledge.
        Include how to find the app on their phone and basic usage.
        Keep it under 5 steps if possible.
        """
        
        # Add context if available
        if relevant_context:
            prompt = f"""
            Here is some information about using {app_name}:
            
            {relevant_context}
            
            Based on this information and your knowledge, {prompt}
            """
        
        messages = [
            {"role": "system", "content": self._get_personalized_system_prompt()},
            {"role": "user", "content": prompt}
        ]
        
        # Generate response with fallback options
        response, _ = await self.llm_provider.generate_response(messages)
        return response
    
    async def recommend_entertainment(self, category: str, preferences: Optional[List[str]] = None) -> str:
        """
        Generate entertainment recommendations based on category and preferences using RAG.
        
        Args:
            category: Type of entertainment (books, music, movies, etc.)
            preferences: Optional list of user preferences
            
        Returns:
            str: Personalized recommendations
        """
        prefs = ""
        if preferences:
            prefs = f"They enjoy {', '.join(preferences)}. "
            
        tech_level = self.user_profile.get("tech_comfort", "beginner")
        
        # Build search query based on preferences
        search_query = f"{category} recommendations for seniors"
        if preferences:
            search_query += f" who like {', '.join(preferences)}"
            
        # Retrieve relevant entertainment recommendations
        relevant_context = await self.knowledge_base.retrieve_relevant_context(search_query)
        
        prompt = f"""
        Please recommend 3 {category} for an elderly person. {prefs}
        Their technology comfort level is {tech_level}.
        For each recommendation, include:
        1. Title
        2. Very brief description (1 sentence)
        3. Why they might enjoy it
        4. Simple way to access it
        
        Keep the overall response brief and friendly.
        """
        
        # Add context if available
        if relevant_context:
            prompt = f"""
            Here are some {category} recommendations that might be relevant:
            
            {relevant_context}
            
            Based on this information and your knowledge, {prompt}
            """
        
        messages = [
            {"role": "system", "content": self._get_personalized_system_prompt()},
            {"role": "user", "content": prompt}
        ]
        
        # Generate response with fallback options
        response, _ = await self.llm_provider.generate_response(messages)
        return response
    
    async def provide_health_tip(self, concern: Optional[str] = None) -> str:
        """
        Provide general health and wellness advice using RAG.
        
        Args:
            concern: Optional specific health concern
            
        Returns:
            str: General wellness advice
        """
        # Build search query based on concern
        search_query = "health tips for seniors"
        if concern:
            search_query = f"{concern} advice for elderly"
            
        # Retrieve relevant health information
        relevant_context = await self.knowledge_base.retrieve_relevant_context(search_query)
        
        if concern:
            prompt = f"""
            Provide gentle, general wellness advice related to '{concern}' for an elderly person.
            Include:
            1. Simple explanation
            2. General wellness tip (nothing that replaces medical advice)
            3. When they should consider speaking to a healthcare professional
            
            Keep it brief, supportive, and non-alarming.
            """
        else:
            prompt = """
            Share a helpful wellness tip appropriate for elderly users.
            Focus on general wellbeing, simple exercises, nutrition, or mental health.
            Keep it positive, achievable, and brief.
            """
            
        # Add context if available
        if relevant_context:
            prompt = f"""
            Here is some health information that might be relevant:
            
            {relevant_context}
            
            Based on this information and your knowledge, {prompt}
            """
        
        messages = [
            {"role": "system", "content": self._get_personalized_system_prompt()},
            {"role": "user", "content": prompt}
        ]
        
        # Generate response with fallback options
        response, _ = await self.llm_provider.generate_response(messages)
        return response

    def update_user_profile(self, new_information: Dict[str, Any]) -> None:
        """
        Update the user profile with new information.
        
        Args:
            new_information: Dictionary with profile information to update
        """
        self.user_profile.update(new_information)
        
        # Save profile to disk for persistence
        try:
            os.makedirs("user_profiles", exist_ok=True)
            user_id = self.user_profile.get("id", "default_user")
            
            with open(f"user_profiles/{user_id}.json", "w") as f:
                json.dump(self.user_profile, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving user profile: {str(e)}")
        
    def extract_intent(self, user_message: str) -> Dict[str, Any]:
        """
        Extract the user's intent from their message.
        
        Args:
            user_message: The message from the user
            
        Returns:
            Dict with intent information
        """
        # Basic intent detection - would be expanded with proper NLU in production
        message = user_message.lower()
        
        if any(word in message for word in ["help", "how to", "use", "open", "find", "app"]):
            return {"intent": "tech_help", "confidence": 0.8}
            
        if any(word in message for word in ["recommend", "book", "movie", "music", "watch", "read", "listen"]):
            return {"intent": "entertainment", "confidence": 0.8}
            
        if any(word in message for word in ["health", "pain", "hurt", "feel", "doctor", "medicine"]):
            return {"intent": "health_advice", "confidence": 0.8}
            
        if any(word in message for word in ["hello", "hi", "hey", "good morning", "good afternoon"]):
            return {"intent": "greeting", "confidence": 0.9}
            
        return {"intent": "conversation", "confidence": 0.6}
        
    async def add_to_knowledge_base(self, category: str, content: str) -> bool:
        """
        Add new information to the knowledge base.
        
        Args:
            category: Category for the information (tech, health, entertainment)
            content: The content to add
            
        Returns:
            bool: Success status
        """
        try:
            filename = f"{category}_{int(asyncio.get_event_loop().time())}.txt"
            filepath = os.path.join(self.knowledge_base.knowledge_dir, filename)
            
            with open(filepath, "w") as f:
                f.write(content)
                
            # Reinitialize knowledge base to include new content
            await self.knowledge_base.initialize()
            return True
            
        except Exception as e:
            logger.error(f"Error adding to knowledge base: {str(e)}")
            return False

# Example usage
async def main():
    # Initialize with a sample user profile
    user_profile = {
        "id": "user123",
        "name": "Margaret",
        "age": 76,
        "interests": ["gardening", "mystery novels", "classical music"],
        "health_conditions": ["arthritis", "mild hearing loss"],
        "tech_comfort": "beginner"
    }
    
    # Create and initialize assistant
    assistant = ElderCareAssistant(user_profile)
    await assistant.initialize()
    
    # Add custom knowledge to the knowledge base
    await assistant.add_to_knowledge_base(
        "tech",
        """
        # Tips for Using Video Calling
        
        ## Zoom
        1. Find the blue Zoom icon on your device
        2. Tap it to open
        3. Tap "Join a Meeting" to join someone else's call
        4. Enter the meeting ID number they sent you
        5. Tap "Join" and then "Join with Video"
        
        ## FaceTime (Apple devices)
        1. Find the green FaceTime icon
        2. Tap to open
        3. Tap the "+" button to start a new call
        4. Type the name or number of the person you want to call
        5. Tap the video camera icon for a video call
        """
    )
    
    # Process a sample message
    response = await assistant.process_message("I'm feeling a bit lonely today.")
    print(f"Response: {response}")
    
    # Get tech help using RAG
    tech_help = await assistant.get_tech_help("Zoom")
    print(f"\nTech Help for Zoom: {tech_help}")
    
    # Get entertainment recommendations using RAG
    recommendations = await assistant.recommend_entertainment("books", ["mystery", "history"])
    print(f"\nBook Recommendations: {recommendations}")
    
    # Get health advice using RAG
    health_tip = await assistant.provide_health_tip("arthritis")
    print(f"\nHealth Tip for Arthritis: {health_tip}")

if __name__ == "__main__":
    asyncio.run(main())