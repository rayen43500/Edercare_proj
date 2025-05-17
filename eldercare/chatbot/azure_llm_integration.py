import os
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AzureLLMProvider:
    """
    Provider for Azure AI Inference SDK to allow the ElderCare Assistant
    to use advanced language models like GPT-4.1 through GitHub's AI service.
    """
    
    def __init__(self):
        """Initialize the Azure LLM provider."""
        self.endpoint = "https://models.github.ai/inference"
        self.model = "openai/gpt-4.1"
        
        # Get token from environment variables
        self.token = os.environ.get("GITHUB_TOKEN")
        if not self.token:
            logger.warning("GITHUB_TOKEN environment variable not set. Please set it to use Azure LLM.")
        
        self.client = None
        
    def initialize(self):
        """Initialize the client if token is available."""
        if not self.token:
            logger.error("Cannot initialize client: No GITHUB_TOKEN provided")
            return False
            
        try:
            self.client = ChatCompletionsClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(self.token),
            )
            logger.info(f"Azure LLM client initialized with model {self.model}")
            return True
        except Exception as e:
            logger.error(f"Error initializing Azure LLM client: {str(e)}")
            return False
    
    async def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.7, 
        max_tokens: Optional[int] = None
    ) -> Tuple[str, str]:
        """
        Generate a response using the Azure LLM.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Temperature for response generation
            max_tokens: Maximum tokens in response (optional)
            
        Returns:
            Tuple[str, str]: (response text, model used)
        """
        if not self.client:
            if not self.initialize():
                return "Je ne peux pas répondre maintenant car je n'ai pas accès au service d'IA.", "none"
        
        try:
            # Convert message format from OpenAI style to Azure AI style
            azure_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    azure_messages.append(SystemMessage(msg["content"]))
                elif msg["role"] == "user":
                    azure_messages.append(UserMessage(msg["content"]))
                # Assistant messages are handled differently in Azure AI SDK
            
            # Create completion parameters
            params = {
                "messages": azure_messages,
                "temperature": temperature,
                "top_p": 1.0,
                "model": self.model
            }
            
            if max_tokens:
                params["max_tokens"] = max_tokens
            
            # Run in executor to prevent blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.complete(**params)
            )
            
            result = response.choices[0].message.content
            return result, self.model
            
        except Exception as e:
            logger.error(f"Error generating response with Azure LLM: {str(e)}")
            return f"Désolé, j'ai rencontré une erreur: {str(e)}", "error"
    
    def set_token(self, token: str) -> bool:
        """
        Set the GitHub token for authentication.
        
        Args:
            token: GitHub personal access token with models:read permission
            
        Returns:
            bool: Success status
        """
        self.token = token
        return self.initialize()


# Example usage and testing
async def test_azure_llm():
    """Test function for the Azure LLM provider."""
    provider = AzureLLMProvider()
    
    # Set the GitHub token (you could also set it as an environment variable)
    # provider.set_token("your-github-token")
    
    if not provider.initialize():
        print("Failed to initialize. Make sure GITHUB_TOKEN is set.")
        return
    
    messages = [
        {"role": "system", "content": "Vous êtes un assistant pour personnes âgées, serviable et sympathique."},
        {"role": "user", "content": "Quels exercices sont bons pour l'arthrite?"}
    ]
    
    response, model = await provider.generate_response(messages)
    print(f"Response using {model}:\n{response}")


if __name__ == "__main__":
    asyncio.run(test_azure_llm()) 