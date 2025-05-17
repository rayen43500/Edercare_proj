import os
import logging
import asyncio
import subprocess
import tempfile
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GitModelProvider:
    """
    Provider for integrating a machine learning model from a Git repository.
    This allows the ElderCare Assistant to use custom models shared via Git.
    """
    
    def __init__(self, repo_url: Optional[str] = None, model_path: Optional[str] = None):
        """
        Initialize the Git Model provider.
        
        Args:
            repo_url: URL of the Git repository containing the model
            model_path: Path within the repository to the model files
        """
        self.repo_url = repo_url
        self.model_path = model_path
        self.model_dir = os.path.join(os.path.dirname(__file__), "git_models")
        self.model = None
        self.tokenizer = None
        
    async def initialize(self) -> bool:
        """
        Initialize by cloning the Git repository and loading the model.
        
        Returns:
            bool: Success status
        """
        if not self.repo_url:
            logger.error("Cannot initialize: No Git repository URL provided")
            return False
            
        try:
            # Create model directory if it doesn't exist
            os.makedirs(self.model_dir, exist_ok=True)
            
            # Generate a directory name based on repo URL
            repo_name = self.repo_url.split("/")[-1].split(".")[0]
            repo_dir = os.path.join(self.model_dir, repo_name)
            
            # Clone or pull the repository
            if os.path.exists(repo_dir):
                logger.info(f"Updating existing repository at {repo_dir}")
                result = subprocess.run(
                    ["git", "-C", repo_dir, "pull"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                logger.info(f"Git pull result: {result.stdout}")
            else:
                logger.info(f"Cloning repository to {repo_dir}")
                result = subprocess.run(
                    ["git", "clone", self.repo_url, repo_dir],
                    capture_output=True,
                    text=True,
                    check=True
                )
                logger.info(f"Git clone result: {result.stdout}")
            
            # Set the full model path
            full_model_path = repo_dir
            if self.model_path:
                full_model_path = os.path.join(repo_dir, self.model_path)
            
            # Load the model
            await self._load_model(full_model_path)
            
            logger.info(f"Git model initialized from {self.repo_url}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Git command failed: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Error initializing Git model: {str(e)}")
            return False
            
    async def _load_model(self, model_path: str) -> None:
        """
        Load the model from the specified path.
        
        Args:
            model_path: Path to the model files
        """
        try:
            # Import transformers only when needed to avoid startup delays
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            logger.info(f"Loading model from {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",  # Will use GPU if available
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    async def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.7, 
        max_tokens: Optional[int] = None
    ) -> Tuple[str, str]:
        """
        Generate a response using the Git model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Temperature for response generation
            max_tokens: Maximum tokens in response (optional)
            
        Returns:
            Tuple[str, str]: (response text, model source identifier)
        """
        if not self.model or not self.tokenizer:
            if not await self.initialize():
                return "Je ne peux pas répondre maintenant car le modèle n'est pas disponible.", "none"
        
        try:
            # Format the conversation history for the model
            prompt = self._format_messages(messages)
            
            # Run in executor to prevent blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._generate_text(prompt, temperature, max_tokens)
            )
            
            return result, "git_model"
            
        except Exception as e:
            logger.error(f"Error generating response with Git model: {str(e)}")
            return f"Désolé, j'ai rencontré une erreur: {str(e)}", "error"
    
    def _generate_text(
        self, 
        prompt: str, 
        temperature: float = 0.7, 
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text using the loaded model.
        
        Args:
            prompt: The text prompt
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            str: Generated text
        """
        try:
            # Set default max_tokens if not specified
            if not max_tokens:
                max_tokens = 512
                
            # Tokenize the prompt
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            # Generate the response
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.95
            )
            
            # Decode the response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the original prompt from the response
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
                
            return response
            
        except Exception as e:
            logger.error(f"Error in text generation: {str(e)}")
            raise
            
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Format the conversation history into a prompt for the model.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            str: Formatted prompt
        """
        formatted_prompt = ""
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                formatted_prompt += f"System: {content}\n\n"
            elif role == "user":
                formatted_prompt += f"User: {content}\n\n"
            elif role == "assistant":
                formatted_prompt += f"Assistant: {content}\n\n"
                
        # Add final assistant prefix to prompt completion
        formatted_prompt += "Assistant: "
        
        return formatted_prompt

    def set_repository(self, repo_url: str, model_path: Optional[str] = None) -> bool:
        """
        Update the Git repository URL and model path.
        
        Args:
            repo_url: URL of the Git repository containing the model
            model_path: Path within the repository to the model files
            
        Returns:
            bool: Success status
        """
        self.repo_url = repo_url
        self.model_path = model_path
        
        # Reset the model
        self.model = None
        self.tokenizer = None
        
        return True


# Example usage and testing
async def test_git_model():
    """Test function for the Git Model provider."""
    # Example repo URL pointing to a Hugging Face model
    repo_url = "https://github.com/user/model-repo.git"
    
    provider = GitModelProvider(repo_url=repo_url)
    
    if not await provider.initialize():
        print("Failed to initialize Git model.")
        return
    
    messages = [
        {"role": "system", "content": "Vous êtes un assistant pour personnes âgées, serviable et sympathique."},
        {"role": "user", "content": "Quels exercices sont bons pour l'arthrite?"}
    ]
    
    response, model = await provider.generate_response(messages)
    print(f"Response using {model}:\n{response}")


if __name__ == "__main__":
    asyncio.run(test_git_model()) 