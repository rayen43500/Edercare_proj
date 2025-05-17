import asyncio
import logging
from improved_chatbot import ElderCareAssistant

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def chat():
    try:
        # Initialize with a default user profile
        user_profile = {
            "id": "user1",
            "name": "User",
            "age": 65,
            "interests": ["reading", "music"],
            "health_conditions": [],
            "tech_comfort": "beginner"
        }
        
        logger.info("Creating ElderCare Assistant...")
        # Create and initialize assistant
        assistant = ElderCareAssistant(user_profile)
        
        logger.info("Initializing assistant...")
        await assistant.initialize()
        
        logger.info("Processing greeting...")
        # Process the greeting
        response = await assistant.process_message("hi")
        print(f"\nAssistant: {response}\n")
        
        # Keep the chat running
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nGoodbye!")
                break
                
            response = await assistant.process_message(user_input)
            print(f"\nAssistant: {response}\n")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(chat()) 