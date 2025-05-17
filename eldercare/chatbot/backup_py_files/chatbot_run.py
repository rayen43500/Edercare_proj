import asyncio
import logging
from improved_chatbot import ElderCareAssistant

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def run_chatbot():
    """
    Run the ElderCare Assistant chatbot with a user-friendly interface.
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
        
        print("\n=== Welcome to ElderCare Assistant ===\n")
        print("Initializing the assistant...")
        
        # Create and initialize the assistant
        assistant = ElderCareAssistant(
            user_profile=user_profile,
            knowledge_dir="knowledge_base",
            cache_dir="response_cache",
            max_history=10,
            use_local_model=True  # Start with local model
        )
        
        await assistant.initialize()
        print("Assistant initialized successfully!")
        
        # Display welcome message
        print("\nI'm your ElderCare Assistant. I can help you with:")
        print("1. Technology assistance")
        print("2. Health tips and information")
        print("3. Entertainment recommendations")
        print("4. General conversation and support")
        print("\nType 'help' for more information or 'quit' to exit.")
        print("Type 'switch' to toggle between local and external AI models.")
        
        # Main chat loop
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                # Handle special commands
                if user_input.lower() == 'quit':
                    print("\nGoodbye! Have a wonderful day!")
                    break
                elif user_input.lower() == 'help':
                    print("\nAvailable commands:")
                    print("- help: Show this help message")
                    print("- quit: Exit the program")
                    print("- switch: Toggle between local and external AI models")
                    print("- profile: Show your current profile")
                    print("\nYou can also ask me about:")
                    print("- How to use specific apps or technology")
                    print("- Health tips and wellness advice")
                    print("- Entertainment recommendations")
                    print("- General questions and support")
                    continue
                elif user_input.lower() == 'switch':
                    current_mode = "local" if assistant.use_local_model else "external"
                    assistant.switch_model(not assistant.use_local_model)
                    new_mode = "local" if assistant.use_local_model else "external"
                    print(f"\nSwitched from {current_mode} to {new_mode} model.")
                    continue
                elif user_input.lower() == 'profile':
                    print("\nYour current profile:")
                    for key, value in assistant.user_profile.items():
                        print(f"- {key}: {value}")
                    continue
                
                # Process user message
                response = await assistant.process_message(user_input)
                print(f"\nAssistant: {response}")
                
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
        asyncio.run(run_chatbot())
    except KeyboardInterrupt:
        print("\n\nGoodbye! Have a wonderful day!")
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        print("\nI apologize, but I encountered a serious error. Please restart the application.") 