import asyncio
from main import ElderCareKnowledgeBase, LLMProvider

async def main():
    # Initialize the knowledge base
    knowledge_base = ElderCareKnowledgeBase()
    await knowledge_base.initialize(use_local_embeddings=True)
    
    # Initialize the LLM provider
    llm_provider = LLMProvider()
    
    print("Chatbot initialized! Type 'quit' to exit.")
    print("You can ask questions about technology, health, or entertainment.")
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nGoodbye!")
                break
            
            # Get relevant context
            context = await knowledge_base.retrieve_relevant_context(
                user_input,
                use_local_embeddings=True
            )
            
            # Prepare messages
            messages = [
                {"role": "system", "content": "You are a helpful assistant for elderly users."}
            ]
            
            if context:
                messages.append({
                    "role": "system",
                    "content": f"RELEVANT INFORMATION:\n{context}"
                })
            
            messages.append({"role": "user", "content": user_input})
            
            # Generate response
            response, model_used = await llm_provider.generate_response(
                messages,
                temperature=0.7,
                max_tokens=300
            )
            
            print(f"\nAssistant: {response}")
            print(f"(Response generated using: {model_used})")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again.")

if __name__ == "__main__":
    asyncio.run(main()) 