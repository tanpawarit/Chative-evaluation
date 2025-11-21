import os
import sys
from dotenv import load_dotenv

# Add src to python path to allow imports from src/agents
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))



def main():
    # Load environment variables (e.g. OPENAI_API_KEY)
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found in environment or .env file.")
        print("Please create a .env file with OPENAI_API_KEY=sk-...")
        return

    from agents.intent_agent.core import agent

    print("Intent Agent loaded.")
    print("Using mock intents: greet:0.30, book_flight:0.90, cancel_flight:0.70")
    print("Enter text to analyze (or 'q' to quit):")
    
    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() in ('q', 'quit', 'exit'):
                break
            
            if not user_input.strip():
                continue

            print("Analyzing...")
            result = agent.invoke({"input": user_input})
            print("\nAgent Output:")
            print(result)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
