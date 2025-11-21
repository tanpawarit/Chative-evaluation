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

    try:
        from agents.response_agent.core import agent
    except ImportError as e:
        print(f"Error importing agent: {e}")
        print("Make sure you are running this script from the project root.")
        return

    print("Response Agent loaded.")
    print("Enter text to generate a response (or 'q' to quit):")
    
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
            # Print the output string if available, otherwise the whole result
            if isinstance(result, dict) and "output" in result:
                print(result["output"])
            else:
                print(result)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
