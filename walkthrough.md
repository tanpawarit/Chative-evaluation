# Intent Agent Script Setup

I have configured `script_intent_agent.py` to run the Intent Agent interactively.

## Changes
- Modified `script_intent_agent.py` to:
    - Load environment variables using `dotenv`.
    - Import the `agent` from `src.agents.intent_agent.core`.
    - Run an interactive loop to accept user input and display the agent's analysis.

## Verification
I ran the script and tested it with the input "Hello".

### Output
```
Intent Agent loaded.
Using mock intents: greet:0.30, book_flight:0.90, cancel_flight:0.70
Enter text to analyze (or 'q' to quit):

User: Hello
Analyzing...

Agent Output:
(intent<||>greet<||>0.95<||>0.30)##
(language<||>eng<||>1.00<||>1)##
(sentiment<||>positive<||>0.80)##
<|COMPLETED|>
```

## How to Run
1. Ensure you have a `.env` file with `OPENAI_API_KEY`.
2. Run the script:
   ```bash
   python script_intent_agent.py
   ```
3. Enter text to analyze.
4. Type `q` to exit.
