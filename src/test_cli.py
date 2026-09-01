"""
test_cli.py
Quick CLI test — run a conversation without Streamlit.
Usage: python test_cli.py
"""

from dotenv import load_dotenv
load_dotenv()

import database
from state import initial_state
from graph import run_turn


def main():
    print("Loading product database...")
    database.load_all()
    print("\n" + "="*60)
    print("  Conversational Recommender — CLI Test")
    print("  Type 'quit' to exit, 'reset' to start over")
    print("="*60 + "\n")

    state = initial_state()
    print("Assistant: Hi! I can help you find a smartphone or pair of headphones. What are you looking for?\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "reset":
            state = initial_state()
            print("Assistant: Starting over! What can I help you find?\n")
            continue

        state = run_turn(state, user_input)

        print(f"\nAssistant: {state['response']}\n")
        print(f"  [intent={state['intent']} | action={state['action']} | "
              f"category={state['category']} | filters={state['active_filters']} | "
              f"candidates={len(state['candidates'])}]\n")

        if state["action"] == "done":
            break


if __name__ == "__main__":
    main()