#!/usr/bin/env python3
"""
Generate synthetic conversational training data for KhatriVoice.

This script creates properly formatted conversation pairs with special tokens.

Usage:
    python scripts/generate_conversation_data.py --output data/conversations.txt --count 10000
"""

import argparse
import random
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# Conversation templates - diverse Q&A pairs
GREETINGS = [
    ("Hi", "Hello! How can I help you today?"),
    ("Hello", "Hi there! What can I do for you?"),
    ("Hey", "Hey! Nice to hear from you. How can I assist?"),
    ("Good morning", "Good morning! Hope you're having a great start to the day."),
    ("Good afternoon", "Good afternoon! How can I help you?"),
    ("Good evening", "Good evening! What brings you here today?"),
]

GENERAL_QUESTIONS = [
    ("How are you?", "I'm doing well, thank you for asking! How can I help you?"),
    ("What is your name?", "I'm KhatriVoice, an AI assistant designed to help you."),
    ("Who are you?", "I'm an AI assistant. I'm here to help answer your questions."),
    ("What can you do?", "I can answer questions, provide information, and have conversations with you."),
    ("Are you real?", "I'm an artificial intelligence, not a human. But I'm here to help!"),
]

AI_QUESTIONS = [
    ("What is AI?", "AI stands for Artificial Intelligence. It's computer systems designed to perform tasks that typically require human intelligence, like learning, reasoning, and understanding language."),
    ("What is machine learning?", "Machine learning is a type of AI where computers learn from data to improve at tasks without being explicitly programmed."),
    ("What is deep learning?", "Deep learning uses neural networks with many layers to learn patterns from data. It's especially good for images, speech, and text."),
    ("What is a neural network?", "A neural network is a computer system inspired by the human brain. It consists of connected nodes that process information."),
    ("What is NLP?", "NLP stands for Natural Language Processing. It's how computers understand and work with human language."),
    ("What is a transformer?", "A transformer is a neural network architecture that uses attention mechanisms. It's very effective for processing sequences like text."),
    ("What is GPT?", "GPT stands for Generative Pre-trained Transformer. It's a type of language model trained to generate text."),
]

PYTHON_QUESTIONS = [
    ("What is Python?", "Python is a popular programming language known for its simple syntax and versatility. It's used for web development, data science, automation, and more."),
    ("What is a variable in Python?", "A variable is a name that holds a value. In Python, you create one by using equals: x = 5"),
    ("What is a function?", "A function is a reusable block of code. In Python, you define it using 'def': def my_function():"),
    ("What is a list in Python?", "A list is an ordered collection of items. You create it with brackets: my_list = [1, 2, 3]"),
    ("What is a dictionary?", "A dictionary stores key-value pairs. Use braces: my_dict = {'key': 'value'}"),
    ("How do I print in Python?", "Use the print function: print('Hello World')"),
]

TECHNOLOGY_QUESTIONS = [
    ("What is Python used for?", "Python is used for web development, data analysis, machine learning, automation, scripting, and much more."),
    ("What is programming?", "Programming is writing instructions for computers to follow. These instructions are written in programming languages."),
    ("What is a CPU?", "CPU stands for Central Processing Unit. It's the main processor in a computer that performs calculations."),
    ("What is RAM?", "RAM is Random Access Memory. It's temporary storage that the computer uses for active processes."),
    ("What is an API?", "API stands for Application Programming Interface. It's how software components communicate with each other."),
]

EXPLANATION_REQUESTS = [
    ("Explain machine learning simply", "Machine learning teaches computers to learn from data. Instead of following fixed rules, they improve by seeing examples."),
    ("Explain artificial intelligence", "AI is making computers do things that normally need human intelligence - like understanding language, recognizing images, or making decisions."),
    ("Explain coding for beginners", "Coding is giving step-by-step instructions to a computer. You write commands in a programming language, and the computer follows them."),
    ("Explain what is a program", "A program is a set of instructions that tells a computer what to do. It's like a recipe that the computer follows."),
]

THANK_YOUS = [
    ("Thank you", "You're welcome! Feel free to ask more questions."),
    ("Thanks", "Glad to help! Let me know if you need anything else."),
    ("Thanks for helping", "My pleasure! I'm here whenever you need assistance."),
    ("That helps", "Great! Don't hesitate to ask if you have more questions."),
]

GOODBYES = [
    ("Goodbye", "Goodbye! Have a great day!"),
    ("Bye", "Bye! Take care and come back anytime."),
    ("See you later", "See you! Have a wonderful day."),
    ("Take care", "You too! Feel free to chat anytime."),
]

# Combine all conversations
ALL_CONVERSATIONS = (
    GREETINGS + GENERAL_QUESTIONS + AI_QUESTIONS +
    PYTHON_QUESTIONS + TECHNOLOGY_QUESTIONS +
    EXPLANATION_REQUESTS + THANK_YOUS + GOODBYES
)


def generate_conversation_variations(base_convs: list, num_variations: int = 3) -> list:
    """Generate variations of conversations with different phrasings."""
    variations = []

    for user_msg, assistant_msg in base_convs:
        # Add original
        variations.append((user_msg, assistant_msg))

        # Add lowercase versions
        if user_msg[0].isupper():
            variations.append((user_msg.lower(), assistant_msg))

        # Add question mark variations
        if '?' not in user_msg and not user_msg.endswith('.'):
            variations.append((user_msg + '?', assistant_msg))

        # Add "please" variations
        if 'please' not in user_msg.lower() and len(user_msg) > 5:
            variations.append(("Please " + user_msg.lower(), assistant_msg))

    return variations


def format_conversation_with_tokens(user_text: str, assistant_text: str) -> str:
    """Format conversation with special tokens."""
    # These must match vocabulary.py exactly
    USER_TOKEN = ""
    ASSISTANT_TOKEN = ""
    END_TOKEN = "<|end|>"

    return f"{USER_TOKEN}\n{user_text}\n{ASSISTANT_TOKEN}\n{assistant_text}\n{END_TOKEN}"


def format_conversation_multiline(user_text: str, assistant_text: str) -> str:
    """
    Format conversation in multiline format for easier reading.

    Output format:
        User: What is Python?
        AI: Python is a programming language.
    """
    return f"User: {user_text}\nAI: {assistant_text}"


def generate_conversations(count: int, seed: int = 42, output_format: str = "tokens") -> list:
    """
    Generate a list of formatted conversations.

    Args:
        count: Number of conversations to generate
        seed: Random seed
        output_format: 'tokens' (special tokens), 'multiline' (User:/AI:), or 'both'

    Returns:
        List of formatted conversation strings
    """
    random.seed(seed)

    # Start with base conversations
    all_convs = generate_conversation_variations(ALL_CONVERSATIONS)

    # Repeat to reach count
    conversations = []
    while len(conversations) < count:
        # Pick random conversation
        user_msg, assistant_msg = random.choice(all_convs)

        # Format based on output format
        if output_format == "multiline":
            formatted = format_conversation_multiline(user_msg, assistant_msg)
        elif output_format == "both":
            # Use multiline format but with special tokens inline
            formatted = f"User: {user_msg}\nAI: {assistant_msg}\n<|end|>"
        else:  # tokens
            formatted = format_conversation_with_tokens(user_msg, assistant_msg)

        conversations.append(formatted)

    # Shuffle
    random.shuffle(conversations)

    return conversations[:count]


def main():
    parser = argparse.ArgumentParser(description="Generate conversational training data")
    parser.add_argument("--output", type=str, default="data/conversations.txt",
                        help="Output file path")
    parser.add_argument("--count", type=int, default=10000,
                        help="Number of conversations to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--preview", action="store_true",
                        help="Preview first 5 conversations without saving")
    parser.add_argument("--format", type=str, default="tokens",
                        choices=["tokens", "multiline", "both"],
                        help="Output format: 'tokens' (with special tokens), 'multiline' (User:/AI:), or 'both'")

    args = parser.parse_args()

    print(f"Generating {args.count} conversations...")
    conversations = generate_conversations(args.count, seed=args.seed, output_format=args.format)

    if args.preview:
        print("\n=== Preview (first 5 conversations) ===\n")
        for i, conv in enumerate(conversations[:5], 1):
            print(f"--- Conversation {i} ---")
            print(conv)
            print()
        return

    # Save to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(conv + "\n")

    print(f"Saved {len(conversations)} conversations to {output_path}")
    print(f"\nSample:\n{conversations[0]}")


if __name__ == "__main__":
    main()
