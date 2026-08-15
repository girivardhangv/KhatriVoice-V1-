#!/usr/bin/env python3
"""
Generate diverse conversational training data for KhatriVoice.

This script creates a variety of conversation pairs to ensure:
1. Large vocabulary diversity
2. No excessive repetition
3. Natural conversation flow
"""

import random
import argparse
from pathlib import Path

# Conversation templates with varied responses
TOPICS = {
    "greetings": [
        ("Hi", ["Hello!", "Hi there!", "Hey!", "Greetings!", "Hello! How can I help you?"]),
        ("Hello", ["Hi!", "Hello!", "Hey there!", "Good to see you!"]),
        ("Hey", ["Hey!", "Hi!", "Hello!", "What's up?"]),
        ("Good morning", ["Good morning!", "Morning!", "Good morning! How are you?"]),
        ("Good afternoon", ["Good afternoon!", "Afternoon!", "Hello!"]),
        ("Good evening", ["Good evening!", "Evening!", "Hello there!"]),
        ("How are you?", ["I'm doing well, thanks!", "I'm great, thank you!", "Doing good! How about you?"]),
    ],
    "python": [
        ("What is Python?", [
            "Python is a high-level programming language known for its simple syntax.",
            "Python is a versatile programming language used for web, data science, and automation.",
            "Python is an interpreted language that emphasizes code readability."
        ]),
        ("What are Python variables?", [
            "Variables in Python are containers for storing data values.",
            "A variable is created when you assign a value using the equals sign.",
            "Python variables can hold any data type and are dynamically typed."
        ]),
        ("What is a Python function?", [
            "A function is a reusable block of code defined with the def keyword.",
            "Functions in Python take inputs, process them, and return outputs.",
            "To define a function, use def followed by the function name and parentheses."
        ]),
        ("What is a Python list?", [
            "A list is an ordered, mutable collection of items in Python.",
            "Lists are created with square brackets and can hold mixed types.",
            "Lists allow indexing, slicing, and modification of elements."
        ]),
        ("What is a Python dictionary?", [
            "A dictionary stores key-value pairs using curly braces.",
            "Dictionaries are unordered collections where each key maps to a value.",
            "Use dictionaries when you need fast lookup by key."
        ]),
    ],
    "programming": [
        ("What is programming?", [
            "Programming is writing instructions for computers to execute.",
            "Programming involves creating algorithms and implementing them in code.",
            "Programming is the process of developing software applications."
        ]),
        ("What is a variable?", [
            "A variable is a named container that stores a value in memory.",
            "Variables hold data that can be changed during program execution.",
            "Variables are fundamental building blocks in programming."
        ]),
        ("What is a loop?", [
            "A loop repeats code until a condition is met.",
            "Loops allow you to execute code multiple times efficiently.",
            "Common loops include for-loops and while-loops."
        ]),
        ("What is debugging?", [
            "Debugging is finding and fixing errors in code.",
            "Debugging involves identifying, analyzing, and correcting bugs.",
            "Debugging is an essential skill for every programmer."
        ]),
        ("What is an algorithm?", [
            "An algorithm is a step-by-step procedure for solving a problem.",
            "Algorithms are the logic behind computer programs.",
            "An algorithm defines how to process input to produce output."
        ]),
    ],
    "ai": [
        ("What is AI?", [
            "AI stands for Artificial Intelligence, simulating human intelligence in machines.",
            "AI enables computers to learn, reason, and make decisions.",
            "AI includes machine learning, natural language processing, and robotics."
        ]),
        ("What is machine learning?", [
            "Machine learning is a type of AI where systems learn from data.",
            "ML algorithms improve automatically through experience.",
            "Machine learning enables pattern recognition without explicit programming."
        ]),
        ("What is deep learning?", [
            "Deep learning uses neural networks with multiple layers.",
            "Deep learning excels at image recognition and language processing.",
            "Deep learning models learn hierarchical representations of data."
        ]),
        ("What is a neural network?", [
            "A neural network is computing systems inspired by biological neurons.",
            "Neural networks consist of interconnected nodes that process information.",
            "Neural networks are the foundation of deep learning."
        ]),
        ("What is NLP?", [
            "NLP is Natural Language Processing, enabling computers to understand text.",
            "NLP combines linguistics and AI to process human language.",
            "NLP powers chatbots, translation, and text analysis."
        ]),
    ],
    "technology": [
        ("What is a CPU?", [
            "CPU is the Central Processing Unit, the brain of a computer.",
            "CPUs execute instructions and perform calculations.",
            "The CPU processes data and controls other components."
        ]),
        ("What is RAM?", [
            "RAM is Random Access Memory, providing temporary data storage.",
            "RAM stores data that the CPU needs quick access to.",
            "More RAM allows more programs to run simultaneously."
        ]),
        ("What is an API?", [
            "An API is an Application Programming Interface for software communication.",
            "APIs define how software components should interact.",
            "APIs enable integration between different applications."
        ]),
        ("What is a database?", [
            "A database is an organized collection of structured data.",
            "Databases store, retrieve, and manage data efficiently.",
            "Common databases include SQL and NoSQL systems."
        ]),
        ("What is cloud computing?", [
            "Cloud computing delivers computing services over the internet.",
            "Cloud provides on-demand storage, processing, and applications.",
            "Cloud computing offers scalability and cost efficiency."
        ]),
    ],
    "web": [
        ("What is HTML?", [
            "HTML is HyperText Markup Language for creating web pages.",
            "HTML defines the structure and content of websites.",
            "HTML uses tags to mark up elements on a page."
        ]),
        ("What is CSS?", [
            "CSS is Cascading Style Sheets for styling web pages.",
            "CSS controls layout, colors, and fonts of websites.",
            "CSS separates content from presentation."
        ]),
        ("What is JavaScript?", [
            "JavaScript is a scripting language for interactive web pages.",
            "JavaScript runs in browsers and on servers with Node.js.",
            "JavaScript enables dynamic content and user interactions."
        ]),
        ("What is a website?", [
            "A website is a collection of web pages under one domain.",
            "Websites are accessible via browsers over the internet.",
            "Websites can be static or dynamic with databases."
        ]),
    ],
    "data": [
        ("What is data analysis?", [
            "Data analysis examines data to extract insights and patterns.",
            "Analysis helps make informed decisions from data.",
            "Tools include Python, R, and spreadsheet applications."
        ]),
        ("What is data science?", [
            "Data science combines statistics, programming, and domain expertise.",
            "Data scientists analyze complex data to solve problems.",
            "Data science uses machine learning and visualization."
        ]),
        ("What is big data?", [
            "Big data refers to large, complex datasets requiring special tools.",
            "Big data is characterized by volume, velocity, and variety.",
            "Big data analytics reveal patterns and trends."
        ]),
    ],
    "helpful": [
        ("Thanks", ["You're welcome!", "Happy to help!", "Anytime!", "Glad I could assist!"]),
        ("Thank you", ["You're welcome!", "My pleasure!", "No problem!", "You're very welcome!"]),
        ("That helps", ["Great!", "Glad to hear that!", "Happy it helped!", "Perfect!"]),
        ("I understand now", ["Excellent!", "That's great!", "Wonderful!", "Glad I could clarify!"]),
    ],
    "goodbye": [
        ("Goodbye", ["Goodbye! Have a great day!", "Bye! Take care!", "Goodbye! Come back anytime!"]),
        ("Bye", ["Bye!", "See you later!", "Take care!", "Goodbye!"]),
        ("See you", ["See you!", "Until next time!", "Take care!", "See you soon!"]),
    ],
}


def generate_conversation(count: int, seed: int = 42) -> list:
    """Generate diverse conversations."""
    random.seed(seed)
    conversations = []

    # Flatten topics
    all_questions = []
    for category, qa_pairs in TOPICS.items():
        for question, answers in qa_pairs:
            all_questions.append((question, answers))

    # Generate conversations
    while len(conversations) < count:
        # Pick a random question
        question, answers = random.choice(all_questions)
        answer = random.choice(answers)

        # Format with special tokens
        formatted = f"User: {question}\nAI: {answer}"
        conversations.append(formatted)

    random.shuffle(conversations)
    return conversations[:count]


def generate_multi_turn_conversation() -> list:
    """Generate multi-turn conversations for better context learning."""
    conversations = []

    # Create some multi-turn conversation flows
    multi_turn_flows = [
        [
            ("Hi", "Hello! How can I help you today?"),
            ("What is Python?", "Python is a popular programming language known for its simplicity."),
            ("Thanks!", "You're welcome! Feel free to ask more questions."),
        ],
        [
            ("Hello", "Hi there! What would you like to know?"),
            ("Can you explain machine learning?", "Machine learning is AI that learns from data patterns."),
            ("That helps!", "Great! Let me know if you have other questions."),
            ("Goodbye", "Goodbye! Have a wonderful day!"),
        ],
        [
            ("Hey", "Hey! What can I help you with?"),
            ("What is a neural network?", "A neural network is a computing system inspired by the brain."),
            ("How does it work?", "It processes data through layers of interconnected nodes."),
            ("Thanks for explaining", "Happy to help! Ask anytime."),
        ],
    ]

    for flow in multi_turn_flows:
        conv_lines = []
        for user_text, ai_text in flow:
            conv_lines.append(f"User: {user_text}")
            conv_lines.append(f"AI: {ai_text}")
        conversations.append("\n".join(conv_lines))

    return conversations


def main():
    parser = argparse.ArgumentParser(description="Generate diverse training data")
    parser.add_argument("--output", type=str, default="data/diverse_conversations.txt")
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-multi-turn", action="store_true", default=True)

    args = parser.parse_args()

    print(f"Generating {args.count} single-turn conversations...")
    conversations = generate_conversation(args.count, seed=args.seed)

    if args.include_multi_turn:
        print("Adding multi-turn conversations...")
        multi_turn = generate_multi_turn_conversation()
        conversations.extend(multi_turn)

    # Shuffle
    random.shuffle(conversations)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(conv + "\n")

    print(f"Saved {len(conversations)} conversations to {output_path}")

    # Print stats
    print(f"\nStatistics:")
    print(f"  Total conversations: {len(conversations)}")
    print(f"  Single-turn: {len(conversations) - len(generate_multi_turn_conversation())}")
    print(f"  Multi-turn: {len(generate_multi_turn_conversation())}")


if __name__ == "__main__":
    main()
