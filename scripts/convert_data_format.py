#!/usr/bin/env python3
"""
Convert conversation data from 'User:/AI:' format to special token format.

Converts:
    User: Hello
    AI: Hi there!

To:
    <user>
    Hello
    <|assistant|>
    Hi there!
    <|end|>
"""

import argparse
from pathlib import Path
import re


def convert_conversation_file(input_path: str, output_path: str) -> dict:
    """
    Convert a conversation file to the proper format.

    Args:
        input_path: Path to input file
        output_path: Path to output file

    Returns:
        Statistics about the conversion
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Read input file
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.strip().split('\n')

    conversations = []
    current_conversation = []
    stats = {
        'total_lines': len(lines),
        'user_turns': 0,
        'assistant_turns': 0,
        'conversations': 0,
    }

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check for User: prefix
        if line.lower().startswith('user:'):
            stats['user_turns'] += 1
            user_text = line[5:].strip()  # Remove "User:" prefix
            current_conversation.append(f"<user>\n{user_text}")
        # Check for AI: prefix
        elif line.lower().startswith('ai:'):
            stats['assistant_turns'] += 1
            assistant_text = line[3:].strip()  # Remove "AI:" prefix
            current_conversation.append(f"<|assistant|>\n{assistant_text}\n<|end|>")

            # Check if we've completed a pair (user + assistant)
            if len(current_conversation) >= 2:
                stats['conversations'] += 1
                conversations.append('\n'.join(current_conversation))
                current_conversation = []

    # Handle any remaining partial conversation
    if current_conversation:
        conversations.append('\n'.join(current_conversation))

    # Write output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        # Each conversation pair on separate line for training
        for conv in conversations:
            f.write(conv + '\n\n')

    return stats


def convert_ai_models_file(input_path: str, output_path: str) -> dict:
    """
    Convert AI models data file to conversation format.

    The ai_models_100k_lines.txt contains structured model descriptions.
    We'll convert each line into a Q&A format.

    Args:
        input_path: Path to input file
        output_path: Path to output file

    Returns:
        Statistics about the conversion
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    conversations = []
    stats = {
        'total_lines': len(lines),
        'converted': 0,
    }

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Parse: AI_Model_000001 | Base_Model: GPT | Type: Large Language Model | ...
        # Create Q&A pairs about AI models

        # Extract model info
        parts = line.split('|')
        if len(parts) < 2:
            continue

        model_id = parts[0].strip()
        attributes = {}
        for part in parts[1:]:
            if ':' in part:
                key, value = part.split(':', 1)
                attributes[key.strip()] = value.strip()

        if 'Base_Model' in attributes and 'Type' in attributes:
            stats['converted'] += 1

            # Create multiple Q&A pairs from each model entry
            base_model = attributes.get('Base_Model', 'Unknown')
            model_type = attributes.get('Type', 'Unknown')

            # Q&A 1: What type of model is X?
            q1 = f"What type of model is {base_model}?"
            a1 = f"{base_model} is a {model_type}."
            conversations.append(f"<user>\n{q1}\n<|assistant|>\n{a1}\n<|end|>")

            # Q&A 2: Tell me about [model]
            q2 = f"Tell me about {base_model}."
            details = []
            for key, value in attributes.items():
                details.append(f"{key}: {value}")
            a2 = f"{base_model} is a model with these characteristics: {', '.join(details)}."
            conversations.append(f"<user>\n{q2}\n<|assistant|>\n{a2}\n<|end|>")

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for conv in conversations:
            f.write(conv + '\n\n')

    return stats


def main():
    parser = argparse.ArgumentParser(description="Convert training data to special token format")
    parser.add_argument('--input', type=str, required=True, help='Input file path')
    parser.add_argument('--output', type=str, required=True, help='Output file path')
    parser.add_argument('--type', type=str, choices=['conversation', 'ai_models'], default='conversation',
                        help='Type of conversion')

    args = parser.parse_args()

    print(f"Converting {args.input} -> {args.output}")
    print(f"Type: {args.type}")

    if args.type == 'conversation':
        stats = convert_conversation_file(args.input, args.output)
        print(f"\nConversion complete!")
        print(f"  Total lines: {stats['total_lines']}")
        print(f"  User turns: {stats['user_turns']}")
        print(f"  Assistant turns: {stats['assistant_turns']}")
        print(f"  Conversation pairs: {stats['conversations']}")
    else:
        stats = convert_ai_models_file(args.input, args.output)
        print(f"\nConversion complete!")
        print(f"  Total lines: {stats['total_lines']}")
        print(f"  Converted: {stats['converted']}")

    print(f"\nOutput saved to: {args.output}")


if __name__ == "__main__":
    main()
