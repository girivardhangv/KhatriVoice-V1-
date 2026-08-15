#!/usr/bin/env python3
"""
Convert all training data files to proper special token format.

This script handles multiple data formats and combines them into
a single training file with proper <user> and <|assistant> tokens.
"""

import argparse
from pathlib import Path
import re


def parse_user_ai_pairs(content: str) -> list:
    """
    Parse content with 'User:' and 'AI:' or 'Assistant:' markers.
    Handles multi-line content properly.
    """
    conversations = []

    # Normalize the content - handle both "AI:" and "Assistant:"
    content = re.sub(r'\bAssistant:', 'AI:', content)

    # Split by User: markers but keep the marker
    parts = re.split(r'(?=User:)', content, flags=re.IGNORECASE)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Match User: ... AI: ... pattern
        match = re.match(r'User:\s*(.+?)\s*AI:\s*(.+?)(?=User:|$)', part,
                        flags=re.DOTALL | re.IGNORECASE)
        if match:
            user_text = match.group(1).strip()
            assistant_text = match.group(2).strip()

            # Clean up any remaining formatting
            user_text = re.sub(r'\s+', ' ', user_text)
            assistant_text = re.sub(r'\s+', ' ', assistant_text)

            if user_text and assistant_text:
                conversations.append({
                    'user': user_text,
                    'assistant': assistant_text
                })

    return conversations


def parse_inline_pairs(content: str) -> list:
    """
    Parse content where User:/Assistant: appear inline on same line.
    """
    conversations = []

    for line in content.split('\n'):
        line = line.strip()
        if not line:
            continue

        # Match "User: ... Assistant: ..." or "User: ... AI: ..."
        match = re.match(r'User:\s*(.+?)\s+(?:AI|Assistant):\s*(.+)$', line,
                        re.IGNORECASE)
        if match:
            user_text = match.group(1).strip()
            assistant_text = match.group(2).strip()
            if user_text and assistant_text:
                conversations.append({
                    'user': user_text,
                    'assistant': assistant_text
                })

    return conversations


def parse_plain_text_with_questions(content: str) -> list:
    """
    Parse plain text paragraphs and convert to Q&A format when appropriate.
    """
    conversations = []

    # Look for question-like patterns
    question_patterns = [
        (r'What is (.+?)\?', 'What is {}?'),
        (r'How (?:can|do|should) (?:I|you|we) (.+?)\?', 'How can I {}?'),
        (r'Why (?:does|is|do|are) (.+?)\?', 'Why {}?'),
    ]

    paragraphs = re.split(r'\n\n+', content)

    for para in paragraphs:
        para = para.strip()
        if not para or len(para) < 20:
            continue

        # If it's a Q&A pattern, extract it
        for pattern, template in question_patterns:
            match = re.search(pattern, para, re.IGNORECASE)
            if match:
                question = match.group(0)
                # Use the paragraph as the answer
                answer = para.replace(question, '').strip()
                if answer and len(answer) > 10:
                    conversations.append({
                        'user': question,
                        'assistant': answer
                    })
                break

    return conversations


def convert_file(input_path: str, file_type: str = 'auto') -> list:
    """
    Convert a single file to conversation pairs.

    Args:
        input_path: Path to input file
        file_type: 'user_ai', 'inline', 'plain', or 'auto'
    """
    input_path = Path(input_path)
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    conversations = []

    if file_type == 'auto':
        # Auto-detect format based on content
        if 'User:' in content and 'AI:' in content:
            if '\nUser:' in content:
                # Each User: is on new line (separate turns)
                file_type = 'user_ai'
            else:
                # User: and AI: same line
                file_type = 'inline'
        else:
            file_type = 'plain'

    if file_type == 'user_ai':
        conversations.extend(parse_user_ai_pairs(content))
    elif file_type == 'inline':
        conversations.extend(parse_inline_pairs(content))
    elif file_type == 'plain':
        conversations.extend(parse_plain_text_with_questions(content))

    return conversations


def format_conversation(conv: dict) -> str:
    """Format a single conversation pair with special tokens."""
    return f"<user>\n{conv['user']}\n <|assistant|>\n{conv['assistant']}\n<|end|>"


def main():
    parser = argparse.ArgumentParser(description="Convert all training data")
    parser.add_argument('--output', type=str, default='data/processed/train.txt',
                        help='Output file path')
    parser.add_argument('--files', type=str, nargs='+',
                        help='Input files to process')
    parser.add_argument('--append', action='store_true',
                        help='Append to existing output file')

    args = parser.parse_args()

    # Default files if none specified
    if not args.files:
        args.files = [
            'data/diverse_conversations.txt',
            'khatri_voice_user_ai_100k.txt',
            'cooking.txt',
        ]

    all_conversations = []
    stats = {}

    for filepath in args.files:
        filepath = Path(filepath)
        if not filepath.exists():
            print(f"Skipping {filepath} (not found)")
            continue

        print(f"Processing {filepath}...")
        conversations = convert_file(str(filepath))
        stats[filepath.name] = len(conversations)
        all_conversations.extend(conversations)
        print(f"  Found {len(conversations)} conversation pairs")

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mode = 'a' if args.append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        for conv in all_conversations:
            f.write(format_conversation(conv) + '\n\n')

    print(f"\n{'='*50}")
    print(f"Total conversation pairs: {len(all_conversations)}")
    print(f"Output saved to: {output_path}")
    print(f"\nBreakdown:")
    for filename, count in stats.items():
        print(f"  {filename}: {count} pairs")


if __name__ == "__main__":
    main()
