
#!/usr/bin/env python3
import re
from pathlib import Path
import random
from collections import defaultdict

USER_TOKEN = '<user>'
ASSISTANT_TOKEN = '<|assistant>'
END_TOKEN = '<|end|>'

def parse_conversations(content):
    conversations = []
    blocks = re.split(r'[
][\s]*[
]', content)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if USER_TOKEN not in block or ASSISTANT_TOKEN not in block or END_TOKEN not in block:
            continue
        user_pattern = re.escape(USER_TOKEN) + r'[\s]*[
](.+?)[
][\s]*' + re.escape(ASSISTANT_TOKEN)
        user_match = re.search(user_pattern, block, re.DOTALL)
        if not user_match:
            continue
        user_text = user_match.group(1).strip()
        asst_pattern = re.escape(ASSISTANT_TOKEN) + r'[\s]*[
](.+?)[
][\s]*' + re.escape(END_TOKEN)
        asst_match = re.search(asst_pattern, block, re.DOTALL)
        if not asst_match:
            continue
        asst_text = asst_match.group(1).strip()
        if len(user_text) < 2 or len(asst_text) < 2:
            continue
        if 'User:' in user_text or 'Assistant:' in asst_text:
            continue
        conversations.append({'user': user_text, 'assistant': asst_text})
    return conversations

def filter_quality_conversations(conversations):
    filtered = []
    for conv in conversations:
        user_text = conv['user']
        asst_text = conv['assistant']
        if len(asst_text) > 300:
            continue
        if '[
]' in repr(asst_text):
            asst_text = asst_text.split('[
]')[0].strip()
        if len(asst_text) < 5:
            continue
        no_quality_patterns = ['abstract', 'introduction', 'conclusion', 'section', 'et al', 'figure', 'table', 'references', 'this paper', 'we propose', 'we present']
        text_lower = (user_text + ' ' + asst_text).lower()
        if any(pat in text_lower for pat in no_quality_patterns):
            continue
        conv['assistant'] = asst_text
        filtered.append(conv)
    return filtered

def dedup_with_variety(conversations, max_per_question=5):
    by_question = defaultdict(list)
    for conv in conversations:
        user_norm = conv['user'].lower().strip()
        asst_norm = conv['assistant'].lower().strip()
        exists = False
        for existing in by_question[user_norm]:
            if existing['assistant'].lower().strip() == asst_norm:
                exists = True
                break
        if not exists and len(by_question[user_norm]) < max_per_question:
            by_question[user_norm].append(conv)
    result = []
    for convs in by_question.values():
        result.extend(convs)
    return result

def format_conversation(conv):
    return USER_TOKEN + '[
]' + conv['user'] + '[
]' + ASSISTANT_TOKEN + '[
]' + conv['assistant'] + '[
]' + END_TOKEN

def main():
    input_path = Path('data/processed/train_backup.txt')
    if not input_path.exists():
        input_path = Path('data/processed/train.txt')
    print('Reading', input_path)
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    print('Parsing conversations...')
    conversations = parse_conversations(content)
    print('Found', len(conversations), 'raw conversation pairs')
    print('Filtering for quality...')
    conversations = filter_quality_conversations(conversations)
    print(len(conversations), 'pairs after quality filter')
    print('Deduplicating...')
    conversations = dedup_with_variety(conversations, max_per_question=5)
    print(len(conversations), 'pairs after dedup')
    random.seed(42)
    random.shuffle(conversations)
    small_size = min(5000, len(conversations))
    small_dataset = conversations[:small_size]
    small_path = Path('data/processed/train_small.txt')
    with open(small_path, 'w', encoding='utf-8') as f:
        for conv in small_dataset:
            f.write(format_conversation(conv) + '[
][
]')
    print('Small dataset:', small_size, 'conversations')
    full_path = Path('data/processed/train.txt')
    with open(full_path, 'w', encoding='utf-8') as f:
        for conv in conversations:
            f.write(format_conversation(conv) + '[
][
]')
    print('Full dataset:', len(conversations), 'conversations')
    print('SUMMARY:', len(conversations), 'total')
    for i, conv in enumerate(small_dataset[:5]):
        print('Sample', i+1)
        print('  User:', conv['user'][:60])
        print('  Assistant:', conv['assistant'][:60])

if __name__ == '__main__':
    main()
