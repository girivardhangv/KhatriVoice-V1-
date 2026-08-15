#!/usr/bin/env python3
import json
import argparse
import random
from pathlib import Path
from collections import defaultdict

def create_conversations(dataset_split, max_convs, min_words=10, max_words=300):
    # Filter for english and valid
    messages = []
    dropped_non_en = 0
    dropped_invalid = 0
    
    for row in dataset_split:
        if row['lang'] != 'en':
            dropped_non_en += 1
            continue
        if row.get('deleted', False) or row.get('review_result') is False:
            dropped_invalid += 1
            continue
        messages.append(row)
        
    print(f"Loaded {len(messages)} English messages (Dropped {dropped_non_en} non-EN, {dropped_invalid} invalid)")
    
    # Build tree
    children_map = defaultdict(list)
    msg_dict = {}
    
    for msg in messages:
        msg_id = msg['message_id']
        parent_id = msg['parent_id']
        msg_dict[msg_id] = msg
        children_map[parent_id].append(msg_id)
        
    # Reconstruct paths (DFS)
    paths = []
    
    def dfs(current_id, current_path):
        msg = msg_dict[current_id]
        new_path = current_path + [msg]
        
        children = children_map.get(current_id, [])
        if not children:
            # Leaf node, add the path if it's long enough and ends with assistant
            paths.append(new_path)
            return

        for child_id in children:
            if child_id in msg_dict:
                dfs(child_id, new_path)

    # Roots are messages with parent_id=None
    roots = children_map.get(None, [])
    for root_id in roots:
        if root_id in msg_dict:
            dfs(root_id, [])
            
    # Format and filter paths
    conversations = []
    seen = set()
    
    user_token = "<user>"
    assistant_token = "<|assistant>"
    end_token = "<|end|>"
    
    for path in paths:
        # We want to end on an assistant message. If the path ends on user, truncate it by 1.
        if path and path[-1]['role'] != 'assistant':
            path = path[:-1]
            
        if len(path) < 2:
            continue
            
        # Build text
        conv_text = []
        word_count = 0
        valid = True
        
        user_msgs = 0
        assistant_msgs = 0
        
        for msg in path:
            role = msg['role']
            text = msg['text'].strip()
            word_count += len(text.split())
            
            if role == 'prompter':
                conv_text.append(f"{user_token}\n{text}")
                user_msgs += len(text.split())
            elif role == 'assistant':
                conv_text.append(f"{assistant_token}\n{text}\n{end_token}")
                assistant_msgs += len(text.split())
            else:
                valid = False
                break
                
        if not valid or word_count < min_words or word_count > max_words:
            continue
            
        final_str = "\n".join(conv_text)
        
        # Deduplicate
        if final_str in seen:
            continue
        seen.add(final_str)
        
        conversations.append({
            "text": final_str,
            "word_count": word_count,
            "user_words": user_msgs,
            "assistant_words": assistant_msgs
        })
        
    return conversations

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-conversations", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-words", type=int, default=10)
    parser.add_argument("--max-words", type=int, default=400)
    parser.add_argument("--output-dir", type=str, default="data/processed")
    args = parser.parse_args()
    
    print("Loading OpenAssistant/oasst2...")
    from datasets import load_dataset
    dataset = load_dataset("OpenAssistant/oasst2")
    
    random.seed(args.seed)
    
    print("\nProcessing training split...")
    train_convs = create_conversations(dataset['train'], args.max_conversations, args.min_words, args.max_words)
    
    print("\nProcessing validation split...")
    val_convs = create_conversations(dataset['validation'], int(args.max_conversations * 0.1), args.min_words, args.max_words)
    
    # Shuffle and trim
    random.shuffle(train_convs)
    train_convs = train_convs[:args.max_conversations]
    
    random.shuffle(val_convs)
    val_subset_size = min(len(val_convs), int(args.max_conversations * 0.1))
    val_convs = val_convs[:val_subset_size]
    
    print(f"\nFinal training conversations: {len(train_convs)}")
    print(f"Final validation conversations: {len(val_convs)}")
    
    # Stats
    train_lens = [c['word_count'] for c in train_convs] if train_convs else [0]
    train_user_lens = [c['user_words'] for c in train_convs] if train_convs else [0]
    train_asst_lens = [c['assistant_words'] for c in train_convs] if train_convs else [0]
    
    stats = {
        "total_training_conversations": len(train_convs),
        "total_validation_conversations": len(val_convs),
        "average_conversation_words": sum(train_lens) / max(1, len(train_lens)),
        "average_user_words": sum(train_user_lens) / max(1, len(train_user_lens)),
        "average_assistant_words": sum(train_asst_lens) / max(1, len(train_asst_lens)),
        "max_words": max(train_lens, default=0),
        "min_words": min(train_lens, default=0)
    }
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "oasst2_train.txt", "w", encoding="utf-8") as f:
        for c in train_convs:
            f.write(c['text'] + "\n\n")
            
    with open(out_dir / "oasst2_val.txt", "w", encoding="utf-8") as f:
        for c in val_convs:
            f.write(c['text'] + "\n\n")
            
    with open(out_dir / "oasst2_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
        
    print("\nExample Conversation:")
    print("="*50)
    if train_convs:
        for c in random.sample(train_convs, min(5, len(train_convs))):
            print(c['text'].encode('cp1252', errors='replace').decode('cp1252'))
            print('---')
    print("="*50)
    
    print(f"\nSaved to {out_dir}/oasst2_train.txt and oasst2_val.txt")
    print("\nNext step: update your training config to point to 'data/processed/oasst2_train.txt' and train the tokenizer.")
    print("\nTo train, use:")
    print("  python scripts/train.py --config configs/small.yaml --conversation-mode")

if __name__ == "__main__":
    main()
