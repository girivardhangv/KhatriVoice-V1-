import sys
with open('scripts/prepare_oasst2.py', 'r', encoding='utf-8') as f:
    content = f.read()
    
new_content = content.replace(
    '''if train_convs:
        print(train_convs[0]['text'])''',
    '''if train_convs:
        for c in random.sample(train_convs, min(5, len(train_convs))):
            print(c['text'].encode('cp1252', errors='replace').decode('cp1252'))
            print('---')'''
)
with open('scripts/prepare_oasst2.py', 'w', encoding='utf-8') as f:
    f.write(new_content)
