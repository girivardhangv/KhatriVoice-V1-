from datasets import load_dataset
dataset = load_dataset("OpenAssistant/oasst2")
print("Splits:", dataset.keys())
print("Columns:", dataset['train'].column_names)
print("Example:")
for k, v in dataset['train'][0].items():
    print(f"  {k}: {repr(v)[:100]}")
