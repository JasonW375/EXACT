import json
import os
from collections import defaultdict

# File paths
input_json_path = "/path/to/CT-CHAT2/VQA_dataset/filtered_valid_vqa_with_preds_from_heatmap.json"
output_dir = "/path/to/CT-CHAT2/VQA_dataset/by_category_from_heatmap"

# Question type markers
type_tokens = {
    'long_answer': '<long_answer>',
    'short_answer': '<short_answer>',
    'multiple_choice': '<multiple_choice>',
    'report_generation': '<report_generation>'
}

# Check that the input file exists
if not os.path.exists(input_json_path):
    print(f"Error: Input file not found: {input_json_path}")
    exit(1)

# Create the output directory
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

# Read the JSON file
print("Reading JSON file...")
with open(input_json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Loaded {len(data)} entries from JSON")

# Group the data by category
categorized_data = {
    'long_answer': [],
    'short_answer': [],
    'multiple_choice': [],
    'report_generation': []
}

# Counters
stats = defaultdict(int)

# Process every entry
for entry in data:
    entry_id = entry.get('id', '')
    
    # Decide which category this entry belongs to
    category = None
    
    # Method 1: infer the category from the id prefix
    for cat_name in type_tokens.keys():
        if entry_id.startswith(cat_name):
            category = cat_name
            break
    
    # Method 2: fall back to the tokens inside the conversations
    if category is None:
        for conv in entry.get('conversations', []):
            if conv.get('from') == 'human':
                value = conv.get('value', '')
                for cat_name, token in type_tokens.items():
                    if token in value:
                        category = cat_name
                        break
                if category:
                    break
    
    # Append to the matching category
    if category:
        categorized_data[category].append(entry)
        stats[category] += 1
    else:
        stats['unknown'] += 1
        print(f"Warning: Could not determine category for entry: {entry_id}")

# Report the statistics
print("\n" + "="*50)
print("Categorization Statistics:")
print("="*50)
for category in type_tokens.keys():
    count = stats[category]
    print(f"{category:20s}: {count:6d} entries")
print(f"{'unknown':20s}: {stats['unknown']:6d} entries")
print(f"{'TOTAL':20s}: {sum(stats.values()):6d} entries")
print("="*50)

# Write one JSON file per category
print("\nSaving categorized files...")
for category, entries in categorized_data.items():
    if entries:  # skip empty categories
        output_path = os.path.join(output_dir, f"{category}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(entries)} entries to: {output_path}")

print("\nDone!")