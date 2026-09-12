import json

# Input and output file paths
input_path = "/path/to/CT-CHAT/output_validation_vicuna_24000.json"
output_base = "/path/to/CT-CHAT/output_validation_vicuna_24000_"

type_tokens = {
    'long_answer': '<long_answer>',
    'short_answer': '<short_answer>',
    'multiple_choice': '<multiple_choice>',
    'report_generation': '<report_generation>'
}
data_by_type = {k: [] for k in type_tokens}

with open(input_path, 'r', encoding='utf-8') as fin:
    all_items = json.load(fin)

for item in all_items:
    # Inspect every conversation of each item
    for conv in item['conversations_out']:
        for k, token in type_tokens.items():
            if token in conv['question']:
                # Keep the image together with this QA pair
                data_by_type[k].append({
                    'image': item['image'],
                    'question': conv['question'],
                    'answer': conv['answer']
                })

# Write one file per question type
for k in type_tokens:
    with open(f"{output_base}{k}.json", "w", encoding='utf-8') as fout:
        json.dump(data_by_type[k], fout, ensure_ascii=False, indent=2)

print('Categorization complete.')