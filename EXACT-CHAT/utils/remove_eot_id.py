import json

# Input JSON file
input_file = "/path/to/CT-CHAT2/our_valid_data/all_types_question_validation_results_prediction_from_heatmap/output_validation_new_llama_all_temp0.0_tokens1024_checkpoint-38000.json"
output_file = "/path/to/CT-CHAT2/our_valid_data/all_types_question_validation_results_prediction_from_heatmap/output_validation_new_llama_all_temp0.0_tokens1024_checkpoint-38000_cleaned.json"

# Load the JSON data
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Strip <|eot_id|> from every answer
for item in data:
    if 'conversations_out' in item:
        for conversation in item['conversations_out']:
            if 'answer' in conversation:
                # Remove any <|eot_id|> token from the answer
                conversation['answer'] = conversation['answer'].replace('<|eot_id|>', '')

# Save the cleaned JSON
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"Done. Cleaned file written to: {output_file}")
print(f"Processed {len(data)} records")