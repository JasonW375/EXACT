import json
import csv
import re
import os

# File paths
json_input_path = "/path/to/CT-CHAT2/VQA_dataset/filtered_valid_vqa_with_preds.json"
csv_path = "/path/to/CT_Report/CT_Report16_classification/heatmap_ft/ct-rate/disease_predictions.csv"
json_output_path = "/path/to/CT-CHAT2/VQA_dataset/filtered_valid_vqa_with_preds_from_heatmap.json"

# Check that the input files exist
if not os.path.exists(json_input_path):
    print(f"Error: JSON input file not found: {json_input_path}")
    exit(1)

if not os.path.exists(csv_path):
    print(f"Error: CSV file not found: {csv_path}")
    exit(1)

# Read the CSV and build the prediction lookup table
print("Reading CSV file...")
predictions_dict = {}

with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        volume_name = row['VolumeName']
        # Every column except VolumeName is a disease prediction
        disease_preds = {k: v for k, v in row.items() if k != 'VolumeName'}
        predictions_dict[volume_name] = disease_preds

print(f"Loaded predictions for {len(predictions_dict)} volumes")

# Read the JSON file
print("Reading JSON file...")
with open(json_input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Loaded {len(data)} entries from JSON")

# Disease order is taken from the CSV header
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    headers = next(reader)
    disease_names = [h for h in headers if h != 'VolumeName']

print(f"Disease names: {disease_names}")

# Derive the volume name from the image name (drop the .nii.gz suffix)
def get_volume_name(image_name):
    return image_name.replace('.nii.gz', '')

# Build the prediction string for one volume
def build_prediction_string(disease_preds):
    pred_parts = []
    for disease in disease_names:
        value = disease_preds.get(disease, '0')
        pred_parts.append(f"{disease}={value}")
    return "; ".join(pred_parts) + "."

# Replace the prediction values inside a conversation
def replace_predictions_in_text(text, new_pred_string):
    # Replace everything between "Known frontend model predictions (disease-wise): " and <long_answer>
    pattern = r"(Known frontend model predictions \(disease-wise\): )([^<]+)(<long_answer>)"
    replacement = r"\1" + new_pred_string + r"\3"
    return re.sub(pattern, replacement, text)

# Process every entry
updated_count = 0
not_found_count = 0
not_found_volumes = set()

for entry in data:
    image_name = entry['image']
    volume_name = get_volume_name(image_name)
    
    # Check whether this volume has predictions
    if volume_name in predictions_dict:
        disease_preds = predictions_dict[volume_name]
        new_pred_string = build_prediction_string(disease_preds)
        
        # Replace the prediction block in every matching human turn
        for conv in entry['conversations']:
            if conv['from'] == 'human' and 'Known frontend model predictions' in conv['value']:
                old_value = conv['value']
                new_value = replace_predictions_in_text(old_value, new_pred_string)
                conv['value'] = new_value
                updated_count += 1
    else:
        not_found_count += 1
        not_found_volumes.add(volume_name)

print(f"\nProcessing complete:")
print(f"- Updated {updated_count} conversations")
print(f"- {not_found_count} conversations with volumes not found in CSV")
if not_found_volumes:
    print(f"- Volumes not found: {sorted(list(not_found_volumes))[:10]}...")  # show the first 10 only

# Write the updated JSON file
print(f"\nSaving to {json_output_path}...")
with open(json_output_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("Done!")