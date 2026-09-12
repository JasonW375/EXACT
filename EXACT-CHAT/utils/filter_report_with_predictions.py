import json
import pandas as pd
import sys
import os

# Disease columns, in the order expected by the prompt.
DISEASE_COLUMNS = [
    "Medical material",
    "Arterial wall calcification",
    "Cardiomegaly",
    "Pericardial effusion",
    "Coronary artery wall calcification",
    "Hiatal hernia",
    "Lymphadenopathy",
    "Emphysema",
    "Atelectasis",
    "Lung nodule",
    "Lung opacity",
    "Pulmonary fibrotic sequela",
    "Pleural effusion",
    "Mosaic attenuation pattern",
    "Peribronchial thickening",
    "Consolidation",
    "Bronchiectasis",
    "Interlobular septal thickening"
]

def get_volume_name_from_image(image_filename):
    """Strip the .nii.gz / .npz suffix to get the volume name."""
    if image_filename.endswith('.nii.gz'):
        return image_filename.replace('.nii.gz', '')
    elif image_filename.endswith('.npz'):
        return image_filename.replace('.npz', '')
    else:
        return image_filename

def format_disease_predictions(predictions_dict):
    """Format the disease predictions as a string."""
    predictions_str = "; ".join([f"{disease}={int(predictions_dict[disease])}" 
                                  for disease in DISEASE_COLUMNS])
    return predictions_str

def add_predictions_to_human_value(original_value, predictions_str):
    """Append the disease predictions to the human turn.

    The <report_generation> tag is moved to the end, so the value becomes
    "<image> <question> Known frontend model predictions (disease-wise):
    <predictions>.<report_generation>".
    """
    # Drop the existing <report_generation> tag.
    value = original_value.replace('<report_generation>', '').strip()
    
    # Re-attach it after the predictions.
    new_value = f"{value} Known frontend model predictions (disease-wise): {predictions_str}.<report_generation>"
    
    return new_value

def filter_and_add_predictions(input_json_path, csv_path, output_json_path):
    """Keep the report-generation entries and attach the disease predictions.

    :param input_json_path: training JSON to filter
    :param csv_path: path to disease_predictions.csv
    :param output_json_path: path of the filtered JSON to write
    """
    # Load the predictions.
    print(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"CSV holds {len(df)} records")
    
    # Index by VolumeName for fast lookup.
    df.set_index('VolumeName', inplace=True)
    csv_volumes = set(df.index)
    print(f"Volume names in the CSV: {list(csv_volumes)[:5]}... (showing first 5)")
    
    # Load the training JSON.
    print(f"\nReading input JSON: {input_json_path}")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    print(f"Input JSON holds {len(input_data)} records")
    
    # Filter and rewrite.
    filtered_data = []
    matched_count = 0
    unmatched_count = 0
    non_report_count = 0
    
    for item in input_data:
        # Entries without conversations are skipped.
        if 'conversations' not in item:
            continue
        
        # Keep only the report_generation entries.
        is_report_generation = False
        for conv in item['conversations']:
            if conv.get('type') == 'report_generation':
                is_report_generation = True
                break
        
        if not is_report_generation:
            non_report_count += 1
            continue
        
        # Image file name and the matching volume name.
        image_filename = item.get('image', '')
        volume_name = get_volume_name_from_image(image_filename)
        
        # Skip volumes that are missing from the CSV.
        if volume_name not in csv_volumes:
            unmatched_count += 1
            continue
        
        matched_count += 1
        
        # Disease predictions for this volume.
        predictions = df.loc[volume_name]
        predictions_dict = predictions.to_dict()
        
        # Format them for the prompt.
        disease_str = format_disease_predictions(predictions_dict)
        
        # Build the replacement entry.
        new_item = {
            "id": item['id'],
            "image": item['image'],
            "conversations": []
        }
        
        # Copy each conversation turn.
        for conv in item['conversations']:
            new_conv = conv.copy()
            
            # Attach the predictions to the report_generation human turn.
            if (conv.get('from') == 'human' and 
                conv.get('type') == 'report_generation'):
                original_value = conv.get('value', '')
                new_conv['value'] = add_predictions_to_human_value(original_value, disease_str)
            
            new_item['conversations'].append(new_conv)
        
        filtered_data.append(new_item)
    
    # Report the statistics.
    print(f"\nStatistics:")
    print(f"- Input JSON records: {len(input_data)}")
    print(f"- Not report_generation: {non_report_count}")
    print(f"- report_generation but missing from the CSV: {unmatched_count}")
    print(f"- Matched and kept: {matched_count}")
    print(f"- Output JSON records: {len(filtered_data)}")
    
    # Write the output.
    print(f"\nWriting output JSON to: {output_json_path}")
    
    # Create the output directory if needed.
    output_dir = os.path.dirname(output_json_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=4, ensure_ascii=False)
    
    print(f"Done: kept {len(filtered_data)} report generation records")
    
    # Show the first record as a sanity check.
    if filtered_data:
        print(f"\nFirst record:")
        print(f"ID: {filtered_data[0]['id']}")
        print(f"Image: {filtered_data[0]['image']}")
        if filtered_data[0]['conversations']:
            first_conv = filtered_data[0]['conversations'][0]
            print(f"First conversation type: {first_conv.get('type')}")
            print(f"Value preview: {first_conv.get('value', '')[:150]}...")
    
    return filtered_data

if __name__ == "__main__":
    # Default paths.
    input_json_path = "/path/to/CT-CHAT2/VQA_dataset/filtered_train_vqa.json"
    csv_path = "/path/to/disease_predictions.csv"
    output_json_path = "/path/to/CT-CHAT2/our_train_data/class_invalid_report_generation.json"
    
    # Optional command-line overrides.
    if len(sys.argv) >= 2:
        input_json_path = sys.argv[1]
    if len(sys.argv) >= 3:
        csv_path = sys.argv[2]
    if len(sys.argv) >= 4:
        output_json_path = sys.argv[3]
    
    print("="*80)
    print("Filter report-generation entries and attach disease predictions")
    print("="*80)
    print(f"Input JSON: {input_json_path}")
    print(f"CSV file: {csv_path}")
    print(f"Output JSON: {output_json_path}")
    print("="*80)
    
    # Both inputs have to exist.
    if not os.path.exists(input_json_path):
        print(f"Error: input JSON not found: {input_json_path}")
        sys.exit(1)
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)
    
    # Run.
    filter_and_add_predictions(input_json_path, csv_path, output_json_path)