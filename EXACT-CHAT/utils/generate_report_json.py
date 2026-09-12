import json
import os
import random
import pandas as pd
import sys

# Question templates, sampled at random for each volume.
QUESTION_TEMPLATES = [
    "Write a radiology report for the following CT scan.",
    "Could you write the radiology report for this chest CT scan?",
    "Could you create a report for this chest CT scan?",
    "Produce the report for this CT image.",
    "I need a detailed report for the given chest CT image.",
    "Please provide the radiology report for the chest CT image mentioned.",
    "Can you generate the report for the following chest CT scan?",
    "Provide the radiology report for this CT scan.",
    "I need the radiology report for the given chest CT volume.",
    "Generate radiology report for the CT scan.",
    "Write a radiology report for the following CT volume.",
    "Could you create a report for this chest CT volume?",
    "Generate radiology report for the CT volume.",
    "Would you mind generating the radiology report for the specified chest CT volume?",
    "Please give the radiology report for the specified chest CT volume.",
    "Create a report for this chest CT scan.",
    "Can you produce the radiology report for the attached chest CT scan?",
    "Please provide the radiology report for the chest CT scan mentioned.",
    "Can you generate the report for the following chest CT volume?",
    "Please generate the report for the chest CT volume provided.",
    "Can you produce the radiology report for the attached chest CT image?",
    "Please generate the report for the chest CT image provided.",
    "Could you write the radiology report for this chest CT volume?",
    "Produce the report for this CT volume.",
    "Create a report for this chest CT.",
    "Can you produce the radiology report for the attached chest CT volume?",
    "Would you mind generating the radiology report for the specified chest CT scan?",
    "Produce the report for this CT scan.",
    "Create a report for this chest CT volume.",
    "I need a detailed report for the given chest CT volume.",
    "Please generate the report for the chest CT scan provided.",
    "Please provide the radiology report for the chest CT volume mentioned.",
    "Please give the radiology report for the specified chest CT scan.",
    "Generate radiology report for the CT.",
    "Can you generate the report for the following chest CT image?",
    "I need the radiology report for the given chest CT scan.",
    "I need the radiology report for the given chest CT image.",
    "Provide the radiology report for this CT volume.",
    "Would you mind generating the radiology report for the specified chest CT image?",
    "Please give the radiology report for the specified chest CT image.",
    "I need a detailed report for the given chest CT scan.",
    "Provide the radiology report for this CT image."
]

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

def get_volume_name_from_npz(npz_filename):
    """Strip the .npz suffix to get the volume name."""
    return npz_filename.replace('.npz', '')

def format_disease_predictions(predictions_dict):
    """Format the disease predictions as a string."""
    predictions_str = "; ".join([f"{disease}={int(predictions_dict[disease])}" 
                                  for disease in DISEASE_COLUMNS])
    return predictions_str

def generate_json_data(embeddings_dir, csv_path, output_json_path, seed=42):
    """Build the report-generation JSON.

    :param embeddings_dir: directory holding the .npz embeddings
    :param csv_path: path to disease_predictions.csv
    :param output_json_path: path of the JSON file to write
    :param seed: seed for the question-template sampling
    """
    # Seed the RNG so the sampled questions are reproducible.
    random.seed(seed)
    
    # Load the predictions.
    print(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"CSV holds {len(df)} records")
    
    # Index by VolumeName for fast lookup.
    df.set_index('VolumeName', inplace=True)
    
    # Collect the .npz files.
    print(f"\nScanning embeddings directory: {embeddings_dir}")
    npz_files = [f for f in os.listdir(embeddings_dir) if f.endswith('.npz')]
    npz_files.sort()  # Sort for a stable order.
    print(f"Found {len(npz_files)} npz files")
    
    # Build the records.
    json_data = []
    matched_count = 0
    unmatched_volumes = []
    
    for idx, npz_file in enumerate(npz_files):
        volume_name = get_volume_name_from_npz(npz_file)
        
        # Skip volumes that are missing from the CSV.
        if volume_name not in df.index:
            unmatched_volumes.append(volume_name)
            continue
        
        matched_count += 1
        
        # Disease predictions for this volume.
        predictions = df.loc[volume_name]
        predictions_dict = predictions.to_dict()
        
        # Format them for the prompt.
        disease_str = format_disease_predictions(predictions_dict)
        
        # Pick a question template at random.
        question = random.choice(QUESTION_TEMPLATES)
        
        # Build the human turn.
        human_value = f"<image>\n{question} Known frontend model predictions (disease-wise): {disease_str}.<report_generation>"
        
        # Build the record; the answer is left empty for inference.
        entry = {
            "id": f"report_generation_{idx}",
            "image": npz_file,
            "conversations": [
                {
                    "type": "report_generation",
                    "from": "human",
                    "value": human_value
                },
                {
                    "from": "gpt",
                    "value": ""
                }
            ]
        }
        
        json_data.append(entry)
    
    # Report the statistics.
    print(f"\nStatistics:")
    print(f"- NPZ files: {len(npz_files)}")
    print(f"- Matched: {matched_count}")
    print(f"- Unmatched: {len(unmatched_volumes)}")
    
    if unmatched_volumes:
        print(f"\nFirst 10 unmatched volumes:")
        for vol in unmatched_volumes[:10]:
            print(f"  - {vol}")
        if len(unmatched_volumes) > 10:
            print(f"  ... and {len(unmatched_volumes) - 10} more")
    
    # Write the JSON.
    print(f"\nWriting JSON to: {output_json_path}")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"Done: {len(json_data)} records written")
    
    return json_data

if __name__ == "__main__":
    # Default paths.
    embeddings_dir = "/path/to/CT-CHAT2/our_valid_data_radchest/embeddings"
    csv_path = "/path/to/CT_Report/CT_Report16_classification/heatmap_ft/radchest/disease_predictions.csv"
    output_json_path = "/path/to/CT-CHAT2/our_valid_data_radchest/report_generation.json"
    
    # Optional command-line overrides.
    if len(sys.argv) >= 2:
        embeddings_dir = sys.argv[1]
    if len(sys.argv) >= 3:
        csv_path = sys.argv[2]
    if len(sys.argv) >= 4:
        output_json_path = sys.argv[3]
    
    print("="*80)
    print("Generate the report-generation JSON")
    print("="*80)
    print(f"Embeddings directory: {embeddings_dir}")
    print(f"CSV file: {csv_path}")
    print(f"Output JSON: {output_json_path}")
    print("="*80)
    
    # Both inputs have to exist.
    if not os.path.exists(embeddings_dir):
        print(f"Error: embeddings directory not found: {embeddings_dir}")
        sys.exit(1)
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)
    
    # Generate.
    generate_json_data(embeddings_dir, csv_path, output_json_path)