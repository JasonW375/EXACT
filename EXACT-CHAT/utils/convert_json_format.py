#!/usr/bin/env python3
"""Convert translated_reports.json to the training format.

1. Rewrite .nii.gz file names as .npz
2. Drop the disease prediction block and keep the base prompt only
"""
import json
import argparse
from pathlib import Path


def convert_json(input_path, output_path, prompt_template=None):
    """Convert a JSON file to the report-generation format.
    
    Args:
        input_path: path to the input JSON file
        output_path: path to the output JSON file
        prompt_template: optional custom prompt template
    """
    
    # Default prompt template
    if prompt_template is None:
        prompt_template = "<image>\nPlease give the radiology report for the specified chest CT scan.<report_generation>"
    
    print(f"\n{'='*60}")
    print("JSON format conversion")
    print(f"{'='*60}")
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Prompt template: {prompt_template}")
    print(f"{'='*60}\n")
    
    # Load the source JSON
    print("Reading the JSON file...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Read {len(data)} samples\n")
    
    # Convert the records
    converted_data = []
    conversion_stats = {
        'total': len(data),
        'nii_gz_to_npz': 0,
        'nii_to_npz': 0,
        'already_npz': 0,
        'prompt_modified': 0,
        'prompt_unchanged': 0
    }
    
    print("Converting records...")
    for item in data:
        # ==================== File name ====================
        original_image = item['image']
        
        if original_image.endswith('.nii.gz'):
            new_image = original_image.replace('.nii.gz', '.npz')
            conversion_stats['nii_gz_to_npz'] += 1
        elif original_image.endswith('.nii'):
            new_image = original_image.replace('.nii', '.npz')
            conversion_stats['nii_to_npz'] += 1
        elif original_image.endswith('.npz'):
            new_image = original_image
            conversion_stats['already_npz'] += 1
        else:
            # No known extension: append .npz
            new_image = f"{original_image}.npz"
            conversion_stats['nii_gz_to_npz'] += 1
        
        # ==================== conversations ====================
        new_conversations = []
        
        for conv in item['conversations']:
            if conv['from'] == 'human':
                # Check whether the prompt carries disease predictions
                original_value = conv['value']
                
                if 'Known frontend model predictions' in original_value:
                    # Replace it with the plain prompt
                    new_value = prompt_template
                    conversion_stats['prompt_modified'] += 1
                else:
                    # Leave it untouched
                    new_value = original_value
                    conversion_stats['prompt_unchanged'] += 1
                
                new_conversations.append({
                    'type': conv.get('type', 'report_generation'),
                    'from': 'human',
                    'value': new_value
                })
            else:
                # Assistant replies are kept verbatim
                new_conversations.append(conv)
        
        # ==================== New record ====================
        converted_item = {
            'id': item.get('id', f"report_generation_{len(converted_data)}"),
            'image': new_image,
            'conversations': new_conversations
        }
        
        converted_data.append(converted_item)
    
    # ==================== Save ====================
    print(f"\nSaving to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=4, ensure_ascii=False)
    
    # ==================== Statistics ====================
    print(f"\n{'='*60}")
    print("Conversion complete.")
    print(f"{'='*60}")
    print(f"Total samples: {conversion_stats['total']}")
    print(f"\nFile name conversion:")
    print(f"  .nii.gz -> .npz: {conversion_stats['nii_gz_to_npz']}")
    print(f"  .nii -> .npz: {conversion_stats['nii_to_npz']}")
    print(f"  already .npz: {conversion_stats['already_npz']}")
    print(f"\nPrompt conversion:")
    print(f"  modified (predictions removed): {conversion_stats['prompt_modified']}")
    print(f"  unchanged: {conversion_stats['prompt_unchanged']}")
    print(f"{'='*60}\n")
    
    # ==================== Example ====================
    if converted_data:
        print("Converted example (first sample):")
        print("-"*60)
        print(f"ID: {converted_data[0]['id']}")
        print(f"Original image: {data[0]['image']}")
        print(f"Converted: {converted_data[0]['image']}")
        print(f"\nOriginal prompt:")
        print(data[0]['conversations'][0]['value'][:150] + "...")
        print(f"\nConverted prompt:")
        print(converted_data[0]['conversations'][0]['value'])
        print("-"*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Convert CT-CHAT JSON: .nii.gz -> .npz, drop disease predictions'
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='/path/to/CT-CHAT2/our_valid_data_mianyang/translated_reports.json',
        help='Path to the input JSON file'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='/path/to/CT-CHAT2/our_valid_data_mianyang/translated_reports_converted.json',
        help='Path to the output JSON file'
    )
    
    parser.add_argument(
        '--prompt', '-p',
        type=str,
        default='<image>\nPlease give the radiology report for the specified chest CT scan.<report_generation>',
        help='Custom prompt template'
    )
    
    args = parser.parse_args()
    
    # Run the conversion
    convert_json(args.input, args.output, args.prompt)


if __name__ == '__main__':
    main()