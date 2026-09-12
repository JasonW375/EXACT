import json
from collections import Counter
import re
import sys
import os

# Resolve the input file path
if len(sys.argv) > 1:
    json_file = sys.argv[1]
else:
    json_file = '/path/to/CT-CHAT2/VQA_dataset/by_category/report.json'

# Check that the file exists
if not os.path.exists(json_file):
    print(f"Error: file '{json_file}' does not exist")
    print(f"Usage: python {sys.argv[0]} <json_file_path>")
    sys.exit(1)

print(f"Reading file: {json_file}\n")

# Load the JSON file
with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Collect every report-generation question
questions = []

# Walk the data and extract the questions
for item in data:
    if 'conversations' in item:
        for conv in item['conversations']:
            if conv.get('from') == 'human' and conv.get('type') == 'report_generation':
                # Keep only the question text (drop <image> and the disease predictions)
                value = conv.get('value', '')
                
                # Extract the bare question with regular expressions
                # Remove the <image> tag
                question = re.sub(r'<image>', '', value)
                # Cut everything from "Known frontend model predictions" onwards
                question = re.split(r'Known frontend model predictions', question)[0]
                # Remove the <report_generation> tag
                question = re.sub(r'<report_generation>', '', question)
                # Trim surrounding whitespace
                question = question.strip()
                
                if question:
                    questions.append(question)

# Count how often each question occurs
question_counter = Counter(questions)

# Print the results
print(f"{len(questions)} questions in total")
print(f"{len(question_counter)} distinct question types\n")
print("=" * 80)
print("Question types and their frequencies:\n")

# Sort by descending frequency
for question, count in question_counter.most_common():
    print(f"Count: {count}")
    print(f"Question: {question}")
    print("-" * 80)

print("\nDone.")