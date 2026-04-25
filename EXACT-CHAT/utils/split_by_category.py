import json
import os
from collections import defaultdict

# 文件路径
input_json_path = "/path/to/CT-CHAT2/VQA_dataset/filtered_valid_vqa_with_preds_from_heatmap.json"
output_dir = "/path/to/CT-CHAT2/VQA_dataset/by_category_from_heatmap"

# 类型标记
type_tokens = {
    'long_answer': '<long_answer>',
    'short_answer': '<short_answer>',
    'multiple_choice': '<multiple_choice>',
    'report_generation': '<report_generation>'
}

# 检查输入文件是否存在
if not os.path.exists(input_json_path):
    print(f"Error: Input file not found: {input_json_path}")
    exit(1)

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

# 读取JSON文件
print("Reading JSON file...")
with open(input_json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Loaded {len(data)} entries from JSON")

# 按类别分类数据
categorized_data = {
    'long_answer': [],
    'short_answer': [],
    'multiple_choice': [],
    'report_generation': []
}

# 统计信息
stats = defaultdict(int)

# 处理每个条目
for entry in data:
    entry_id = entry.get('id', '')
    
    # 判断条目属于哪个类别
    category = None
    
    # 方法1: 通过id前缀判断
    for cat_name in type_tokens.keys():
        if entry_id.startswith(cat_name):
            category = cat_name
            break
    
    # 方法2: 如果id没有匹配，通过conversations中的token判断
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
    
    # 添加到对应类别
    if category:
        categorized_data[category].append(entry)
        stats[category] += 1
    else:
        stats['unknown'] += 1
        print(f"Warning: Could not determine category for entry: {entry_id}")

# 输出统计信息
print("\n" + "="*50)
print("Categorization Statistics:")
print("="*50)
for category in type_tokens.keys():
    count = stats[category]
    print(f"{category:20s}: {count:6d} entries")
print(f"{'unknown':20s}: {stats['unknown']:6d} entries")
print(f"{'TOTAL':20s}: {sum(stats.values()):6d} entries")
print("="*50)

# 保存每个类别的JSON文件
print("\nSaving categorized files...")
for category, entries in categorized_data.items():
    if entries:  # 只保存非空的类别
        output_path = os.path.join(output_dir, f"{category}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(entries)} entries to: {output_path}")

print("\nDone!")