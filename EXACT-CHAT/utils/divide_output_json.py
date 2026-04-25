import json

# 输入/输出文件路径
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
    # 检查每条item下的所有对话
    for conv in item['conversations_out']:
        for k, token in type_tokens.items():
            if token in conv['question']:
                # 新结构保留image和这条问答
                data_by_type[k].append({
                    'image': item['image'],
                    'question': conv['question'],
                    'answer': conv['answer']
                })

# 写入文件
for k in type_tokens:
    with open(f"{output_base}{k}.json", "w", encoding='utf-8') as fout:
        json.dump(data_by_type[k], fout, ensure_ascii=False, indent=2)

print('分类完成！')