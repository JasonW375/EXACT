import json

# 读取原始 JSON 文件
input_file = "/path/to/CT-CHAT2/our_valid_data/all_types_question_validation_results_prediction_from_heatmap/output_validation_new_llama_all_temp0.0_tokens1024_checkpoint-38000.json"
output_file = "/path/to/CT-CHAT2/our_valid_data/all_types_question_validation_results_prediction_from_heatmap/output_validation_new_llama_all_temp0.0_tokens1024_checkpoint-38000_cleaned.json"

# 读取 JSON 数据
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 遍历并清除 <|eot_id|>
for item in data:
    if 'conversations_out' in item:
        for conversation in item['conversations_out']:
            if 'answer' in conversation:
                # 移除 answer 中的 <|eot_id|>
                conversation['answer'] = conversation['answer'].replace('<|eot_id|>', '')

# 保存清理后的 JSON
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"处理完成！清理后的文件已保存至: {output_file}")
print(f"共处理 {len(data)} 条记录")