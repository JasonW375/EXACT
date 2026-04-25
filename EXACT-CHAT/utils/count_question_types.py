import json
from collections import Counter
import re
import sys
import os

# 获取文件路径
if len(sys.argv) > 1:
    json_file = sys.argv[1]
else:
    json_file = '/path/to/CT-CHAT2/VQA_dataset/by_category/report.json'

# 检查文件是否存在
if not os.path.exists(json_file):
    print(f"错误: 文件 '{json_file}' 不存在")
    print(f"用法: python {sys.argv[0]} <json文件路径>")
    sys.exit(1)

print(f"正在读取文件: {json_file}\n")

# 读取JSON文件
with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 存储所有问题
questions = []

# 遍历数据提取问题
for item in data:
    if 'conversations' in item:
        for conv in item['conversations']:
            if conv.get('from') == 'human' and conv.get('type') == 'report_generation':
                # 提取问题部分（去除<image>标签和disease predictions部分）
                value = conv.get('value', '')
                
                # 使用正则表达式提取纯问题文本
                # 移除<image>标签
                question = re.sub(r'<image>', '', value)
                # 提取问题部分（在"Known frontend model predictions"之前）
                question = re.split(r'Known frontend model predictions', question)[0]
                # 移除<report_generation>标签
                question = re.sub(r'<report_generation>', '', question)
                # 清理前后空白
                question = question.strip()
                
                if question:
                    questions.append(question)

# 统计每种问题类型的出现次数
question_counter = Counter(questions)

# 输出结果
print(f"总共有 {len(questions)} 个问题")
print(f"不重复的问题类型有 {len(question_counter)} 种\n")
print("=" * 80)
print("各问题类型及其出现次数：\n")

# 按出现次数降序排列
for question, count in question_counter.most_common():
    print(f"次数: {count}")
    print(f"问题: {question}")
    print("-" * 80)

print("\n统计完成！")