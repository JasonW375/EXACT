#!/usr/bin/env python3
"""
转换 translated_reports.json 格式
1. 将 .nii.gz 文件名改为 .npz
2. 移除疾病预测信息，只保留基础prompt
"""
import json
import argparse
from pathlib import Path


def convert_json(input_path, output_path, prompt_template=None):
    """
    转换JSON格式
    
    Args:
        input_path: 输入JSON文件路径
        output_path: 输出JSON文件路径
        prompt_template: 自定义prompt模板 (可选)
    """
    
    # 默认prompt模板
    if prompt_template is None:
        prompt_template = "<image>\nPlease give the radiology report for the specified chest CT scan.<report_generation>"
    
    print(f"\n{'='*60}")
    print("JSON格式转换脚本")
    print(f"{'='*60}")
    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"Prompt模板: {prompt_template}")
    print(f"{'='*60}\n")
    
    # 读取原始JSON
    print("正在读取JSON文件...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✓ 读取了 {len(data)} 个样本\n")
    
    # 转换数据
    converted_data = []
    conversion_stats = {
        'total': len(data),
        'nii_gz_to_npz': 0,
        'nii_to_npz': 0,
        'already_npz': 0,
        'prompt_modified': 0,
        'prompt_unchanged': 0
    }
    
    print("正在转换数据...")
    for item in data:
        # ==================== 转换文件名 ====================
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
            # 如果没有后缀，添加.npz
            new_image = f"{original_image}.npz"
            conversion_stats['nii_gz_to_npz'] += 1
        
        # ==================== 处理conversations ====================
        new_conversations = []
        
        for conv in item['conversations']:
            if conv['from'] == 'human':
                # 检查是否包含疾病预测
                original_value = conv['value']
                
                if 'Known frontend model predictions' in original_value:
                    # 替换为简单prompt
                    new_value = prompt_template
                    conversion_stats['prompt_modified'] += 1
                else:
                    # 保持原样
                    new_value = original_value
                    conversion_stats['prompt_unchanged'] += 1
                
                new_conversations.append({
                    'type': conv.get('type', 'report_generation'),
                    'from': 'human',
                    'value': new_value
                })
            else:
                # gpt回复保持不变
                new_conversations.append(conv)
        
        # ==================== 构建新条目 ====================
        converted_item = {
            'id': item.get('id', f"report_generation_{len(converted_data)}"),
            'image': new_image,
            'conversations': new_conversations
        }
        
        converted_data.append(converted_item)
    
    # ==================== 保存结果 ====================
    print(f"\n正在保存到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=4, ensure_ascii=False)
    
    # ==================== 打印统计信息 ====================
    print(f"\n{'='*60}")
    print("✓ 转换完成！")
    print(f"{'='*60}")
    print(f"总样本数: {conversion_stats['total']}")
    print(f"\n文件名转换:")
    print(f"  .nii.gz → .npz: {conversion_stats['nii_gz_to_npz']}")
    print(f"  .nii → .npz: {conversion_stats['nii_to_npz']}")
    print(f"  已是 .npz: {conversion_stats['already_npz']}")
    print(f"\nPrompt转换:")
    print(f"  已修改（移除疾病预测）: {conversion_stats['prompt_modified']}")
    print(f"  保持不变: {conversion_stats['prompt_unchanged']}")
    print(f"{'='*60}\n")
    
    # ==================== 显示示例 ====================
    if converted_data:
        print("转换后的示例（第一个样本）:")
        print("-"*60)
        print(f"ID: {converted_data[0]['id']}")
        print(f"原始图像: {data[0]['image']}")
        print(f"转换后: {converted_data[0]['image']}")
        print(f"\n原始Prompt:")
        print(data[0]['conversations'][0]['value'][:150] + "...")
        print(f"\n转换后Prompt:")
        print(converted_data[0]['conversations'][0]['value'])
        print("-"*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='转换CT-CHAT JSON格式：.nii.gz→.npz，移除疾病预测'
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='/path/to/CT-CHAT2/our_valid_data_mianyang/translated_reports.json',
        help='输入JSON文件路径'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='/path/to/CT-CHAT2/our_valid_data_mianyang/translated_reports_converted.json',
        help='输出JSON文件路径'
    )
    
    parser.add_argument(
        '--prompt', '-p',
        type=str,
        default='<image>\nPlease give the radiology report for the specified chest CT scan.<report_generation>',
        help='自定义prompt模板'
    )
    
    args = parser.parse_args()
    
    # 执行转换
    convert_json(args.input, args.output, args.prompt)


if __name__ == '__main__':
    main()