import json
import pandas as pd
import sys
import os

# 疾病列名
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
    """从图像文件名提取volume名称（去除后缀）"""
    # 处理类似 train_1_a_1.nii.gz 的文件名
    # 可能需要根据实际情况调整
    if image_filename.endswith('.nii.gz'):
        return image_filename.replace('.nii.gz', '')
    elif image_filename.endswith('.npz'):
        return image_filename.replace('.npz', '')
    else:
        return image_filename

def format_disease_predictions(predictions_dict):
    """格式化疾病预测为字符串"""
    predictions_str = "; ".join([f"{disease}={int(predictions_dict[disease])}" 
                                  for disease in DISEASE_COLUMNS])
    return predictions_str

def add_predictions_to_human_value(original_value, predictions_str):
    """
    在human的value中添加疾病预测信息
    假设原始value格式为: <image>\n问题<report_generation>
    添加后格式为: <image>\n问题 Known frontend model predictions (disease-wise): 预测结果.<report_generation>
    """
    # 移除原有的<report_generation>标签
    value = original_value.replace('<report_generation>', '').strip()
    
    # 添加预测信息和标签
    new_value = f"{value} Known frontend model predictions (disease-wise): {predictions_str}.<report_generation>"
    
    return new_value

def filter_and_add_predictions(input_json_path, csv_path, output_json_path):
    """
    从训练JSON中筛选report generation条目并添加分类预测
    
    参数:
    - input_json_path: 输入的训练JSON文件路径
    - csv_path: disease_predictions.csv文件路径
    - output_json_path: 输出的筛选后JSON文件路径
    """
    # 读取CSV文件
    print(f"正在读取CSV文件: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"CSV包含 {len(df)} 条记录")
    
    # 将VolumeName设为索引以便快速查找
    df.set_index('VolumeName', inplace=True)
    csv_volumes = set(df.index)
    print(f"CSV中的volume名称: {list(csv_volumes)[:5]}... (showing first 5)")
    
    # 读取输入JSON文件
    print(f"\n正在读取输入JSON文件: {input_json_path}")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    print(f"输入JSON包含 {len(input_data)} 条记录")
    
    # 筛选并处理数据
    filtered_data = []
    matched_count = 0
    unmatched_count = 0
    non_report_count = 0
    
    for item in input_data:
        # 检查是否有conversations字段
        if 'conversations' not in item:
            continue
        
        # 检查是否是report_generation类型
        is_report_generation = False
        for conv in item['conversations']:
            if conv.get('type') == 'report_generation':
                is_report_generation = True
                break
        
        if not is_report_generation:
            non_report_count += 1
            continue
        
        # 获取图像文件名和volume名称
        image_filename = item.get('image', '')
        volume_name = get_volume_name_from_image(image_filename)
        
        # 检查该volume是否在CSV中
        if volume_name not in csv_volumes:
            unmatched_count += 1
            continue
        
        matched_count += 1
        
        # 获取该volume的疾病预测
        predictions = df.loc[volume_name]
        predictions_dict = predictions.to_dict()
        
        # 格式化疾病预测字符串
        disease_str = format_disease_predictions(predictions_dict)
        
        # 创建新的条目（深拷贝）
        new_item = {
            "id": item['id'],
            "image": item['image'],
            "conversations": []
        }
        
        # 处理每个conversation
        for conv in item['conversations']:
            new_conv = conv.copy()
            
            # 如果是report_generation类型的human消息，添加预测信息
            if (conv.get('from') == 'human' and 
                conv.get('type') == 'report_generation'):
                original_value = conv.get('value', '')
                new_conv['value'] = add_predictions_to_human_value(original_value, disease_str)
            
            new_item['conversations'].append(new_conv)
        
        filtered_data.append(new_item)
    
    # 输出统计信息
    print(f"\n统计信息:")
    print(f"- 输入JSON总记录数: {len(input_data)}")
    print(f"- 非report_generation类型: {non_report_count}")
    print(f"- report_generation类型但不在CSV中: {unmatched_count}")
    print(f"- 成功匹配并筛选: {matched_count}")
    print(f"- 输出JSON记录数: {len(filtered_data)}")
    
    # 保存输出JSON文件
    print(f"\n正在保存输出JSON文件到: {output_json_path}")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_json_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=4, ensure_ascii=False)
    
    print(f"完成! 筛选出 {len(filtered_data)} 条report generation记录")
    
    # 显示第一条示例
    if filtered_data:
        print(f"\n第一条记录示例:")
        print(f"ID: {filtered_data[0]['id']}")
        print(f"Image: {filtered_data[0]['image']}")
        if filtered_data[0]['conversations']:
            first_conv = filtered_data[0]['conversations'][0]
            print(f"First conversation type: {first_conv.get('type')}")
            print(f"Value preview: {first_conv.get('value', '')[:150]}...")
    
    return filtered_data

if __name__ == "__main__":
    # 默认路径
    input_json_path = "/path/to/CT-CHAT2/VQA_dataset/filtered_train_vqa.json"
    csv_path = "/path/to/CT_Report/CT_Report16_classification/heatmap_ft/分布内valid/disease_predictions.csv"
    output_json_path = "/path/to/CT-CHAT2/our_train_data/class_invalid_report_generation.json"
    
    # 允许从命令行参数指定路径
    if len(sys.argv) >= 2:
        input_json_path = sys.argv[1]
    if len(sys.argv) >= 3:
        csv_path = sys.argv[2]
    if len(sys.argv) >= 4:
        output_json_path = sys.argv[3]
    
    print("="*80)
    print("筛选Report Generation条目并添加分类预测")
    print("="*80)
    print(f"输入JSON: {input_json_path}")
    print(f"CSV文件: {csv_path}")
    print(f"输出JSON: {output_json_path}")
    print("="*80)
    
    # 检查路径是否存在
    if not os.path.exists(input_json_path):
        print(f"错误: 输入JSON文件不存在: {input_json_path}")
        sys.exit(1)
    
    if not os.path.exists(csv_path):
        print(f"错误: CSV文件不存在: {csv_path}")
        sys.exit(1)
    
    # 执行筛选和添加预测
    filter_and_add_predictions(input_json_path, csv_path, output_json_path)