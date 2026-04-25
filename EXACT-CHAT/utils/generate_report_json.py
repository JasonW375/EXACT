import json
import os
import random
import pandas as pd
import sys

# 定义所有可能的问题模板
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

def get_volume_name_from_npz(npz_filename):
    """从npz文件名提取volume名称（去除.npz后缀）"""
    return npz_filename.replace('.npz', '')

def format_disease_predictions(predictions_dict):
    """格式化疾病预测为字符串"""
    predictions_str = "; ".join([f"{disease}={int(predictions_dict[disease])}" 
                                  for disease in DISEASE_COLUMNS])
    return predictions_str

def generate_json_data(embeddings_dir, csv_path, output_json_path, seed=42):
    """
    生成新的JSON文件
    
    参数:
    - embeddings_dir: embeddings目录路径
    - csv_path: disease_predictions.csv文件路径
    - output_json_path: 输出的JSON文件路径
    - seed: 随机种子
    """
    # 设置随机种子以保证可重复性
    random.seed(seed)
    
    # 读取CSV文件
    print(f"正在读取CSV文件: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"CSV包含 {len(df)} 条记录")
    
    # 将VolumeName设为索引以便快速查找
    df.set_index('VolumeName', inplace=True)
    
    # 获取embeddings目录下的所有npz文件
    print(f"\n正在扫描embeddings目录: {embeddings_dir}")
    npz_files = [f for f in os.listdir(embeddings_dir) if f.endswith('.npz')]
    npz_files.sort()  # 排序以保证顺序一致
    print(f"找到 {len(npz_files)} 个npz文件")
    
    # 生成JSON数据
    json_data = []
    matched_count = 0
    unmatched_volumes = []
    
    for idx, npz_file in enumerate(npz_files):
        volume_name = get_volume_name_from_npz(npz_file)
        
        # 检查该volume是否在CSV中
        if volume_name not in df.index:
            unmatched_volumes.append(volume_name)
            continue
        
        matched_count += 1
        
        # 获取该volume的疾病预测
        predictions = df.loc[volume_name]
        predictions_dict = predictions.to_dict()
        
        # 格式化疾病预测字符串
        disease_str = format_disease_predictions(predictions_dict)
        
        # 随机选择一个问题模板
        question = random.choice(QUESTION_TEMPLATES)
        
        # 构建完整的human消息
        human_value = f"<image>\n{question} Known frontend model predictions (disease-wise): {disease_str}.<report_generation>"
        
        # 构建JSON条目
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
    
    # 输出统计信息
    print(f"\n统计信息:")
    print(f"- 总NPZ文件数: {len(npz_files)}")
    print(f"- 成功匹配: {matched_count}")
    print(f"- 未匹配: {len(unmatched_volumes)}")
    
    if unmatched_volumes:
        print(f"\n前10个未匹配的volume:")
        for vol in unmatched_volumes[:10]:
            print(f"  - {vol}")
        if len(unmatched_volumes) > 10:
            print(f"  ... 还有 {len(unmatched_volumes) - 10} 个")
    
    # 保存JSON文件
    print(f"\n正在保存JSON文件到: {output_json_path}")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"完成! 生成了 {len(json_data)} 条记录")
    
    return json_data

if __name__ == "__main__":
    # 默认路径
    embeddings_dir = "/path/to/CT-CHAT2/our_valid_data_radchest/embeddings"
    csv_path = "/path/to/CT_Report/CT_Report16_classification/heatmap_ft/radchest/disease_predictions.csv"
    output_json_path = "/path/to/CT-CHAT2/our_valid_data_radchest/report_generation.json"
    
    # 允许从命令行参数指定路径
    if len(sys.argv) >= 2:
        embeddings_dir = sys.argv[1]
    if len(sys.argv) >= 3:
        csv_path = sys.argv[2]
    if len(sys.argv) >= 4:
        output_json_path = sys.argv[3]
    
    print("="*80)
    print("生成Report Generation JSON文件")
    print("="*80)
    print(f"Embeddings目录: {embeddings_dir}")
    print(f"CSV文件: {csv_path}")
    print(f"输出JSON: {output_json_path}")
    print("="*80)
    
    # 检查路径是否存在
    if not os.path.exists(embeddings_dir):
        print(f"错误: Embeddings目录不存在: {embeddings_dir}")
        sys.exit(1)
    
    if not os.path.exists(csv_path):
        print(f"错误: CSV文件不存在: {csv_path}")
        sys.exit(1)
    
    # 生成JSON
    generate_json_data(embeddings_dir, csv_path, output_json_path)