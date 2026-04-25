import json
import os
from typing import Dict, List, Any
import time
import shutil
import tempfile
from openai import OpenAI  # OpenAI SDK
import openai
import requests
class OrganClassifier:
    def __init__(self, api_key: str, base_url: str = "https://www.dmxapi.cn/v1", model_name: str = "gpt-3.5-turbo"):
        """
        初始化器官分类器，使用 dmxapi.cn API
        
        Args:
            api_key: dmxapi.cn API密钥
            base_url: API基URL（修改为 /v1，根据文档确认）
            model_name: 模型名称（默认为 gpt-3.5-turbo，更常见支持）
        """
        self.api_key = api_key
        self.base_url = base_url  # 修改：默认为 https://www.dmxapi.cn/v1
        self.model_name = model_name
        self.organ_names = ["lung", "trachea and bronchie", "pleura", 
                           "mediastinum", "heart", "esophagus", "global"]
    
    def classify_finding(self, finding_text: str) -> str:
        """
        调用 dmxapi.cn API 判断异常描述对应的器官（使用 requests）
        
        Args:
            finding_text: 异常描述文本
            
        Returns:
            器官名称、"global"（正常但不匹配）或 "error"（API 调用失败）
        """
        prompt = f"""You are a medical expert. Given the following medical finding description, 
        identify which organ it primarily relates to.
        
        Finding: {finding_text}
        
        Choose from the following organs:
        - lung: for lungs, pulmonary, lobes, segments
        - trachea and bronchie: for trachea, bronchi, airways
        - pleura: for pleural cavity, pleural effusion
        - mediastinum: for mediastinal structures, lymph nodes in mediastinum
        - heart: for cardiac, pericardium, coronary
        - esophagus: for esophageal structures
        - global: if it doesn't belong to any of the above specific organs
        
        Respond with ONLY the organ name (exactly as shown above), nothing else."""
        
        # API 请求体（兼容 OpenAI 格式）
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a medical expert specializing in radiology."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 20
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        endpoint = f"{self.base_url}/chat/completions"  # 假设端点；根据文档调整
        
        try:
            response = requests.post(endpoint, json=payload, headers=headers)
            response.raise_for_status()  # 抛出 HTTP 错误
            
            # 打印原始响应以调试（可选）
            print(f"Debug: Raw response for '{finding_text[:50]}...': {response.text[:200]}...")
            
            data = response.json()  # 解析 JSON
            # 访问 choices（假设兼容 OpenAI 格式）
            if 'choices' in data and len(data['choices']) > 0:
                organ = data['choices'][0]['message']['content'].strip().lower()
            else:
                # 如果格式不同，尝试其他路径（例如，直接 'content' 字段）
                organ = data.get('content', '').strip().lower()
                if not organ:
                    raise ValueError("Unexpected response format")
            
            # 验证：如果正常输出但不属于列表（不匹配六个器官），设置为 "global"
            if organ not in self.organ_names:
                print(f"Warning: Unrecognized organ '{organ}' (normal output but not in list) for finding: {finding_text}")
                organ = "global"
            
            return organ
            
        except requests.exceptions.RequestException as e:
            print(f"HTTP error calling dmxapi.cn API for finding '{finding_text}': {e}")
            if 'response' in locals() and response is not None:
                print(f"Response details: {response.text}")
            return "error"  # 修改：错误时返回 "error"
        except (ValueError, KeyError) as e:
            print(f"Parsing error for finding '{finding_text}': {e}")
            return "error"  # 修改：错误时返回 "error"
        except Exception as e:
            print(f"Unexpected error for finding '{finding_text}': {e}")
            return "error"  # 修改：错误时返回 "error"
    
    # 以下方法保持不变（process_sample, _process_splits, process_json_file 等）
    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单个样本，添加organ字段（与findings平级）
        """
        findings = sample.get("findings", {})
        # 若已存在 organ 且键数量匹配，跳过
        if "organ" in sample and isinstance(sample["organ"], dict) and len(sample["organ"]) == len(findings):
            print(f"Skip (already has organ): {sample.get('name', 'Unknown')}")
            return sample

        organ_dict = {}
        print(f"Processing sample: {sample.get('name', 'Unknown')}")
        for key, finding_text in findings.items():
            print(f"  Classifying finding {key}: {finding_text[:60]}...")
            organ = self.classify_finding(finding_text)
            organ_dict[key] = organ
            print(f"    -> {organ}")
            time.sleep(0.4)  # 轻微延迟避免速率限制
        sample["organ"] = organ_dict
        return sample

    def _process_splits(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理带有 train/val/test 等分割的结构
        """
        for split_name, split_list in data.items():
            if isinstance(split_list, list):
                print(f"\n== Split: {split_name} (count={len(split_list)}) ==")
                for i, sample in enumerate(split_list):
                    if isinstance(sample, dict) and "findings" in sample:
                        print(f"[{split_name}] {i+1}/{len(split_list)}")
                        self.process_sample(sample)
        return data

    def process_json_file(self, input_file: str, output_file: str = None, in_place: bool = False):
        """
        处理JSON文件：支持
        1) 顶层为列表
        2) 顶层为单个样本
        3) 顶层为 { "train": [...], "val": [...], ... }
        """
        if in_place or output_file is None:
            output_file = input_file

        same_path = os.path.abspath(input_file) == os.path.abspath(output_file)
        if same_path:
            backup_file = input_file + ".bak"
            if not os.path.exists(backup_file):
                shutil.copyfile(input_file, backup_file)
                print(f"[备份] 已创建: {backup_file}")
            else:
                print(f"[备份] 已存在: {backup_file}")

        print(f"[读取] {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            print(f"检测到顶层为列表，共 {len(data)} 个样本")
            for i, sample in enumerate(data):
                if isinstance(sample, dict) and "findings" in sample:
                    print(f"[List] {i+1}/{len(data)}")
                    self.process_sample(sample)
        elif isinstance(data, dict):
            # 判断是否是包含 split 的结构
            if any(isinstance(v, list) for v in data.values()):
                data = self._process_splits(data)
            elif "findings" in data:
                data = self.process_sample(data)
            else:
                print("未发现可处理的结构，保持原样。")
        else:
            raise ValueError("不支持的JSON结构")

        # 原子写回
        print(f"[写入] {output_file}")
        dir_name = os.path.dirname(os.path.abspath(output_file)) or "."
        fd, tmp_path = tempfile.mkstemp(prefix="organ_", suffix=".json", dir=dir_name)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as tf:
                json.dump(data, tf, ensure_ascii=False, indent=2)
            os.replace(tmp_path, output_file)
            print("[完成] 已更新文件")
        except Exception as e:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise e

def main():
    # 配置参数
    API_KEY = "sk-Ri5FYf6PpM9upp6uZlHlaeDWTqOwr1GhcuC60i8I5xTfVPNl"  # dmxapi.cn API密钥
    BASE_URL = "https://www.dmxapi.cn/v1"  # 修改：添加 /v1
    MODEL_NAME = "gpt-3.5-turbo"  # 修改：尝试更常见的模型
    INPUT_FILE = "/FM_data/bxg/CT_Report/CT_Report9_test/ReXGroundingCT/test.json"  # 输入文件路径
    OUTPUT_FILE = "/FM_data/bxg/CT_Report/CT_Report9_test/ReXGroundingCT/output_with_organs.json"  # 输出文件路径
    
    # 创建分类器实例
    classifier = OrganClassifier(api_key=API_KEY, base_url=BASE_URL, model_name=MODEL_NAME)
    
    # 处理文件
    classifier.process_json_file(INPUT_FILE, OUTPUT_FILE)
    
    # 可选：显示处理结果示例
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        result = json.load(f)
        if isinstance(result, list) and len(result) > 0:
            print("\n示例输出（第一个样本）:")
            print(json.dumps(result[0], ensure_ascii=False, indent=2))

# 批处理版本和测试函数保持不变
# ...（从您的代码复制 batch_process 和 test_single_finding）

if __name__ == "__main__":
    # 运行主程序
    main()
    
    # 或者运行批处理
    # batch_process()
    
    # 或者测试单个finding
    # test_single_finding()