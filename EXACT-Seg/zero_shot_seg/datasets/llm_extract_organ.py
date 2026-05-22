
import json
import os
from typing import Dict, Any
import time
import shutil
import tempfile
import requests


class OrganClassifier:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://www.dmxapi.cn/v1",
        model_name: str = "gpt-3.5-turbo"
    ):
        """
        Initialize the organ classifier using the dmxapi.cn API.

        Args:
            api_key: dmxapi.cn API key
            base_url: API base URL
            model_name: Model name
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.organ_names = [
            "lung",
            "trachea and bronchie",
            "pleura",
            "mediastinum",
            "heart",
            "esophagus",
            "global",
        ]

    def classify_finding(self, finding_text: str) -> str:
        """
        Call the dmxapi.cn API to determine which organ the finding belongs to.

        Args:
            finding_text: Finding description text

        Returns:
            Organ name, "global" (valid output but not matched), or "error" (API call failed)
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

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a medical expert specializing in radiology."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 20,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        endpoint = f"{self.base_url}/chat/completions"

        try:
            response = requests.post(endpoint, json=payload, headers=headers)
            response.raise_for_status()

            print(f"Debug: Raw response for '{finding_text[:50]}...': {response.text[:200]}...")

            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                organ = data["choices"][0]["message"]["content"].strip().lower()
            else:
                organ = data.get("content", "").strip().lower()
                if not organ:
                    raise ValueError("Unexpected response format")

            if organ not in self.organ_names:
                print(f"Warning: Unrecognized organ '{organ}' for finding: {finding_text}")
                organ = "global"

            return organ

        except requests.exceptions.RequestException as e:
            print(f"HTTP error when calling dmxapi.cn API for finding '{finding_text}': {e}")
            if "response" in locals() and response is not None:
                print(f"Response details: {response.text}")
            return "error"
        except (ValueError, KeyError) as e:
            print(f"Parsing error for finding '{finding_text}': {e}")
            return "error"
        except Exception as e:
            print(f"Unexpected error for finding '{finding_text}': {e}")
            return "error"

    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single sample and add the 'organ' field at the same level as 'findings'.
        """
        findings = sample.get("findings", {})

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
            time.sleep(0.4)

        sample["organ"] = organ_dict
        return sample

    def _process_splits(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data with split structure such as train/val/test.
        """
        for split_name, split_list in data.items():
            if isinstance(split_list, list):
                print(f"\n== Split: {split_name} (count={len(split_list)}) ==")
                for i, sample in enumerate(split_list):
                    if isinstance(sample, dict) and "findings" in sample:
                        print(f"[{split_name}] {i + 1}/{len(split_list)}")
                        self.process_sample(sample)
        return data

    def process_json_file(self, input_file: str, output_file: str = None, in_place: bool = False):
        """
        Process a JSON file. Supported structures:
        (1) top-level list
        (2) top-level single sample
        (3) top-level dict like {"train": [...], "val": [...], ...}
        """
        if in_place or output_file is None:
            output_file = input_file

        same_path = os.path.abspath(input_file) == os.path.abspath(output_file)
        if same_path:
            backup_file = input_file + ".bak"
            if not os.path.exists(backup_file):
                shutil.copyfile(input_file, backup_file)
                print(f"[Backup] Created: {backup_file}")
            else:
                print(f"[Backup] Already exists: {backup_file}")

        print(f"[Read] {input_file}")
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            print(f"Detected a top-level list with {len(data)} samples")
            for i, sample in enumerate(data):
                if isinstance(sample, dict) and "findings" in sample:
                    print(f"[List] {i + 1}/{len(data)}")
                    self.process_sample(sample)
        elif isinstance(data, dict):
            if any(isinstance(v, list) for v in data.values()):
                data = self._process_splits(data)
            elif "findings" in data:
                data = self.process_sample(data)
            else:
                print("No supported structure found. Keeping the file unchanged.")
        else:
            raise ValueError("Unsupported JSON structure")

        print(f"[Write] {output_file}")
        dir_name = os.path.dirname(os.path.abspath(output_file)) or "."
        fd, tmp_path = tempfile.mkstemp(prefix="organ_", suffix=".json", dir=dir_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tf:
                json.dump(data, tf, ensure_ascii=False, indent=2)
            os.replace(tmp_path, output_file)
            print("[Done] File has been updated")
        except Exception as e:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise e


def main():
    # Configuration
    API_KEY = os.getenv("DMX_API_KEY", "YOUR_API_KEY_HERE")
    BASE_URL = "https://www.dmxapi.cn/v1"
    MODEL_NAME = "gpt-3.5-turbo"
    INPUT_FILE = "/path/to/%%%/ReXGroundingCT/test.json"
    OUTPUT_FILE = "/path/to/%%%/ReXGroundingCT/output_with_organs.json"

    classifier = OrganClassifier(api_key=API_KEY, base_url=BASE_URL, model_name=MODEL_NAME)
    classifier.process_json_file(INPUT_FILE, OUTPUT_FILE)

    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        result = json.load(f)
        if isinstance(result, list) and len(result) > 0:
            print("\nExample output (first sample):")
            print(json.dumps(result[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
