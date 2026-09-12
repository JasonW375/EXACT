import os
import json
import time
import random
import argparse
import pandas as pd
from openai import OpenAI

class CTReportTranslator:
    def __init__(self, excel_file, disease_csv, output_dir, api_key, base_url, batch_size=50):
        self.excel_file = excel_file
        self.disease_csv = disease_csv
        self.output_dir = output_dir
        self.batch_size = batch_size
        
        # Output and checkpoint files.
        self.output_file = os.path.join(output_dir, "translated_reports.json")
        self.checkpoint_file = os.path.join(output_dir, "checkpoint.json")
        
        # OpenAI-compatible client.
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        # Question templates, sampled at random for each report.
        self.question_templates = [
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
            "Please give the radiology report for the specified chest CT image?",
            "I need a detailed report for the given chest CT scan.",
            "Provide the radiology report for this CT image."
        ]
        
        # System prompt for the translation task.
        self.system_prompt = """You are a professional medical imaging report translator specializing in chest CT reports.
Your task is to translate Chinese CT reports into English while strictly following the style and structure of the provided English examples.

CRITICAL STYLE REQUIREMENTS:
1. **Report Structure**: Use ONLY "Findings:" and "Impression:" sections
2. **Format**: Match the example reports exactly:
   - "Findings:" section: continuous paragraphs describing anatomical structures systematically
   - "Impression:" section: concise summary separated by double spaces (e.g., "Finding1.  Finding2.  Finding3.")
3. **Terminology**: Use the exact medical terminology from examples (e.g., "calibration", "as far as can be observed", "no occlusive pathology")
4. **Sentence Patterns**: Mirror the sentence structures from examples:
   - "Trachea and both main bronchi are open."
   - "No pathologically enlarged lymph nodes were detected..."
   - "When examined in the lung parenchyma window;"
   - "Mediastinal structures cannot be evaluated optimally because contrast material is not given."

TRANSLATION PRINCIPLES:
1. **Accuracy**: Translate all medical findings accurately
2. **Completeness**: Include all information from the Chinese report
3. **Style Consistency**: Make the output indistinguishable from the example reports
4. **Professional Tone**: Maintain formal medical language
5. **Systematic Order**: Follow anatomical order (mediastinum → lungs → upper abdomen → bones)

The goal is to produce an English report that reads as if it were originally written by the same radiologist who wrote the example reports.
"""
        
        # Example reports that define the target style.
        self.example_reports = """## Example English CT Reports

### Example 1: Normal/Minimal Findings
**Report:**
Findings: Trachea and both main bronchi were open and no obstructive pathology was detected. Mediastinal vascular structures could not be optimally evaluated due to the absence of IV contrast in the cardiac examination, and the calibration of the vascular structures, heart contour and size are normal as far as can be observed. No pericardial-pleural effusion or increased thickness was detected. No pathological increase in wall thickness was observed in the thoracic esophagus. No lymph node was detected in the mediastinum and in both axillary regions in pathological size and appearance. In the evaluation made in the lung parenchyma window: No active infiltration or mass lesion was detected in both lungs. Ventilation of both lungs is natural. No lytic or destructive lesions were observed in the bone structures within the image. Vertebral corpus heights are preserved. Bilateral neural foramina are open. Impression:  Findings within normal limits.

---

### Example 2: Hiatal Hernia
**Report:**
Findings: Trachea is in the midline of both main bronchi and no obstructive parotology is observed in the lumen. The mediastinum could not be evaluated optimally in the non-contrast examination. As far as can be seen; Calibration of mediastinal major vascular structures is natural. Heart contour, size is normal. Pericardial effusion-thickening was not observed. In the mediastinum, lymph nodes with short axes below 1 cm that did not reach pathological dimensions were observed. Thoracic esophagus calibration was normal and no significant tumoral wall thickening was detected. Sliding type hiatal hernia was observed at the lower end of the esophagus. When examined in the lung parenchyma window; Aeration of both lung parenchyma is normal and no nodular or infiltrative lesion is detected in the lung parenchyma. Pleural effusion-thickening was not detected. Upper abdominal organs included in the sections are normal. No space-occupying lesion was detected in the liver that entered the cross-sectional area. Bilateral adrenal glands were normal and no space-occupying lesion was detected. At the thoracic level, mild scoliosis with right-facing scoliosis was observed. Vertebral corpus heights are preserved. Impression: Hiatal hernia . Mild scoliosis with right thoracic opening

---

### Example 3: Emphysema and Atelectasis
**Report:**
Findings: Mediastinal structures were evaluated as suboptimal since the examination was unenhanced. As far as can be seen; Trachea and lumen of both main bronchi are open. No occlusive pathology was detected in the trachea and lumen of both main bronchi. Calibration of thoracic main vascular structures is natural. No dilatation was detected in the thoracic aorta. Heart contour size is natural. Pericardial thickening-effusion was not detected. Thoracic esophagus calibration was normal and no significant pathological wall thickening was detected. No lymph node was detected in mediastinal and bilateral hilar pathological size and appearance. When examined in the lung parenchyma window; Subsegmental atelectatic changes were observed in the left lung inferior lingular segment. Mild emphysematous changes are present in both lungs. Subsegmental atelectatic changes were observed in the left lung lower lobe mediobasal segment. Subsegmental atelectasis was observed in the medial segment of the right lung middle lobe. Upper abdominal sections entering the examination area are natural. Bilateral adrenal gland calibration was normal and no space-occupying lesion was detected. No lytic-destructive lesion was detected in bone structures. Impression:  Mild emphysematous changes in both lungs, subsegmental atelectasis in both lungs.

---

### Example 4: Bronchiectasis and Peribronchial Thickening
**Report:**
Findings: Trachea and both main bronchi were open and no obstructive pathology was detected. Mediastinal vascular structures could not be evaluated optimally because the cardiac examination was without IV contrast. Calibration of vascular structures, heart contour, size are natural. Pericardial-pleural effusion was not detected. No pathological increase in wall thickness is observed in the thoracic esophagus. In the mediastinum, in both axillary regions and in the supraclavicular fossa, no lymph nodes are observed in pathological size and appearance. In the evaluation made in the lung parenchyma window: No active infiltration or mass lesion was detected in both lungs. There are milimal emphysematous changes in both lungs. There are diffuse mild ectasia and peribronchial thickness increases in bilateral bronchial structures. In the upper abdominal sections within the image, no pathology was detected as far as it can be observed within the borders of non-contrast CT. No lytic or destructive lesions were detected in the bone structures within the image. Impression:  Diffuse mild ectasia and peribronchial thickness increases in bronchial structures in both lungs, minimal emphysematous changes.

---

### Example 5: Covid-19 Pneumonia
**Report:**
Findings: Trachea, both main bronchi are open. Mediastinal main vascular structures, heart contour, size are normal. Thoracic aorta diameter is normal. Pericardial effusion-thickening was not observed. Thoracic esophagus calibration was normal and no significant tumoral wall thickening was detected. No enlarged lymph nodes in prevascular, pre-paratracheal, subcarinal or bilateral hilar-axillary pathological dimensions were detected. In general, peripherally located consolidation-ground glass areas are observed in both lungs. The outlook is compatible with Covid-19 pneumonia. Upper abdominal organs included in the sections are normal. No space-occupying lesion was detected in the liver that entered the cross-sectional area. Bilateral adrenal glands were normal and no space-occupying lesion was detected. Bone structures in the study area are natural. Vertebral corpus heights are preserved. Impression:  Typical-probable Covid-19 pneumonia.

---

### Example 6: Pleural Effusion and Lymphadenopathy
**Report:**
Findings:  Mediastinal structures were evaluated as suboptimal because the examination was unenhanced. As far as can be seen; The left breast was not observed (operated). No mass lesion with discernible borders was observed in the right breast. Conglomerate lymphadenopathies associated with each other in the paraesophageal area, adjacent to the bilateral infra-supraclavicular, right upper-lower paratracheal, left lower paratracheal, subcarinal, right hilar and right lower lobe bronchi are observed. It was measured in the short axis of the right upper paratracheal area (35 mm in the previous examination). Trachea, both main bronchi are open. No occlusive pathology was observed in the lumen. Mediastinal main vascular structures, heart contour, size are normal. In the pericardial space, an effusion reaching 7 mm in thickness is observed at its thickest part (15 mm at its thickest part in the previous examination). Thoracic esophagus calibration was normal and no significant tumoral wall thickening was detected. When examined in the lung parenchyma window; Effusion reaching a thickness of 32 mm in the right pleural space (27 mm in the previous examination) and reaching a thickness of 10 mm in the left pleural space was observed. A mosaic attenuation pattern is observed in both lungs (small airway disease? small vessel disease?). It is recommended to be evaluated together with the clinic. In the middle and lower lobes of the right lung, the most prominent interlobar-interlobular septal thickening in the middle lobe and focal ground-glass densities were observed in the peripheral subpleural areas of both lungs. Thickening is observed in the bilateral peribronchovascular interstitium. Findings were evaluated as secondary to infective-inflammatory processes. Fibroatelectasis sequelae are observed in the left lung inferior lingular segment and right lung middle and lower lobe. No mass lesion with distinguishable borders was detected in both lungs. Liver, gallbladder, spleen, both adrenal glands and pancreas are normal as far as can be seen on non-contrast images. No stones were observed in both kidneys. Left-facing scoliosis was observed in the thoracic vertebral column. Vertebral corpus heights are normal. No lytic-destructive lesion in favor of metastasis was observed in bone structures. Impression: Findings were evaluated as secondary to infective-inflammatory events.  Left-facing scoliosis in the thoracic vertebral column.

---

### Example 7: Lung Nodules and Calcifications
**Report:**
Findings: CTO is within the normal range. Calibration of the main mediastinal vascular structures is natural. In the anterior mediastinum, there is thymic tissue in conical configuration in which hypodense areas compatible with fatty involution are observed. It does not show a significant mass effect. No pathologically sized and configured lymph nodes were detected in the mediastinum and at both hilar levels. Thoracic esophagus calibration was normal and no significant tumoral wall thickening was detected. When examined in the lung parenchyma window; Calibration of trachea and main bronchi is normal. Both hemithorax are symmetrical. There is a decrease in density consistent with emphysema in both lungs. A subpleural nodule with a diameter of 2 mm is observed in the middle lobe of the right lung. There are densities in the right lung and laterobasal segment that are considered compatible with pleuroparenchymal sequelae. A subpleural nodule with a diameter of 2 mm is observed at the laterobasal level of the left lung. There was no finding compatible with bilateral pleural effusion, pneumothorax, pneumonia. At the apical level of the left lung, bone fragments and a density compatible with foreign body are observed between the muscle planes posteriorly. In the upper abdominal organs, including sections; A decrease in density consistent with steatosis is observed in the liver. No space-occupying lesion was detected in the liver that entered the cross-sectional area. The gallbladder is slightly contracted. Bilateral adrenal glands were normal and no space-occupying lesion was detected. At the level of the thorax inlet, there are densities compatible with the foreign body in the intermuscular fascia level on the right, and in the subcutaneous soft tissue planes on the left, more caudally in the subcutaneous soft tissue planes on the right. Again, at the level of the left hemithorax, adjacent to the intercostal musculature, densities compatible with multiple foreign bodies are observed in the subcutaneous soft tissue planes bilaterally more caudally. There is a density compatible with a superposed foreign body on the outer cortex in the 7th rib posterior on the left. Degenerative changes are observed in the bone structure. There are sequelae changes and densities compatible with the foreign body at the level of the left scapula body-coracoid process. Impression:  Findings consistent with emphysema in both lungs, a few millimetric non-specific nodules formation.  Density compatible with subcutaneous fatty planes and multiple foreign bodies superposed to muscle planes in the posterior and lateral sections of both lungs.  Density compatible with the foreign body superposed on the outer cortex at the 7th rib posterior on the left.  Post-traumatic cortical irregularities, millimetric bone fragments in the left scapula.

---

### Example 8: Cardiomegaly and Atherosclerosis
**Report:**
Findings: Calibration of the aortic arch is at the maximal physiological limit. Calibration of other major vascular structures in the mediastinal is natural. CTO is within the normal range. No lymph node was detected in the mediastinum and in both hilar levels in pathological size and configuration. Thoracic esophagus calibration was normal and no significant tumoral wall thickening was detected. Mild hiatal hernia is observed. When examined in the lung parenchyma window; Both hemithorax are symmetrical. The calibration of the trachea and main bronchi is normal and their lumens are clear. Density reduction compatible with mild emphysema is observed. On the right, a nonspecific nodular density of 5x3 mm is observed superposed on the minor fissure. In the left lung, there is linear density consistent with band atelectasis-sequelae changes in the inferior lingular segment. Nonspecific density increases are observed in the lower lobes of both lungs, more prominently in the dorsal areas and adjacent to the interlobar fissure on the right. Dependent was evaluated as consistent with vascular density. Bilateral pleural effusion pneumothorax was not detected. There are bilateral irregular density increases in the perinephric areas. A decrease in density is observed in the liver, which is compatible with steatosis. Although the spleen is ventral and caudally lobulated in the contour, nodular appearance is observed, but there may be a structural variational appearance. No significant density difference was detected at this level. A clear evaluation cannot be made in the non-contrast examination. Bilateral adrenal glands were normal and no space-occupying lesion was detected. Degenerative changes are observed in the bone structures in the study area. There is narrowing of the spinal canal at the dorso- lumbar level. Impression: Density increases, mild sequelae changes and mild emphysema appearance, which are primarily evaluated as compatible with the dependent vascular density observed in the dorsal subpleural area in both lower lobes.  Hepatostetaosis.  Hiatal hernia. Intense degenerative changes in bone structure.

---

### Example 9: Consolidation and Interlobular Septal Thickening
**Report:**
Findings: Mediastinal structures were evaluated as suboptimal since the examination was unenhanced. As far as can be seen; Trachea and lumen of both main bronchi are open. No occlusive pathology was detected in the trachea and lumen of both main bronchi. An image of a catheter extending superiorly to the vena cava was observed. Heart contour size is natural. Pericardial thickening-effusion was not detected. Calcified atherosclerotic changes were observed in the wall of the thoracic aorta. Thoracic esophagus calibration was normal, and no significant pathological wall thickening was detected in the non-contrast examination. A few calcified lymph nodes with a short axis smaller than 1 cm were observed in the left hilar region. In addition, lymph nodes measuring 1 cm in the short axis of the largest were observed in the upper-lower paratracheal prevascular aorticopulmonary region. When examined in the lung parenchyma window; Interlobular septal thickenings and alveolar consolidation areas were observed in the upper lobe of the left lung. The appearance may be secondary to cardiac pathology. Infectious process can be considered in the separate diagnosis. Clinical laboratory correlation and post-treatment control are recommended. There are patches of ground glass density increases in both lungs. A few parenchymal nodules, the largest of which was 8 mm in diameter, were observed in the right lung. Between the bilateral pleural leaves, pleural effusion with a thickness of 24 mm on the right and 37 mm on the left, and atelectatic changes in the adjacent lung parenchyma were observed. A few dense 6 mm diameter calculi were observed in the gallbladder lumen in the upper abdominal sections that entered the study area. Bilateral adrenal gland calibration was normal and no space-occupying lesion was detected. Degenerative changes were observed in bone structures. Impression:  Patchy ground-glass density increases in both lungs, parenchymal nodules in the right lung.  Diffuse septal thickenings and areas of alveolar consolidation in the upper lobe of the left lung (secondary to cardiac pathology? Infectious process?). Clinical-laboratory correlation and post-treatment control are recommended.  Bilateral pleural effusion, atelectatic changes.  Cholelithiasis.  Degenerative changes in bone structure.

---

### Example 10: Mosaic Attenuation Pattern
**Report:**
Findings: Trachea, both main bronchi are open. Mediastinal main vascular structures, heart contour, size are normal. Thoracic aorta diameter is normal. Pericardial effusion-thickening was not observed. Thoracic esophageal calibration was normal and no significant tumoral wall thickening was detected. Large hiatal hernia is observed. There are small lymph nodes with a short axis measuring up to 5 mm in the mediastinum, especially at the aorticopulmonary window and at the level of the trachea carina. When examined in the lung parenchyma window; There is a mosaic attenuation pattern of thickenings in the interlobular septa in both lungs. Slightly patchy ground glass densities are observed in the apical level of the upper lobe of the right lung and the lateral part of the middle lobe of the right lung. A few millimetric nonspecific nodules are observed in both lungs. The largest measured 4 mm in the upper lobe of the right lung in series 2 images 224. No nodular or infiltrative lesion was detected in both lung parenchyma. Upper abdominal organs included in the sections are normal. No space-occupying lesion was detected in the liver that entered the cross-sectional area. Bilateral adrenal glands were normal and no space-occupying lesion was detected. There is a diffuse density decrease in the bone structures in the examination area, and there are degenerative height losses in the vertebral corpuscles. Secondary to the fractures, left-facing scoliosis is observed. Impression: Thickening of interlobular septa in both lungs, mosaic attenuation pattern, and slightly patchy ground-glass densities in the right lung. Findings were primarily evaluated in favor of pulmonary edema. Clinical laboratory correlation is recommended for the onset of an infectious process.  Atherosclerosis . Osteoparotic appearance in bone structures, degenerative in vertebral corpuscles Fractures . Left-facing scoliosis . Small oval lymph nodes in the mediastinum

---
"""

    def load_excel_data(self):
        """Load the CT reports from the Excel file."""
        try:
            df = pd.read_excel(self.excel_file)
            print(f"Loaded Excel file with {len(df)} records")
            print(f"Excel columns: {df.columns.tolist()}")
            return df
        except Exception as e:
            print(f"Failed to read the Excel file: {e}")
            return None
    
    def load_disease_predictions(self):
        """Load the disease-prediction CSV."""
        try:
            df = pd.read_csv(self.disease_csv)
            print(f"Loaded disease predictions with {len(df)} records")
            # Index by VolumeName.
            df.set_index('VolumeName', inplace=True)
            return df
        except Exception as e:
            print(f"Failed to read the disease-prediction CSV: {e}")
            return None
    
    def format_disease_predictions(self, disease_row):
        """Format the disease predictions as a string."""
        disease_names = [
            "Medical material", "Arterial wall calcification", "Cardiomegaly", 
            "Pericardial effusion", "Coronary artery wall calcification", "Hiatal hernia",
            "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule", "Lung opacity",
            "Pulmonary fibrotic sequela", "Pleural effusion", "Mosaic attenuation pattern",
            "Peribronchial thickening", "Consolidation", "Bronchiectasis", 
            "Interlobular septal thickening"
        ]
        
        predictions = []
        for disease in disease_names:
            value = disease_row.get(disease, 0)
            predictions.append(f"{disease}={int(value)}")
        
        return "; ".join(predictions)
    
    def call_gpt(self, input_text, max_retries=8):
        """Translate one report through the chat API."""
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model="gpt-4.1",
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": input_text}
                    ],
                    temperature=0.2,
                    max_tokens=2048
                )
                
                return completion.choices[0].message.content
                
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Request failed, retrying in {wait_time}s... error: {e}")
                    time.sleep(wait_time)
                else:
                    print(f"Giving up after {max_retries} attempts. Error: {e}")
                    return None
    
    def load_checkpoint(self):
        """Load the checkpoint file that tracks progress."""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as file:
                    return json.load(file)
            except Exception as e:
                print(f"Failed to read the checkpoint file: {e}")
        return {"processed_ids": []}
    
    def save_checkpoint(self, processed_ids):
        """Write the checkpoint file."""
        checkpoint = {"processed_ids": list(processed_ids)}
        with open(self.checkpoint_file, 'w', encoding='utf-8') as file:
            json.dump(checkpoint, file, ensure_ascii=False, indent=4)
    
    def load_results(self):
        """Load the translations produced so far."""
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r', encoding='utf-8') as file:
                    return json.load(file)
            except Exception as e:
                print(f"Failed to read the output file: {e}")
        return []
    
    def save_results(self, results):
        """Write the translations to disk."""
        with open(self.output_file, 'w', encoding='utf-8') as file:
            json.dump(results, file, ensure_ascii=False, indent=4)
    
    def preview_data(self, excel_df, disease_df, num_samples=3):
        """Preview the inputs before translating (no patient identifiers are shown)."""
        print("\n" + "="*80)
        print("Data preview - first {} samples".format(num_samples))
        print("="*80)
        print("Note: names, patient IDs and other identifiers are hidden for privacy")
        print("="*80)
        
        for idx in range(min(num_samples, len(excel_df))):
            row = excel_df.iloc[idx]
            case_id = str(row.get('检查ID', f'case_{idx}'))
            volume_name = f"CT_{case_id}"
            
            print(f"\nSample {idx + 1}:")
            print(f"  Exam ID: {case_id}")
            print(f"  Volume name: {volume_name}")
            print(f"  Exam prompt (first 150 chars): {str(row.get('检查提示', ''))[:150]}...")
            
            if volume_name in disease_df.index:
                disease_predictions = self.format_disease_predictions(disease_df.loc[volume_name])
                print(f"  [OK] Disease predictions: {disease_predictions[:150]}...")
            else:
                print(f"  [MISSING] Disease predictions: not found (check whether VolumeName is {volume_name})")
        
        print("\n" + "="*80)
        print("Data format check:")
        print("1. Whether the exam ID is displayed correctly")
        print("2. Whether the volume name follows the CT_<exam ID> format")
        print("3. Whether the disease predictions were found")
        print("="*80)
        response = input("\nProceed with this data? Type 'yes' to continue: ")
        return response.lower() == 'yes'
    
    def process_data(self):
        """Load the inputs and translate the reports."""
        # Load the inputs.
        excel_df = self.load_excel_data()
        disease_df = self.load_disease_predictions()
        
        if excel_df is None or disease_df is None:
            print("Failed to load the inputs, exiting")
            return
        
        # Preview them.
        if not self.preview_data(excel_df, disease_df, num_samples=3):
            print("Cancelled by the user")
            return
        
        # Restore progress from the checkpoint and the existing output.
        checkpoint = self.load_checkpoint()
        processed_ids = set(checkpoint["processed_ids"])
        results = self.load_results()
        
        print(f"Loaded {len(processed_ids)} processed sample IDs and {len(results)} existing translations")
        
        # Counters for the samples and the save batches.
        total_processed = len(processed_ids)
        batch_counter = 0
        report_counter = len(results)
        
        # Walk over every sample.
        for idx, row in excel_df.iterrows():
            # Exam ID and report text (using the real Excel column names).
            case_id = str(row.get('检查ID', f'case_{idx}'))
            chinese_report_full = str(row.get('检查提示', ''))
            
            # Skip empty reports.
            if pd.isna(chinese_report_full) or chinese_report_full.strip() in ['', 'nan', 'None']:
                print(f"Warning: empty exam prompt for {case_id}, skipping")
                continue
            
            # The exam prompt already is the full report.
            # No further splitting is needed.
            full_chinese_report = chinese_report_full.strip()
            
            # Skip anything already processed.
            if case_id in processed_ids:
                print(f"{case_id} already processed, skipping")
                continue
            
            print(f"Processing {case_id}...")
            
            # Volume name, assumed to follow the CT_<exam ID> format.
            volume_name = f"CT_{case_id}"
            
            # Disease predictions.
            if volume_name in disease_df.index:
                disease_predictions = self.format_disease_predictions(disease_df.loc[volume_name])
            else:
                print(f"Warning: no disease predictions for {volume_name}, falling back to all zeros")
                disease_predictions = "; ".join([f"{d}=0" for d in [
                    "Medical material", "Arterial wall calcification", "Cardiomegaly",
                    "Pericardial effusion", "Coronary artery wall calcification", "Hiatal hernia",
                    "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule", "Lung opacity",
                    "Pulmonary fibrotic sequela", "Pleural effusion", "Mosaic attenuation pattern",
                    "Peribronchial thickening", "Consolidation", "Bronchiectasis",
                    "Interlobular septal thickening"
                ]])
            
            # Build the translation request.
            input_text = f"{self.example_reports}\n\n## Chinese Report to Translate\n\n{full_chinese_report}\n\n"
            input_text += "Please translate the above Chinese CT report into English, following the exact style, structure, and terminology of the example reports provided. "
            input_text += "IMPORTANT: The Chinese report may not have explicit 'Findings:' and 'Impression:' sections. You need to:\n"
            input_text += "1. Analyze the content and organize it into proper 'Findings:' and 'Impression:' sections\n"
            input_text += "2. The 'Findings:' section should describe detailed anatomical observations\n"
            input_text += "3. The 'Impression:' section should be a concise summary of key findings\n"
            input_text += "4. Use ONLY 'Findings:' and 'Impression:' sections in your output\n"
            input_text += "5. Maintain professional medical language and systematic anatomical descriptions matching the examples."
            
            # Translate.
            english_report = self.call_gpt(input_text)
            
            if english_report is None:
                print(f"Translation failed, skipping {case_id}")
                continue
            
            # Pick a question template at random.
            question_template = random.choice(self.question_templates)
            
            # Build the question.
            question = f"<image>\n{question_template} Known frontend model predictions (disease-wise): {disease_predictions}.<report_generation>"
            
            # Build the record.
            result = {
                "id": f"report_generation_{report_counter}",
                "image": f"{volume_name}.nii.gz",
                "conversations": [
                    {
                        "type": "report_generation",
                        "from": "human",
                        "value": question
                    },
                    {
                        "from": "gpt",
                        "value": english_report
                    }
                ]
            }
            
            results.append(result)
            processed_ids.add(case_id)
            
            # Update the counters.
            total_processed += 1
            batch_counter += 1
            report_counter += 1
            
            # Flush every batch_size samples.
            if batch_counter >= self.batch_size:
                self.save_results(results)
                self.save_checkpoint(processed_ids)
                print(f"Batch saved, {total_processed} samples processed so far")
                batch_counter = 0
            
            # Stay well below the API rate limit.
            time.sleep(1)
        
        # Flush whatever is left.
        if batch_counter > 0:
            self.save_results(results)
            self.save_checkpoint(processed_ids)
        
        print(f"Done: {total_processed} samples processed, results written to {self.output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Translate the MianYang Chinese CT reports into English via an "
                    "OpenAI-compatible chat API.")
    parser.add_argument("--excel-file", required=True,
                        help="Spreadsheet of the raw Chinese reports")
    parser.add_argument("--disease-csv", required=True,
                        help="disease_predictions.csv for the same cohort")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for the translated reports and checkpoints")
    parser.add_argument("--base-url", default="https://www.dmxapi.cn/v1/",
                        help="OpenAI-compatible API endpoint")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Save results every N samples")
    args = parser.parse_args()

    # The API key is read from the environment: never hard-code a credential in a
    # file that gets committed.
    api_key = os.environ.get("TRANSLATE_API_KEY")
    if not api_key:
        raise SystemExit(
            "Set TRANSLATE_API_KEY to your API key before running, e.g.\n"
            "    export TRANSLATE_API_KEY=...")

    os.makedirs(args.output_dir, exist_ok=True)

    translator = CTReportTranslator(args.excel_file, args.disease_csv, args.output_dir,
                                    api_key, args.base_url, batch_size=args.batch_size)
    translator.process_data()

if __name__ == "__main__":
    main()