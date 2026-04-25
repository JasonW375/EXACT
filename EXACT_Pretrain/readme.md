# EXACT: EXplainable Abnormality-aware ChesT CT Foundation Model
## 📄 About
**EXACT** is an advanced 3D Chest CT foundation model designed for explainable abnormality perception. It extends our previous work, **Chest-OMDL** (published in MIDL 2025), by introducing the **Y-Mamba** architecture and a **Multi-Instance Learning (MIL)** framework. Unlike traditional CLIP-based models that only provide global semantic alignment, EXACT generates voxel-level **Anomaly-aware Maps (AAmap)**, enabling both robust multi-disease diagnosis and fine-grained localization.
---
## 🛠️ Environment Setup
We recommend using Anaconda to manage the environment. Please follow the steps below strictly to avoid installation issues with `mamba-ssm`.
### 1. Create and Activate Environment
```bash
conda create -n exact python=3.10
conda activate exact
2. Install PyTorch (CUDA 12.1)
Ensure your CUDA version is compatible.

<BASH>
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
3. Install Basic Dependencies
<BASH>
pip install -r requirements.txt
4. Install Mamba-SSM (Crucial Step)
mamba-ssm requires compilation. We use --no-build-isolation to ensure it uses the installed PyTorch and CUDA environment.

<BASH>
pip install --no-build-isolation mamba-ssm==2.2.4
Note: If you encounter any ModuleNotFoundError during runtime, please install the missing packages manually using pip (e.g., pip install package_name).

📂 Data Preprocessing
1. Training & In-Distribution Testing Data
For the standard training and testing dataset, use the following script:

<BASH>
python EXACT/data_preprocessed/data_preprocessed.py
2. Out-of-Distribution (OOD) / External Data
For external datasets or data with different distributions, the preprocessing consists of two steps:

Step 1: Basic Preprocessing

<BASH>
python EXACT/data_preprocessed/new_data_preprocessed.py
Step 2: Orientation Adjustment
To align the data orientation with the training set, run the flip script.
Note: This is applicable if the original data orientation matches EXACT/example_data/val_data.nii.gz.

<BASH>
python EXACT/data_preprocessed/flip_data.py
After these two steps, the data is ready for inference.

🚀 Usage (Multi-disease Diagnosis)
EXACT supports two diagnostic modes for multi-disease classification: Zero-shot inference based on the pre-trained foundation model, and Supervised Fine-tuning for higher diagnostic precision.

Mode 1: Weakly Supervised Pre-training (Zero-shot Foundation)
To perform the initial pre-training using image-level disease labels (automatically extracted from radiology reports) and coarse organ segmentation masks as weak supervision:

<BASH>
python train.py
After pre-training, the model is capable of Zero-shot Diagnosis by applying anatomically-constrained Top-k pooling directly on the generated 18-channel AAmaps, without any task-specific training.

Mode 2: Supervised Fine-tuning (AAmap-based Classifier)
To further boost diagnostic performance, we provide a lightweight supervised framework. The pre-trained EXACT backbone is frozen, and only a lightweight classifier is trained on top of the generated AAmaps as input features:

<BASH>
python train_supervised.py
Testing / Evaluation
To evaluate either the Zero-shot or Fine-tuned model on the test set:

<BASH>
python test.py
📝 Citation
EXACT is an extension of the following published work. If you find this project useful, please consider citing:

Base Paper (MIDL 2025):

<BIBTEX>
@inproceedings{bai2025chestomdl,
  title={Chest-{OMDL}: Organ-specific Multidisease Detection and Localization in Chest Computed Tomography using Weakly Supervised Deep Learning from Free-text Radiology Report},
  author={Xuguang Bai and Mingxuan Liu and Yifei Chen and Hongjia Yang and Qiyuan Tian},
  booktitle={Medical Imaging with Deep Learning},
  year={2025},
  url={https://openreview.net/forum?id=ns6nq592HX}
}
EXACT (Coming Soon):
(Paper information will be updated upon publication.)