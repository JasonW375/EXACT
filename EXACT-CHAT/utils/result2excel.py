
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarise evaluation results as a table.

Collects the metrics.json file written by each evaluation run and
writes a single CSV / Excel summary.
"""

import json
import pandas as pd
from pathlib import Path
import re
from typing import Dict, List, Any

# Paths.
RESULTS_BASE_DIR = Path("/path/to/CT-CHAT2/llava/train/output")
OUTPUT_CSV = RESULTS_BASE_DIR / "evaluation_summary.csv"
OUTPUT_EXCEL = RESULTS_BASE_DIR / "evaluation_summary.xlsx"

def extract_experiment_name(dir_path: Path) -> str:
    """Derive a short experiment name from the directory name."""
    name = dir_path.name
    
    # Strip the common prefix.
    name = re.sub(r'^predictions_', '', name)
    
    # Pull the key hyper-parameters out of the name.
    # Checkpoint step.
    checkpoint_match = re.search(r'checkpoint(\d+)', name)
    checkpoint_num = checkpoint_match.group(1) if checkpoint_match else "unknown"
    
    # Sampling temperature.
    temp_match = re.search(r'temp([\d\.]+)', name)
    temperature = temp_match.group(1) if temp_match else "unknown"
    
    # Maximum number of generated tokens.
    token_match = re.search(r'tokens(\d+)', name)
    tokens = token_match.group(1) if token_match else "unknown"
    
    # Timestamp, if the name carries one.
    datetime_match = re.search(r'(\d{8}_\d{6})', name)
    datetime_str = datetime_match.group(1) if datetime_match else ""
    
    # Assemble the shortened name.
    simplified_name = f"ckpt{checkpoint_num}_temp{temperature}_tok{tokens}"
    if datetime_str:
        simplified_name += f"_{datetime_str}"
    
    return simplified_name

def load_metrics(metrics_file: Path) -> Dict[str, Any]:
    """Load a metrics.json file."""
    try:
        with open(metrics_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load {metrics_file}: {e}")
        return {}

def extract_generation_metrics(data: Dict) -> Dict[str, float]:
    """Extract the text-generation metrics."""
    gen_data = data.get('generation', {})
    return {
        'BLEU_1': gen_data.get('BLEU_1', 0.0),
        'BLEU_2': gen_data.get('BLEU_2', 0.0),
        'BLEU_3': gen_data.get('BLEU_3', 0.0),
        'BLEU_4': gen_data.get('BLEU_4', 0.0),
        'BLEU_mean': gen_data.get('BLEU_mean', 0.0),
        'ROUGE_L': gen_data.get('ROUGE_L', 0.0),
        'CIDEr': gen_data.get('CIDEr', 0.0),
        'METEOR': gen_data.get('METEOR', 0.0)
    }

def extract_classification_macro_metrics(data: Dict) -> Dict[str, float]:
    """Extract the macro-averaged classification metrics."""
    cls_data = data.get('classification', {})
    macro_data = cls_data.get('macro', {})
    return {
        'cls_macro_precision': macro_data.get('precision', 0.0),
        'cls_macro_recall': macro_data.get('recall', 0.0),
        'cls_macro_f1': macro_data.get('f1', 0.0),
        'cls_macro_accuracy': macro_data.get('accuracy', 0.0)
    }

def extract_crg_metrics(data: Dict) -> Dict[str, float]:
    """Extract the CRG metrics."""
    crg_data = data.get('crg', {})
    return {
        'CRG_score': crg_data.get('CRG', 0.0),
        'CRG_TP': crg_data.get('TP', 0),
        'CRG_FN': crg_data.get('FN', 0),
        'CRG_FP': crg_data.get('FP', 0),
        'CRG_X': crg_data.get('X', 0),
        'CRG_A': crg_data.get('A', 0),
        'CRG_r': crg_data.get('r', 0.0),
        'CRG_U': crg_data.get('U', 0.0),
        'CRG_score_s': crg_data.get('score_s', 0.0)
    }

def extract_per_pathology_metrics(data: Dict) -> Dict[str, float]:
    """Extract the per-pathology metrics."""
    cls_data = data.get('classification', {})
    per_path_data = cls_data.get('per_pathology', [])
    
    metrics = {}
    
    for pathology in per_path_data:
        name = pathology.get('name', 'unknown')
        # Normalise the pathology name so it can serve as a column name.
        clean_name = name.replace(' ', '_').replace('-', '_')
        
        metrics.update({
            f'{clean_name}_precision': pathology.get('precision', 0.0),
            f'{clean_name}_recall': pathology.get('recall', 0.0),
            f'{clean_name}_f1': pathology.get('f1', 0.0),
            f'{clean_name}_accuracy': pathology.get('accuracy', 0.0)
        })
    
    return metrics

def find_all_metrics_files(base_dir: Path) -> List[Path]:
    """Find every metrics.json below the base directory."""
    metrics_files = []
    
    for subdir in base_dir.iterdir():
        if subdir.is_dir():
            metrics_file = subdir / "metrics.json"
            if metrics_file.exists():
                metrics_files.append(metrics_file)
    
    return sorted(metrics_files)

def create_summary_table(metrics_files: List[Path]) -> pd.DataFrame:
    """Build the summary table."""
    all_rows = []
    
    print(f"Found {len(metrics_files)} metrics.json files")
    
    for metrics_file in metrics_files:
        print(f"Processing: {metrics_file.parent.name}")
        
        # Experiment name.
        exp_name = extract_experiment_name(metrics_file.parent)
        
        # Metrics.
        data = load_metrics(metrics_file)
        if not data:
            continue
        
        # One row per experiment.
        row_data = {'experiment': exp_name}
        
        # Generation metrics.
        row_data.update(extract_generation_metrics(data))
        
        # Macro-averaged classification metrics.
        row_data.update(extract_classification_macro_metrics(data))
        
        # CRG metrics.
        row_data.update(extract_crg_metrics(data))
        
        # Per-pathology metrics.
        row_data.update(extract_per_pathology_metrics(data))
        
        all_rows.append(row_data)
    
    if not all_rows:
        print("No valid data found")
        return pd.DataFrame()
    
    # Assemble the DataFrame.
    df = pd.DataFrame(all_rows)
    
    # Column order: experiment name, aggregate metrics, then per-pathology ones.
    cols_order = ['experiment']
    
    # Generation metrics.
    gen_cols = ['BLEU_1', 'BLEU_2', 'BLEU_3', 'BLEU_4', 'BLEU_mean', 'ROUGE_L', 'CIDEr', 'METEOR']
    cols_order.extend(gen_cols)
    
    # Macro-averaged classification metrics.
    cls_macro_cols = ['cls_macro_precision', 'cls_macro_recall', 'cls_macro_f1', 'cls_macro_accuracy']
    cols_order.extend(cls_macro_cols)
    
    # CRG metrics.
    crg_cols = ['CRG_score', 'CRG_TP', 'CRG_FN', 'CRG_FP', 'CRG_X', 'CRG_A', 'CRG_r', 'CRG_U', 'CRG_score_s']
    cols_order.extend(crg_cols)
    
    # Everything else, i.e. the per-pathology columns.
    remaining_cols = [col for col in df.columns if col not in cols_order]
    cols_order.extend(sorted(remaining_cols))
    
    # Apply the new order.
    df = df.reindex(columns=cols_order)
    
    return df

def format_dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Round the metric columns for display."""
    if df.empty:
        return df
    
    # Work on a copy so the caller's DataFrame is left untouched.
    display_df = df.copy()
    
    # Round the metric columns to four decimal places.
    numeric_cols = display_df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if col.startswith(('BLEU', 'ROUGE', 'CIDEr', 'METEOR', 'cls_', 'CRG_score', 'CRG_r')):
            display_df[col] = display_df[col].round(4)
    
    return display_df

def save_results(df: pd.DataFrame):
    """Write the table to CSV and Excel."""
    if df.empty:
        print("Nothing to save")
        return
    
    try:
        # CSV.
        df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
        print(f"CSV written to: {OUTPUT_CSV}")
        
        # Excel, with the column widths adjusted.
        with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Evaluation_Summary', index=False)
            
            # Grab the worksheet so its layout can be tweaked.
            worksheet = writer.sheets['Evaluation_Summary']
            
            # Fit every column to its widest cell.
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)  # Cap at 50 characters.
                worksheet.column_dimensions[column_letter].width = adjusted_width
        
        print(f"Excel written to: {OUTPUT_EXCEL}")
        
    except Exception as e:
        print(f"Error while saving: {e}")

def print_summary_statistics(df: pd.DataFrame):
    """Print summary statistics over the collected experiments."""
    if df.empty:
        return
    
    print(f"\nSummary:")
    print(f"   Experiments: {len(df)}")
    print(f"   Metrics: {len(df.columns) - 1}")  # Excluding the experiment column.
    
    # Statistics over a few headline metrics.
    key_metrics = ['BLEU_mean', 'ROUGE_L', 'cls_macro_f1', 'CRG_score']
    existing_metrics = [m for m in key_metrics if m in df.columns]
    
    if existing_metrics:
        print(f"\nKey metrics:")
        stats_df = df[existing_metrics].describe()
        print(stats_df.round(4))
    
    # Best-performing experiments.
    if 'BLEU_mean' in df.columns:
        best_bleu_idx = df['BLEU_mean'].idxmax()
        best_exp = df.loc[best_bleu_idx, 'experiment']
        best_bleu = df.loc[best_bleu_idx, 'BLEU_mean']
        print(f"\nBest BLEU_mean: {best_exp} ({best_bleu:.4f})")
    
    if 'CRG_score' in df.columns:
        best_crg_idx = df['CRG_score'].idxmax()
        best_exp = df.loc[best_crg_idx, 'experiment']
        best_crg = df.loc[best_crg_idx, 'CRG_score']
        print(f"Best CRG_score: {best_exp} ({best_crg:.4f})")

def main():
    """Entry point."""
    print("Evaluation result summariser")
    print(f"Scanning directory: {RESULTS_BASE_DIR}")
    
    # The base directory has to exist.
    if not RESULTS_BASE_DIR.exists():
        print(f"Directory not found: {RESULTS_BASE_DIR}")
        return
    
    # Collect the metrics.json files.
    metrics_files = find_all_metrics_files(RESULTS_BASE_DIR)
    
    if not metrics_files:
        print("No metrics.json files found")
        return
    
    # Build the summary table.
    df = create_summary_table(metrics_files)
    
    if df.empty:
        print("Could not build the table")
        return
    
    # Round for display.
    display_df = format_dataframe_for_display(df)
    
    # Preview the first few rows.
    print(f"\nEvaluation results (first 5 rows):")
    print("=" * 100)
    
    # The preview shows only the headline columns.
    preview_cols = ['experiment'] + [col for col in df.columns if col in [
        'BLEU_mean', 'ROUGE_L', 'CIDEr', 'METEOR', 
        'cls_macro_precision', 'cls_macro_recall', 'cls_macro_f1', 
        'CRG_score'
    ]]
    
    if len(display_df) > 5:
        print(display_df[preview_cols].head().to_string(index=False))
    else:
        print(display_df[preview_cols].to_string(index=False))
    
    # Write the output files.
    save_results(df)
    
    # Report the statistics.
    print_summary_statistics(df)
    
    print(f"\nDone.")
    print(f"Full results:")
    print(f"   CSV: {OUTPUT_CSV}")
    print(f"   Excel: {OUTPUT_EXCEL}")

if __name__ == "__main__":
    main()
