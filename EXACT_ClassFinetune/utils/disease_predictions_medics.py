import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support
import os


def choose_best_threshold_lefttop(y_true: np.ndarray, y_prob: np.ndarray, fallback: float = 0.5):
    """
    选取ROC曲线中距离(0,1)最近的点作为最佳阈值；并列时先最大化TPR再最小化FPR。
    
    Args:
        y_true: 真实标签
        y_prob: 预测概率
        fallback: 备用阈值
    
    Returns:
        (threshold, auc): 最佳阈值和AUC值
    """
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        if fpr.size == 0 or tpr.size == 0 or thresholds.size == 0:
            return fallback, np.nan
        dist = np.sqrt(fpr**2 + (1.0 - tpr)**2)
        best_dist = np.min(dist)
        idxs = np.where(np.isclose(dist, best_dist))[0]
        if idxs.size == 0:
            return fallback, roc_auc_score(y_true, y_prob)
        if idxs.size > 1:
            tpr_c = tpr[idxs]
            max_tpr = np.max(tpr_c)
            idxs = idxs[tpr_c >= max_tpr - 1e-12]
            if idxs.size > 1:
                fpr_c = fpr[idxs]
                min_fpr = np.min(fpr_c)
                idxs = idxs[fpr_c <= min_fpr + 1e-12]
        thr = float(thresholds[int(idxs[0])])
        auc = roc_auc_score(y_true, y_prob)
        return thr, float(auc)
    except Exception:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = float('nan')
        return fallback, float(auc)


def load_prediction_data(csv_path: str):
    """
    加载预测数据CSV文件
    
    Args:
        csv_path: CSV文件路径
    
    Returns:
        df: DataFrame对象
    """
    # 自动检测分隔符
    with open(csv_path, 'r') as f:
        first_line = f.readline()
    sep = '\t' if '\t' in first_line else ','
    
    df = pd.read_csv(csv_path, sep=sep)
    df = df.replace('', np.nan).dropna()
    
    return df


def extract_pred_gt_columns(df: pd.DataFrame):
    """
    提取预测列和真实标签列
    
    Args:
        df: 输入DataFrame
    
    Returns:
        (pred_cols, gt_cols): 预测列名列表和GT列名列表
    """
    pred_cols = [col for col in df.columns if col.startswith('Pred_')]
    gt_cols = [col for col in df.columns if col.startswith('GT_')]
    
    return pred_cols, gt_cols


def calculate_optimal_thresholds(df: pd.DataFrame, pred_cols: list, gt_cols: list):
    """
    为每个疾病类别计算最优阈值
    
    Args:
        df: 数据DataFrame
        pred_cols: 预测列名列表
        gt_cols: GT列名列表
    
    Returns:
        thresholds: 字典，key为预测列名，value为最优阈值
    """
    thresholds = {}
    
    for pred_col, gt_col in zip(pred_cols, gt_cols):
        y_prob = df[pred_col].astype(float).values
        y_true = df[gt_col].astype(int).values
        thr, _ = choose_best_threshold_lefttop(y_true, y_prob)
        thresholds[pred_col] = thr
    
    return thresholds


def generate_binary_predictions(df: pd.DataFrame, pred_cols: list, thresholds: dict):
    """
    根据最优阈值生成二分类预测结果
    
    Args:
        df: 数据DataFrame
        pred_cols: 预测列名列表
        thresholds: 阈值字典
    
    Returns:
        results_df: 包含样本名和各疾病预测结果的DataFrame
    """
    results = {"VolumeName": df["VolumeName"].values}
    
    for pred_col in pred_cols:
        y_prob = df[pred_col].astype(float).values
        thr = thresholds[pred_col]
        pred_bin = (y_prob >= thr).astype(int)
        
        disease_name = pred_col.replace('Pred_', '')
        results[disease_name] = pred_bin
    
    results_df = pd.DataFrame(results)
    return results_df


def calculate_metrics(df: pd.DataFrame, pred_cols: list, gt_cols: list, thresholds: dict):
    """
    计算每个疾病的评价指标
    
    Args:
        df: 数据DataFrame
        pred_cols: 预测列名列表
        gt_cols: GT列名列表
        thresholds: 阈值字典
    
    Returns:
        metrics_df: 包含各疾病评价指标的DataFrame
    """
    metrics = {}
    
    for pred_col, gt_col in zip(pred_cols, gt_cols):
        y_prob = df[pred_col].astype(float).values
        y_true = df[gt_col].astype(int).values
        thr = thresholds[pred_col]
        
        # 生成二分类预测
        pred_bin = (y_prob >= thr).astype(int)
        
        # 计算AUC
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = np.nan
        
        # 计算Precision, Recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, pred_bin, average='binary', zero_division=0
        )
        
        disease_name = pred_col.replace('Pred_', '')
        metrics[disease_name] = {
            "Threshold": thr,
            "AUC": auc,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        }
    
    metrics_df = pd.DataFrame(metrics).T
    return metrics_df


def process_disease_prediction(csv_path: str, output_dir: str = None):
    """
    主函数：处理疾病预测任务
    
    Args:
        csv_path: 输入CSV文件路径
        output_dir: 输出文件夹路径，默认为输入文件所在文件夹
    
    Returns:
        (results_df, metrics_df): 预测结果DataFrame和评价指标DataFrame
    """
    # 如果未指定输出目录，使用输入文件所在目录
    if output_dir is None:
        output_dir = os.path.dirname(csv_path)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"正在加载数据: {csv_path}")
    df = load_prediction_data(csv_path)
    print(f"数据加载完成，共 {len(df)} 个样本")
    
    print("\n提取预测列和GT列...")
    pred_cols, gt_cols = extract_pred_gt_columns(df)
    print(f"检测到 {len(pred_cols)} 个疾病类别")
    
    print("\n计算各疾病最优阈值...")
    thresholds = calculate_optimal_thresholds(df, pred_cols, gt_cols)
    
    print("\n生成二分类预测结果...")
    results_df = generate_binary_predictions(df, pred_cols, thresholds)
    
    print("\n计算评价指标...")
    metrics_df = calculate_metrics(df, pred_cols, gt_cols, thresholds)
    
    # 保存结果
    pred_save_path = os.path.join(output_dir, 'disease_predictions.csv')
    metrics_save_path = os.path.join(output_dir, 'disease_metrics.csv')
    
    results_df.to_csv(pred_save_path, index=False)
    metrics_df.to_csv(metrics_save_path)
    
    print(f"\n✓ 预测结果已保存到: {pred_save_path}")
    print(f"✓ 评价指标已保存到: {metrics_save_path}")
    
    print("\n=== 评价指标汇总 ===")
    print(metrics_df.round(4))
    
    return results_df, metrics_df


if __name__ == "__main__":
    # 使用示例
    csv_path = '/FM_data/bxg/CT_Report/CT_Report16_classification/heatmap_ft/mianyang/pred.csv'
    output_dir = '/FM_data/bxg/CT_Report/CT_Report16_classification/heatmap_ft/mianyang/'
    
    results_df, metrics_df = process_disease_prediction(csv_path, output_dir)