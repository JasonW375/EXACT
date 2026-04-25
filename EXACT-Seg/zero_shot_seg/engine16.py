import torch
import numpy as np
import os
import csv
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path

def test_one_epoch(test_loader, model, segmentation_criterion, abnormal_criterion, epoch, logger, config, writer, device, best_thresholds, save_heatmap=True):  
    model.eval()  
    loss_list = []  
    abnormal_loss_list = [[] for _ in range(18)]  # 18种疾病的损失记录
    sample_idx = 0  
    np.random.seed(42)  
    
    dataset_size = len(test_loader.dataset)  
    selected_indices = set(np.random.choice(dataset_size, min(40, dataset_size), replace=False))  
    current_dir = config.work_dir  

    if save_heatmap:  
        save_dir = os.path.join(config.work_dir, "test_results")  
        os.makedirs(save_dir, exist_ok=True)  
        heatmap_data = []  

    logger.info(f"Test epoch {epoch}: Abnormal weight = {config.abnormal_loss_weight}")  

    test_loader = tqdm(test_loader, desc=f"Epoch {epoch} Test", leave=True, dynamic_ncols=True)  

    # 初始化评估指标数组（使用18种疾病）  
    processed_predictions = [[] for _ in range(18)]  
    all_targets = [[] for _ in range(18)]  
    tp_sum = np.zeros(18, dtype=int)  
    tn_sum = np.zeros(18, dtype=int)  
    fp_sum = np.zeros(18, dtype=int)  
    fn_sum = np.zeros(18, dtype=int)  

    # 保存样本信息的列表
    all_sample_names = []
    all_sample_predictions = []
    all_sample_targets = []

    # 定义18种疾病名称
    disease_names_18 = [  
        "Medical material",                      # 0
        "Arterial wall calcification",           # 1
        "Cardiomegaly",                          # 2
        "Pericardial effusion",                  # 3
        "Coronary artery wall calcification",    # 4
        "Hiatal hernia",                         # 5
        "Lymphadenopathy",                       # 6
        "Emphysema",                             # 7
        "Atelectasis",                           # 8
        "Lung nodule",                           # 9
        "Lung opacity",                          # 10
        "Pulmonary fibrotic sequela",            # 11
        "Pleural effusion",                      # 12
        "Mosaic attenuation pattern",            # 13
        "Peribronchial thickening",              # 14
        "Consolidation",                         # 15
        "Bronchiectasis",                        # 16
        "Interlobular septal thickening"         # 17
    ]  

    # 器官名称（包含global通道）
    organ_names = ["lung", "trachea and bronchie", "pleura", "mediastinum", "heart", "esophagus", "global"]  
    
    # ✅ 严格按照原始映射的器官疾病关系
    organ_disease_mapping = {  
        "lung": [7, 8, 9, 10, 11, 13, 15, 17],    # Emphysema, Atelectasis, Lung nodule, Lung opacity, Pulmonary fibrotic sequela, Mosaic attenuation pattern, Consolidation, Interlobular septal thickening
        "trachea and bronchie": [14, 16],         # Peribronchial thickening, Bronchiectasis
        "pleura": [12],                           # Pleural effusion
        "mediastinum": [6],                       # Lymphadenopathy
        "heart": [2, 3, 4],                       # Cardiomegaly, Pericardial effusion, Coronary artery wall calcification
        "esophagus": [5],                         # Hiatal hernia
        "global": [0, 1]                          # Medical material, Arterial wall calcification
    }  

    # 创建保存混淆矩阵的目录  
    confusion_matrix_dir = os.path.join(config.work_dir, "test_results", "test_confusion_matrices", f"epoch_{epoch}")  
    os.makedirs(confusion_matrix_dir, exist_ok=True)   
    
    # 创建保存 ROC 曲线的目录  
    roc_curve_dir = os.path.join(config.work_dir, "test_results", "roc_curves", f"epoch_{epoch}")  
    os.makedirs(roc_curve_dir, exist_ok=True)   

    with torch.no_grad():  
        for iter, data in enumerate(test_loader):  
            # if iter > 200:
            #     break
            images, abnormal_targets, sample_names = data  
            images, abnormal_targets = images.to(device), abnormal_targets.to(device)  

            seg_pred, abnormal_preds = model(images)  
            abnormal_pred = abnormal_preds[-1]  # (B, 18, D, H, W)
            
            # 18种疾病的频率
            disease_frequencies = [  
                0.102, 0.2837,  # Medical material, Arterial wall calcification
                0.1072, 0.0705, 0.2476,  # Cardiomegaly, Pericardial effusion, Coronary artery wall calcification
                0.1420, 0.2534, 0.1939, 0.2558, 0.4548,  # Hiatal hernia, Lymphadenopathy, Emphysema, Atelectasis, Lung nodule
                0.3666, 0.2672, 0.1185, 0.0744,  # Lung opacity, Pulmonary fibrotic sequela, Pleural effusion, Mosaic attenuation pattern
                0.1034, 0.1755, 0.0999, 0.0788  # Peribronchial thickening, Consolidation, Bronchiectasis, Interlobular septal thickening
            ]  

            disease_losses, abnormal_pred_avg = Abnormal_loss(
                seg_pred, abnormal_pred, abnormal_targets, disease_frequencies
            )  
            
            abnormal_targets_np = abnormal_targets.cpu().numpy()  
            abnormal_pred_avg_np = abnormal_pred_avg.cpu().numpy()  
            abnormal_targets_np = 1 - abnormal_targets_np  # 将阴性作为1
            abnormal_pred_avg_np = 1 - abnormal_pred_avg_np

            # 保存所有样本信息
            for batch_idx in range(images.size(0)):
                sample_name = sample_names[batch_idx]
                sample_pred = []
                sample_target = []
                
                for disease_idx in range(18):
                    pred_value = abnormal_pred_avg_np[batch_idx, disease_idx]
                    gt_value = abnormal_targets_np[batch_idx, disease_idx]
                    sample_pred.append(pred_value)
                    sample_target.append(gt_value)
                
                all_sample_names.append(sample_name)
                all_sample_predictions.append(sample_pred)
                all_sample_targets.append(sample_target)
            
            # 收集18种疾病的预测结果  
            for disease_idx in range(18):  
                processed_predictions[disease_idx].extend(abnormal_pred_avg_np[:, disease_idx])  
                all_targets[disease_idx].extend(abnormal_targets_np[:, disease_idx])  

            # 记录损失  
            for i, disease_loss in enumerate(disease_losses):  
                abnormal_loss_list[i].append(disease_loss.item())  

            # 计算平均损失（18种疾病）
            abnormal_loss = torch.mean(torch.stack(disease_losses))  
            total_loss = config.abnormal_loss_weight * abnormal_loss  
            loss_list.append(total_loss.item())  

            if save_heatmap and iter in selected_indices:  
                for batch_idx in range(images.size(0)):  
                    heatmap_data.append({  
                        'predictions': [  
                            abnormal_preds[0][batch_idx:batch_idx+1].cpu(),  
                            abnormal_preds[1][batch_idx:batch_idx+1].cpu()   
                        ],
                        'seg_pred': seg_pred[batch_idx:batch_idx+1].cpu(),
                        'targets': abnormal_targets[batch_idx:batch_idx+1].cpu(),  
                        'images': images[batch_idx:batch_idx+1].cpu(),  
                        'sample_name': sample_names[batch_idx]  
                    })
    
    # 初始化指标数组（18种疾病）
    precision_per_disease = np.zeros(18)  
    recall_per_disease = np.zeros(18)  
    accuracy_per_disease = np.zeros(18)  
    f1_per_disease = np.zeros(18)  
    auroc_per_disease = np.zeros(18)  

    roc_dir = os.path.join(config.work_dir, "test_results", "roc_analysis")
    os.makedirs(roc_dir, exist_ok=True)
    
    # 创建一个字典保存所有疾病的数据  
    data_dict = {} 

    best_thresholds_18 = np.array([0.5] * 18)  # 初始化每个疾病的最佳阈值 
    
    # 循环处理每种疾病（18种）
    print("\n" + "="*100)
    print("计算每种疾病的最佳阈值（基于Youden指数）")
    print("="*100)
    
    for disease_idx in range(18):  
        # 获取真实标签和预测概率  
        y_true = np.array(all_targets[disease_idx])  
        y_pred_proba = np.array(processed_predictions[disease_idx])  
        
        # 将数据存入字典  
        disease_name = disease_names_18[disease_idx]  
        data_dict[f"{disease_name}_y_true"] = y_true  
        data_dict[f"{disease_name}_y_pred_proba"] = y_pred_proba  

        # 保存 y_true 和 y_pred_proba
        y_true_path = os.path.join(roc_curve_dir, f'y_true_{disease_name}.npy')  
        y_pred_proba_path = os.path.join(roc_curve_dir, f'y_pred_proba_{disease_name}.npy')  
        
        np.save(y_true_path, y_true)  
        np.save(y_pred_proba_path, y_pred_proba) 

        # 计算ROC曲线  
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)  
        
        # 计算Youden指数
        youden_index = tpr - fpr  
        optimal_idx = np.argmax(youden_index)  
        best_threshold = thresholds[optimal_idx]  
        best_thresholds_18[disease_idx] = best_threshold

        # 打印最佳阈值信息
        print(f"\n{disease_idx:2d}. {disease_names_18[disease_idx]}")  
        print(f"     Optimal Threshold: {best_threshold:.4f}")  
        print(f"     TPR (Sensitivity): {tpr[optimal_idx]:.4f}")  
        print(f"     FPR: {fpr[optimal_idx]:.4f}")  
        print(f"     Youden Index: {youden_index[optimal_idx]:.4f}")
        
        # 绘制ROC曲线  
        plt.figure(figsize=(8, 6))  
        plt.plot(fpr, tpr, label="ROC Curve", color="blue", linewidth=2)  
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
        plt.xlabel("False Positive Rate (FPR)")  
        plt.ylabel("True Positive Rate (TPR)")  
        plt.title(f"ROC Curve for {disease_names_18[disease_idx]}")  
        plt.legend(loc="lower right")  
        
        # 标注阈值点
        for i, threshold in enumerate(thresholds):  
            if i % 10 == 0:
                plt.scatter(fpr[i], tpr[i], color="red", s=10)
                plt.text(fpr[i], tpr[i], f"{threshold:.2f}", fontsize=8, color="black")  
        
        # 保存图片  
        roc_path = os.path.join(roc_dir, f"roc_curve_{disease_names_18[disease_idx]}.png")  
        plt.savefig(roc_path, dpi=300)  
        plt.close()

    # 保存所有疾病数据
    np_path = os.path.join(roc_dir, "disease_data.npz")  
    np.savez_compressed(np_path, **data_dict)  
    print(f"\n保存所有疾病数据到: {np_path}")

    # 创建CSV文件并打印样本预测结果
    predictions_csv_path = os.path.join(config.work_dir, "test_results", f"sample_predictions_epoch_{epoch}.csv")
    with open(predictions_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # 写入CSV头部
        header = ['Sample_Name']
        for disease in disease_names_18:
            header.append(f"{disease}_Prob")
            header.append(f"{disease}_Pred")
            header.append(f"{disease}_GT")
        csv_writer.writerow(header)
        
        # 对每个样本应用阈值并打印结果
        print("\n" + "="*100)
        print("样本预测详情（前5个样本）")
        print("="*100)
        
        for idx, sample_name in enumerate(all_sample_names):
            if idx < 5:  # 只打印前5个样本
                print(f"\n样本: {sample_name}")
                print("Disease                              | Probability | Prediction | Ground Truth")
                print("-------------------------------------|-------------|------------|-------------")
            
            sample_row = [sample_name]
            sample_pred = all_sample_predictions[idx]
            sample_target = all_sample_targets[idx]
            
            for disease_idx, disease_name in enumerate(disease_names_18):
                prob_value = sample_pred[disease_idx]
                pred_value = 1 if prob_value > best_thresholds_18[disease_idx] else 0
                gt_value = int(sample_target[disease_idx])
                
                if idx < 5:  # 只打印前5个样本
                    print(f"{disease_name:35} | {prob_value:.6f} | {pred_value:10} | {gt_value}")
                
                sample_row.append(f"{prob_value:.6f}")
                sample_row.append(f"{pred_value}")
                sample_row.append(f"{gt_value}")
            
            csv_writer.writerow(sample_row)
    
    print(f"\n所有样本的预测值已保存到: {predictions_csv_path}")

    # 计算所有指标（18种疾病）
    print("\n" + "="*100)
    print("计算每种疾病的评估指标")
    print("="*100)
    
    for disease_idx in range(18):  
        y_true = np.array(all_targets[disease_idx])  
        y_pred_proba = np.array(processed_predictions[disease_idx])  
        y_pred = (y_pred_proba > best_thresholds_18[disease_idx]).astype(int)  

        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)  
        save_path = plot_confusion_matrix(y_true, y_pred, disease_names_18[disease_idx], confusion_matrix_dir)  

        # 安全访问混淆矩阵元素
        if cm.shape[0] > 1 and cm.shape[1] > 1:
            tp = cm[1, 1]
            fp = cm[0, 1]
            fn = cm[1, 0]
            tn = cm[0, 0]
        else:
            tp = 0
            fp = 0
            fn = 0
            tn = cm[0, 0] if cm.size > 0 else 0

        # 更新TP、TN、FP、FN的总和  
        tp_sum[disease_idx] = tp  
        tn_sum[disease_idx] = tn  
        fp_sum[disease_idx] = fp  
        fn_sum[disease_idx] = fn  

        # 计算AUROC  
        try:  
            auroc = roc_auc_score(y_true, y_pred_proba)  
        except ValueError:  
            auroc = 0.0  
        auroc_per_disease[disease_idx] = auroc  

        # 计算每个疾病的指标  
        precision_per_disease[disease_idx] = tp / (tp + fp + 1e-8)  
        recall_per_disease[disease_idx] = tp / (tp + fn + 1e-8)  
        accuracy_per_disease[disease_idx] = (tp + tn) / (tp + tn + fp + fn + 1e-8)  
        f1_per_disease[disease_idx] = 2 * precision_per_disease[disease_idx] * recall_per_disease[disease_idx] / (precision_per_disease[disease_idx] + recall_per_disease[disease_idx] + 1e-8)  

        # 计算并绘制ROC曲线（带阈值标注）
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)  
        roc_auc = auc(fpr, tpr)  

        plt.figure()  
        plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))  
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')  

        # 标注每隔20个阈值的点  
        for i in range(0, len(thresholds), 20):
            plt.scatter(fpr[i], tpr[i], color='black', s=10)
            plt.text(fpr[i], tpr[i], f'{thresholds[i]:.2f}', fontsize=8, color='black')  

        plt.xlim([0.0, 1.0])  
        plt.ylim([0.0, 1.05])  
        plt.xlabel('False Positive Rate')  
        plt.ylabel('True Positive Rate')  
        plt.title(f'ROC Curve for {disease_names_18[disease_idx]}')  
        plt.legend(loc='lower right')  

        disease_name_safe = disease_names_18[disease_idx].replace(" ", "_")
        roc_curve_path = os.path.join(roc_curve_dir, f'roc_curve_{disease_name_safe}.png')  
        plt.savefig(roc_curve_path)  
        plt.close()

    # 保存热图  
    if save_heatmap and heatmap_data:  
        logger.info("Generating and saving heatmaps with optimal thresholds...")  
        for data in heatmap_data:  
            save_prediction_heatmaps(  
                predictions=data['predictions'],  
                segmentation_preds=data['seg_pred'],
                targets=data['targets'],  
                images=data['images'],  
                epoch=epoch,  
                organ_names=organ_names,  
                sample_idx=data['sample_name'],  
                base_dir=save_dir,  
                seg_threshold=0.5,
                topk=3, 
                abnormal_threshold=best_thresholds_18,
                disease_names=disease_names_18
            )  

    # 计算平均指标（18种疾病）
    avg_loss = np.mean(loss_list)  
    avg_abnormal_loss = np.mean([np.mean(losses) for losses in abnormal_loss_list if losses])  
    avg_accuracy = np.mean(accuracy_per_disease)  
    avg_precision = np.mean(precision_per_disease)  
    avg_recall = np.mean(recall_per_disease)  
    avg_f1 = np.mean(f1_per_disease)  
    avg_auroc = np.mean(auroc_per_disease)  

    # 打印详细的评估结果
    print("\n" + "="*100)
    print(f"{'Test Epoch ' + str(epoch) + ' - 详细评估结果':^100}")
    print("="*100)
    
    # 打印每种疾病的指标
    print(f"\n{'疾病指标详情':^100}")
    print("-"*100)
    print(f"{'ID':<3} {'Disease Name':<40} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUROC':>7}")
    print("-"*100)
    
    for disease_idx in range(18):
        print(f"{disease_idx:<3} {disease_names_18[disease_idx]:<40} "
              f"{accuracy_per_disease[disease_idx]:>7.4f} "
              f"{precision_per_disease[disease_idx]:>7.4f} "
              f"{recall_per_disease[disease_idx]:>7.4f} "
              f"{f1_per_disease[disease_idx]:>7.4f} "
              f"{auroc_per_disease[disease_idx]:>7.4f}")
    
    print("-"*100)
    
    # 打印平均指标
    print(f"\n{'平均指标（18种疾病）':^100}")
    print("="*100)
    print(f"Average Loss:      {avg_loss:.6f}")
    print(f"Average Accuracy:  {avg_accuracy:.6f}")
    print(f"Average Precision: {avg_precision:.6f}")
    print(f"Average Recall:    {avg_recall:.6f}")
    print(f"Average F1 Score:  {avg_f1:.6f}")
    print(f"Average AUROC:     {avg_auroc:.6f}")
    print("="*100)
    
    # 保存指标到文件
    metrics_file = os.path.join(config.work_dir, "test_results", f"metrics_epoch_{epoch}.txt")
    with open(metrics_file, 'w') as f:
        f.write(f"Test Epoch {epoch} - Evaluation Metrics (18 Diseases)\n")
        f.write("="*100 + "\n\n")
        
        f.write("Per-Disease Metrics:\n")
        f.write("-"*100 + "\n")
        f.write(f"{'ID':<3} {'Disease Name':<40} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUROC':>7}\n")
        f.write("-"*100 + "\n")
        
        for disease_idx in range(18):
            f.write(f"{disease_idx:<3} {disease_names_18[disease_idx]:<40} "
                   f"{accuracy_per_disease[disease_idx]:>7.4f} "
                   f"{precision_per_disease[disease_idx]:>7.4f} "
                   f"{recall_per_disease[disease_idx]:>7.4f} "
                   f"{f1_per_disease[disease_idx]:>7.4f} "
                   f"{auroc_per_disease[disease_idx]:>7.4f}\n")
        
        f.write("-"*100 + "\n\n")
        f.write("Average Metrics (18 Diseases):\n")
        f.write("="*100 + "\n")
        f.write(f"Average Loss:      {avg_loss:.6f}\n")
        f.write(f"Average Accuracy:  {avg_accuracy:.6f}\n")
        f.write(f"Average Precision: {avg_precision:.6f}\n")
        f.write(f"Average Recall:    {avg_recall:.6f}\n")
        f.write(f"Average F1 Score:  {avg_f1:.6f}\n")
        f.write(f"Average AUROC:     {avg_auroc:.6f}\n")
        f.write("="*100 + "\n")
    
    print(f"\n评估指标已保存到: {metrics_file}")
    
    # 记录到tensorboard（如果需要）
    if writer is not None:  
        # 记录总体指标  
        writer.add_scalar('Test/Total_Loss', avg_loss, epoch)  
        writer.add_scalar('Test/Abnormal_Loss', avg_abnormal_loss, epoch)  
        
        # 记录平均指标（18种疾病）
        writer.add_scalar('Test/Avg_Accuracy', avg_accuracy, epoch)  
        writer.add_scalar('Test/Avg_Precision', avg_precision, epoch)  
        writer.add_scalar('Test/Avg_Recall', avg_recall, epoch)  
        writer.add_scalar('Test/Avg_F1', avg_f1, epoch)  
        writer.add_scalar('Test/Avg_AUROC', avg_auroc, epoch)  
        
        # 记录每种疾病的指标  
        for i in range(18):  
            writer.add_scalar(f'Test/Disease_{disease_names_18[i]}/Accuracy', accuracy_per_disease[i], epoch)  
            writer.add_scalar(f'Test/Disease_{disease_names_18[i]}/Precision', precision_per_disease[i], epoch)  
            writer.add_scalar(f'Test/Disease_{disease_names_18[i]}/Recall', recall_per_disease[i], epoch)  
            writer.add_scalar(f'Test/Disease_{disease_names_18[i]}/F1', f1_per_disease[i], epoch)  
            writer.add_scalar(f'Test/Disease_{disease_names_18[i]}/AUROC', auroc_per_disease[i], epoch)  

    logger.info(f"Test epoch {epoch} completed. Average Loss: {avg_loss:.6f}, Average AUROC: {avg_auroc:.6f}")

    return avg_loss


def Abnormal_loss(seg_pred, abnormal_pred, abnormal_targets, disease_frequencies, k=3, epsilon=1e-6, seg_threshold=0.5):  
    """  
    计算异常检测损失（18种疾病）
    
    Args:  
        seg_pred: 分割预测 (B, 7, D, H, W) - 包含global通道
        abnormal_pred: 疾病预测 (B, 18, D, H, W) - 18种疾病
        abnormal_targets: 疾病标签 (B, 18) - 18种疾病标签
        disease_frequencies: 18种疾病的阳性样本频率列表  
        k: top-k取值  
        epsilon: 数值稳定性常数  
        seg_threshold: 分割掩码阈值  
    """  
    B, _, D, H, W = seg_pred.shape  
    num_diseases = abnormal_pred.shape[1]  # 18个疾病通道  
    
    # ✅ 严格按照原始映射：18种疾病到器官的映射
    disease_to_organ = [
        6,  # 0: Medical material -> global
        6,  # 1: Arterial wall calcification -> global
        4,  # 2: Cardiomegaly -> heart
        4,  # 3: Pericardial effusion -> heart
        4,  # 4: Coronary artery wall calcification -> heart
        5,  # 5: Hiatal hernia -> esophagus
        3,  # 6: Lymphadenopathy -> mediastinum
        0,  # 7: Emphysema -> lung
        0,  # 8: Atelectasis -> lung
        0,  # 9: Lung nodule -> lung
        0,  # 10: Lung opacity -> lung
        0,  # 11: Pulmonary fibrotic sequela -> lung
        2,  # 12: Pleural effusion -> pleura
        0,  # 13: Mosaic attenuation pattern -> lung
        1,  # 14: Peribronchial thickening -> trachea and bronchie
        0,  # 15: Consolidation -> lung
        1,  # 16: Bronchiectasis -> trachea and bronchie
        0   # 17: Interlobular septal thickening -> lung
    ]
    
    # 对分割预测进行二值化  
    seg_mask = (seg_pred > seg_threshold).float()  
    
    disease_losses = []  
    disease_predictions = []  
    
    for disease_idx in range(num_diseases):  
        # 获取对应的器官索引  
        organ_idx = disease_to_organ[disease_idx]  
        
        # 将对应器官的分割掩码与疾病预测相乘  
        organ_mask = seg_mask[:, organ_idx:organ_idx+1]  
        disease_pred = abnormal_pred[:, disease_idx:disease_idx+1]  
        
        final_pred = organ_mask * disease_pred  
        
        # 取top-k值的平均  
        top_k_values, _ = torch.topk(final_pred.view(B, -1), k, dim=1)  
        avg_top_k = top_k_values.mean(dim=1)  
        
        # 使用疾病的频率计算权重
        freq = disease_frequencies[disease_idx]
        
        if freq > 0:  # 对存在的疾病计算权重
            weight_pos = (1 - freq + epsilon) / (freq + epsilon)  
            weight_neg = 1.0  
            
            # 计算损失  
            target = abnormal_targets[:, disease_idx]  
            loss = weighted_binary_cross_entropy(avg_top_k, target, weight_pos, weight_neg)  
            loss = loss.mean()  
        else:  # 对不存在的疾病，设置损失为0  
            loss = torch.tensor(0.0, device=abnormal_pred.device)  
        
        disease_losses.append(loss)  
        disease_predictions.append(avg_top_k)  
    
    return disease_losses, torch.stack(disease_predictions, dim=1)


def weighted_binary_cross_entropy(pred, target, weight_pos, weight_neg):
    """加权二元交叉熵损失"""
    bce = -(weight_pos * target * torch.log(pred + 1e-8) + 
            weight_neg * (1 - target) * torch.log(1 - pred + 1e-8))
    return bce


def upsample_3d(tensor, target_size):
    """3D上采样"""
    import torch.nn.functional as F
    return F.interpolate(tensor, size=target_size, mode='trilinear', align_corners=False)


def plot_confusion_matrix(y_true, y_pred, disease_name, save_dir):
    """绘制并保存混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {disease_name}')
    plt.colorbar()
    
    # 添加数值标注
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'confusion_matrix_{disease_name}.png')
    plt.savefig(save_path)
    plt.close()
    
    return save_path


def save_prediction_heatmaps(  
    predictions,   
    segmentation_preds,
    targets,   
    images,   
    epoch,   
    organ_names,   
    sample_idx,  
    base_dir=None,   
    seg_threshold=0.5,
    topk=3,   
    abnormal_threshold=None,
    disease_names=None
):  
    """  
    保存预测热图（18种疾病）
    """  
    # 解包多尺度预测结果  
    low_res_preds, high_res_preds = predictions  
    
    # 获取目标大小（高分辨率）  
    _, _, target_depth, target_height, target_width = high_res_preds.shape  
    
    # 对低分辨率预测进行上采样  
    low_res_upsampled = upsample_3d(low_res_preds, (target_depth, target_height, target_width))  
    
    # 选择使用高分辨率预测  
    high_res_pred = high_res_preds[0].cpu().numpy()  # [18, D, H, W]  
    low_res_pred = low_res_upsampled[0].cpu().numpy()  # [18, D, H, W]  
    
    if base_dir is None:  
        base_dir = Path.cwd()  
    else:  
        base_dir = Path(base_dir)  
    
    save_dir = base_dir / "prediction_heatmaps"  
    epoch_dir = save_dir / f"epoch_{epoch}"  
    sample_dir = epoch_dir / str(sample_idx)  
    sample_dir.mkdir(parents=True, exist_ok=True)  
    
    # 处理分割预测结果
    seg_pred = segmentation_preds[0].cpu().numpy()  # [7, D, H, W]  
    seg_mask = (seg_pred > seg_threshold).astype(np.float32)
    
    target = targets[0].cpu().numpy()  # [18]  
    original_image = images[0, 0].cpu().numpy()  # [D, H, W]  
    
    # 保存原始图像  
    affine = np.eye(4)  
    original_nifti = nib.Nifti1Image(original_image, affine)  
    original_nifti.header['descrip'] = f'Original 3D Image, Epoch: {epoch}'  
    original_save_path = sample_dir / f"original_image.nii.gz"  
    nib.save(original_nifti, original_save_path)  
    
    # ✅ 严格按照原始映射：疾病-器官映射关系（18种疾病）
    disease_organ_mapping = {
        "Medical material": ["global"],
        "Arterial wall calcification": ["global"],
        "Cardiomegaly": ["heart"],
        "Pericardial effusion": ["heart"],
        "Coronary artery wall calcification": ["heart"],
        "Hiatal hernia": ["esophagus"],
        "Lymphadenopathy": ["mediastinum"],
        "Emphysema": ["lung"],
        "Atelectasis": ["lung"],
        "Lung nodule": ["lung"],
        "Lung opacity": ["lung"],
        "Pulmonary fibrotic sequela": ["lung"],
        "Pleural effusion": ["pleura"],
        "Mosaic attenuation pattern": ["lung"],
        "Peribronchial thickening": ["trachea and bronchie"],
        "Consolidation": ["lung"],
        "Bronchiectasis": ["trachea and bronchie"],
        "Interlobular septal thickening": ["lung"]
    }
    
    # 创建预测信息文件  
    info_file = sample_dir / "prediction_info.txt"  
    with open(info_file, "w") as f:  
        f.write(f"Epoch: {epoch}\n")  
        f.write(f"Sample: {sample_idx}\n")  
        f.write(f"Abnormal Detection Parameters: top-{topk}\n")  
        f.write("Disease Predictions (18 Diseases):\n")  
        
        # 存储每个尺度的预测结果  
        prediction_results = {  
            "High-Res": high_res_pred,  
            "Low-Res": low_res_pred  
        }  
        
        # 遍历不同分辨率的预测  
        for res_name, pred in prediction_results.items():  
            f.write(f"\n{res_name} Predictions:\n")  
            
            # 对18种疾病进行处理
            for disease_idx, disease_name in enumerate(disease_names):  
                # 获取疾病预测
                disease_pred = pred[disease_idx]  # [D, H, W]
                disease_label = int(target[disease_idx])
                
                # 获取相关器官的分割掩码
                related_organs = disease_organ_mapping[disease_name]
                combined_mask = np.zeros_like(seg_mask[0])
                for organ_name in related_organs:
                    organ_idx = organ_names.index(organ_name)
                    combined_mask = np.maximum(combined_mask, seg_mask[organ_idx])
                
                # 计算combined_pred
                combined_pred = disease_pred * combined_mask
                
                # 计算top-k平均值
                topk_mean = np.mean(np.sort(combined_pred.flatten())[-topk:])  
                
                # 获取阈值和预测结果
                current_threshold = abnormal_threshold[disease_idx]
                pred_abnormal = int(topk_mean > current_threshold)
                
                # 写入预测信息  
                f.write(f"\n  {disease_idx:2d}. {disease_name}:\n")  
                f.write(f"      Ground Truth: {disease_label}\n")
                f.write(f"      Prediction: {pred_abnormal}\n")  
                f.write(f"      Top-{topk} Mean: {topk_mean:.4f}\n")  
                f.write(f"      Threshold: {current_threshold:.4f}\n")  
                f.write(f"      Related Organs: {', '.join(related_organs)}\n")  
                
                # 创建文件名
                result_str = f"GT{disease_label}_PD{pred_abnormal}"
                
                # 保存预测热图
                nifti_img = nib.Nifti1Image(disease_pred, affine)  
                nifti_img.header['descrip'] = (f'{res_name} 3D Disease: {disease_name}, '
                                              f'Top-{topk} Mean: {topk_mean:.4f}')
                save_path = sample_dir / f"{disease_idx:02d}_{disease_name}_{result_str}_{res_name.lower()}_pred.nii.gz"  
                nib.save(nifti_img, save_path)
                
                # 保存结合器官的预测
                combined_nifti = nib.Nifti1Image(combined_pred, affine)
                combined_save_path = sample_dir / f"{disease_idx:02d}_{disease_name}_{result_str}_{res_name.lower()}_combined_pred.nii.gz"
                nib.save(combined_nifti, combined_save_path)
                
                # 保存器官掩码（只保存一次）
                if res_name == "High-Res":
                    mask_nifti = nib.Nifti1Image(combined_mask, affine)
                    mask_save_path = sample_dir / f"{disease_idx:02d}_{disease_name}_organ_mask.nii.gz"
                    nib.save(mask_nifti, mask_save_path)

    return save_dir


def choose_best_threshold_lefttop(y_true: np.ndarray, y_prob: np.ndarray, fallback: float = 0.5):
    """
    选取 ROC 曲线中距离 (0,1) 最近的点作为最佳阈值；并列时先最大化 TPR 再最小化 FPR。
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
        # 标签全正或全负等情况
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = float('nan')
        return fallback, float(auc)


def test_one_epoch_thresholds_changed(test_loader, model, segmentation_criterion, abnormal_criterion, epoch, logger, config, writer, device, best_thresholds, save_heatmap=True):  
    model.eval()  
    loss_list = []  
    abnormal_loss_list = [[] for _ in range(18)]  # 18种疾病的损失记录
    sample_idx = 0  
    np.random.seed(42)  
    
    dataset_size = len(test_loader.dataset)  
    selected_indices = set(np.random.choice(dataset_size, min(40, dataset_size), replace=False))  
    current_dir = config.work_dir  

    if save_heatmap:  
        save_dir = os.path.join(config.work_dir, "test_results")  
        os.makedirs(save_dir, exist_ok=True)  
        heatmap_data = []  

    logger.info(f"Test epoch {epoch}: Abnormal weight = {config.abnormal_loss_weight}")  

    test_loader = tqdm(test_loader, desc=f"Epoch {epoch} Test", leave=True, dynamic_ncols=True)  

    # 初始化评估指标数组（使用18种疾病）  
    processed_predictions = [[] for _ in range(18)]  
    all_targets = [[] for _ in range(18)]  
    tp_sum = np.zeros(18, dtype=int)  
    tn_sum = np.zeros(18, dtype=int)  
    fp_sum = np.zeros(18, dtype=int)  
    fn_sum = np.zeros(18, dtype=int)  

    # 保存样本信息的列表
    all_sample_names = []
    all_sample_predictions = []
    all_sample_targets = []

    # 定义18种疾病名称
    disease_names_18 = [  
        "Medical material",                      # 0
        "Arterial wall calcification",           # 1
        "Cardiomegaly",                          # 2
        "Pericardial effusion",                  # 3
        "Coronary artery wall calcification",    # 4
        "Hiatal hernia",                         # 5
        "Lymphadenopathy",                       # 6
        "Emphysema",                             # 7
        "Atelectasis",                           # 8
        "Lung nodule",                           # 9
        "Lung opacity",                          # 10
        "Pulmonary fibrotic sequela",            # 11
        "Pleural effusion",                      # 12
        "Mosaic attenuation pattern",            # 13
        "Peribronchial thickening",              # 14
        "Consolidation",                         # 15
        "Bronchiectasis",                        # 16
        "Interlobular septal thickening"         # 17
    ]  

    # 器官名称（包含global通道）
    organ_names = ["lung", "trachea and bronchie", "pleura", "mediastinum", "heart", "esophagus", "global"]  
    
    # ✅ 严格按照原始映射的器官疾病关系
    organ_disease_mapping = {  
        "lung": [7, 8, 9, 10, 11, 13, 15, 17],    # Emphysema, Atelectasis, Lung nodule, Lung opacity, Pulmonary fibrotic sequela, Mosaic attenuation pattern, Consolidation, Interlobular septal thickening
        "trachea and bronchie": [14, 16],         # Peribronchial thickening, Bronchiectasis
        "pleura": [12],                           # Pleural effusion
        "mediastinum": [6],                       # Lymphadenopathy
        "heart": [2, 3, 4],                       # Cardiomegaly, Pericardial effusion, Coronary artery wall calcification
        "esophagus": [5],                         # Hiatal hernia
        "global": [0, 1]                          # Medical material, Arterial wall calcification
    }  

    # 创建保存混淆矩阵的目录  
    confusion_matrix_dir = os.path.join(config.work_dir, "test_results", "test_confusion_matrices", f"epoch_{epoch}")  
    os.makedirs(confusion_matrix_dir, exist_ok=True)   
    
    # 创建保存 ROC 曲线的目录  
    roc_curve_dir = os.path.join(config.work_dir, "test_results", "roc_curves", f"epoch_{epoch}")  
    os.makedirs(roc_curve_dir, exist_ok=True)   

    with torch.no_grad():  
        for iter, data in enumerate(test_loader):  
            # if iter > 200:
            #     break
            images, abnormal_targets, sample_names = data  
            images, abnormal_targets = images.to(device), abnormal_targets.to(device)  

            seg_pred, abnormal_preds = model(images)  
            abnormal_pred = abnormal_preds[-1]  # (B, 18, D, H, W)
            
            # 18种疾病的频率
            disease_frequencies = [  
                0.102, 0.2837,  # Medical material, Arterial wall calcification
                0.1072, 0.0705, 0.2476,  # Cardiomegaly, Pericardial effusion, Coronary artery wall calcification
                0.1420, 0.2534, 0.1939, 0.2558, 0.4548,  # Hiatal hernia, Lymphadenopathy, Emphysema, Atelectasis, Lung nodule
                0.3666, 0.2672, 0.1185, 0.0744,  # Lung opacity, Pulmonary fibrotic sequela, Pleural effusion, Mosaic attenuation pattern
                0.1034, 0.1755, 0.0999, 0.0788  # Peribronchial thickening, Consolidation, Bronchiectasis, Interlobular septal thickening
            ]  

            disease_losses, abnormal_pred_avg = Abnormal_loss(
                seg_pred, abnormal_pred, abnormal_targets, disease_frequencies
            )  
            
            abnormal_targets_np = abnormal_targets.cpu().numpy()  
            abnormal_pred_avg_np = abnormal_pred_avg.cpu().numpy()  
            abnormal_targets_np = 1 - abnormal_targets_np  # 将阴性作为1
            abnormal_pred_avg_np = 1 - abnormal_pred_avg_np

            # 保存所有样本信息
            for batch_idx in range(images.size(0)):
                sample_name = sample_names[batch_idx]
                sample_pred = []
                sample_target = []
                
                for disease_idx in range(18):
                    pred_value = abnormal_pred_avg_np[batch_idx, disease_idx]
                    gt_value = abnormal_targets_np[batch_idx, disease_idx]
                    sample_pred.append(pred_value)
                    sample_target.append(gt_value)
                
                all_sample_names.append(sample_name)
                all_sample_predictions.append(sample_pred)
                all_sample_targets.append(sample_target)
            
            # 收集18种疾病的预测结果  
            for disease_idx in range(18):  
                processed_predictions[disease_idx].extend(abnormal_pred_avg_np[:, disease_idx])  
                all_targets[disease_idx].extend(abnormal_targets_np[:, disease_idx])  

            # 记录损失  
            for i, disease_loss in enumerate(disease_losses):  
                abnormal_loss_list[i].append(disease_loss.item())  

            # 计算平均损失（18种疾病）
            abnormal_loss = torch.mean(torch.stack(disease_losses))  
            total_loss = config.abnormal_loss_weight * abnormal_loss  
            loss_list.append(total_loss.item())  

            if save_heatmap and iter in selected_indices:  
                for batch_idx in range(images.size(0)):  
                    heatmap_data.append({  
                        'predictions': [  
                            abnormal_preds[0][batch_idx:batch_idx+1].cpu(),  
                            abnormal_preds[1][batch_idx:batch_idx+1].cpu()   
                        ],
                        'seg_pred': seg_pred[batch_idx:batch_idx+1].cpu(),
                        'targets': abnormal_targets[batch_idx:batch_idx+1].cpu(),  
                        'images': images[batch_idx:batch_idx+1].cpu(),  
                        'sample_name': sample_names[batch_idx]  
                    })
    
    # 初始化指标数组（18种疾病）
    precision_per_disease = np.zeros(18)  
    recall_per_disease = np.zeros(18)  
    accuracy_per_disease = np.zeros(18)  
    f1_per_disease = np.zeros(18)  
    auroc_per_disease = np.zeros(18)  

    roc_dir = os.path.join(config.work_dir, "test_results", "roc_analysis")
    os.makedirs(roc_dir, exist_ok=True)
    
    # 创建一个字典保存所有疾病的数据  
    data_dict = {} 

    best_thresholds_18 = np.zeros(18)  # 初始化每个疾病的最佳阈值
    
    # 循环处理每种疾病（18种）
    print("\n" + "="*100)
    print("计算每种疾病的最佳阈值（基于距离左上角最近点）")
    print("="*100)
    
    for disease_idx in range(18):  
        # 获取真实标签和预测概率  
        y_true = np.array(all_targets[disease_idx])  
        y_pred_proba = np.array(processed_predictions[disease_idx])  
        
        # 将数据存入字典  
        disease_name = disease_names_18[disease_idx]  
        data_dict[f"{disease_name}_y_true"] = y_true  
        data_dict[f"{disease_name}_y_pred_proba"] = y_pred_proba  

        # 保存 y_true 和 y_pred_proba
        y_true_path = os.path.join(roc_curve_dir, f'y_true_{disease_name}.npy')  
        y_pred_proba_path = os.path.join(roc_curve_dir, f'y_pred_proba_{disease_name}.npy')  
        
        np.save(y_true_path, y_true)  
        np.save(y_pred_proba_path, y_pred_proba) 

        # 使用新的阈值计算方法
        best_threshold, auroc = choose_best_threshold_lefttop(y_true, y_pred_proba, fallback=0.5)
        best_thresholds_18[disease_idx] = best_threshold
        auroc_per_disease[disease_idx] = auroc

        # 计算ROC曲线用于可视化
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
            # 找到最佳阈值对应的点
            dist = np.sqrt(fpr**2 + (1.0 - tpr)**2)
            best_idx = np.argmin(dist)
            best_fpr = fpr[best_idx]
            best_tpr = tpr[best_idx]
        except:
            fpr, tpr = [0, 1], [0, 1]
            best_fpr, best_tpr = 0, 1

        # 打印最佳阈值信息
        print(f"\n{disease_idx:2d}. {disease_names_18[disease_idx]}")  
        print(f"     Optimal Threshold: {best_threshold:.4f}")  
        print(f"     TPR (Sensitivity): {best_tpr:.4f}")  
        print(f"     FPR: {best_fpr:.4f}")  
        print(f"     Distance to (0,1): {np.sqrt(best_fpr**2 + (1.0 - best_tpr)**2):.4f}")
        print(f"     AUROC: {auroc:.4f}")
        
        # 绘制ROC曲线  
        plt.figure(figsize=(8, 6))  
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auroc:.4f})", color="blue", linewidth=2)  
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
        
        # 标注最佳阈值点
        plt.scatter(best_fpr, best_tpr, color='red', s=100, zorder=5, 
                   label=f'Best Threshold = {best_threshold:.4f}')
        plt.plot([best_fpr, 0], [best_tpr, 1], 'r--', alpha=0.3, linewidth=1)
        
        plt.xlabel("False Positive Rate (FPR)")  
        plt.ylabel("True Positive Rate (TPR)")  
        plt.title(f"ROC Curve for {disease_names_18[disease_idx]}")  
        plt.legend(loc="lower right")  
        plt.grid(True, alpha=0.3)
        
        # 保存图片  
        roc_path = os.path.join(roc_dir, f"roc_curve_{disease_names_18[disease_idx]}.png")  
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')  
        plt.close()

    # 保存所有疾病数据
    np_path = os.path.join(roc_dir, "disease_data.npz")  
    np.savez_compressed(np_path, **data_dict)  
    print(f"\n保存所有疾病数据到: {np_path}")

    # 创建CSV文件并打印样本预测结果
    predictions_csv_path = os.path.join(config.work_dir, "test_results", f"sample_predictions_epoch_{epoch}.csv")
    with open(predictions_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # 写入CSV头部
        header = ['Sample_Name']
        for disease in disease_names_18:
            header.append(f"{disease}_Prob")
            header.append(f"{disease}_Pred")
            header.append(f"{disease}_GT")
        csv_writer.writerow(header)
        
        # 对每个样本应用阈值并打印结果
        print("\n" + "="*100)
        print("样本预测详情（前5个样本）")
        print("="*100)
        
        for idx, sample_name in enumerate(all_sample_names):
            if idx < 5:  # 只打印前5个样本
                print(f"\n样本: {sample_name}")
                print("Disease                              | Probability | Prediction | Ground Truth")
                print("-------------------------------------|-------------|------------|-------------")
            
            sample_row = [sample_name]
            sample_pred = all_sample_predictions[idx]
            sample_target = all_sample_targets[idx]
            
            for disease_idx, disease_name in enumerate(disease_names_18):
                prob_value = sample_pred[disease_idx]
                pred_value = 1 if prob_value > best_thresholds_18[disease_idx] else 0
                gt_value = int(sample_target[disease_idx])
                
                if idx < 5:  # 只打印前5个样本
                    print(f"{disease_name:35} | {prob_value:.6f} | {pred_value:10} | {gt_value}")
                
                sample_row.append(f"{prob_value:.6f}")
                sample_row.append(f"{pred_value}")
                sample_row.append(f"{gt_value}")
            
            csv_writer.writerow(sample_row)
    
    print(f"\n所有样本的预测值已保存到: {predictions_csv_path}")

    # 计算所有指标（18种疾病）
    print("\n" + "="*100)
    print("计算每种疾病的评估指标")
    print("="*100)
    
    for disease_idx in range(18):  
        y_true = np.array(all_targets[disease_idx])  
        y_pred_proba = np.array(processed_predictions[disease_idx])  
        y_pred = (y_pred_proba > best_thresholds_18[disease_idx]).astype(int)  

        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)  
        save_path = plot_confusion_matrix(y_true, y_pred, disease_names_18[disease_idx], confusion_matrix_dir)  

        # 安全访问混淆矩阵元素
        if cm.shape[0] > 1 and cm.shape[1] > 1:
            tp = cm[1, 1]
            fp = cm[0, 1]
            fn = cm[1, 0]
            tn = cm[0, 0]
        else:
            tp = 0
            fp = 0
            fn = 0
            tn = cm[0, 0] if cm.size > 0 else 0

        # 更新TP、TN、FP、FN的总和  
        tp_sum[disease_idx] = tp  
        tn_sum[disease_idx] = tn  
        fp_sum[disease_idx] = fp  
        fn_sum[disease_idx] = fn  

        # 计算每个疾病的指标  
        precision_per_disease[disease_idx] = tp / (tp + fp + 1e-8)  
        recall_per_disease[disease_idx] = tp / (tp + fn + 1e-8)  
        accuracy_per_disease[disease_idx] = (tp + tn) / (tp + tn + fp + fn + 1e-8)  
        f1_per_disease[disease_idx] = 2 * precision_per_disease[disease_idx] * recall_per_disease[disease_idx] / (precision_per_disease[disease_idx] + recall_per_disease[disease_idx] + 1e-8)  

        # 绘制ROC曲线（带阈值标注）
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)  
            roc_auc = auc(fpr, tpr)  

            plt.figure()  
            plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))  
            plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')  

            # 找到最佳阈值对应的点并标注
            dist = np.sqrt(fpr**2 + (1.0 - tpr)**2)
            best_idx = np.argmin(dist)
            plt.scatter(fpr[best_idx], tpr[best_idx], color='red', s=100, zorder=5,
                       label=f'Best Threshold = {best_thresholds_18[disease_idx]:.4f}')

            # 标注每隔20个阈值的点  
            for i in range(0, len(thresholds), 20):
                plt.scatter(fpr[i], tpr[i], color='black', s=10)
                plt.text(fpr[i], tpr[i], f'{thresholds[i]:.2f}', fontsize=8, color='black')  

            plt.xlim([0.0, 1.0])  
            plt.ylim([0.0, 1.05])  
            plt.xlabel('False Positive Rate')  
            plt.ylabel('True Positive Rate')  
            plt.title(f'ROC Curve for {disease_names_18[disease_idx]}')  
            plt.legend(loc='lower right')  
            plt.grid(True, alpha=0.3)

            disease_name_safe = disease_names_18[disease_idx].replace(" ", "_")
            roc_curve_path = os.path.join(roc_curve_dir, f'roc_curve_{disease_name_safe}.png')  
            plt.savefig(roc_curve_path, bbox_inches='tight')  
            plt.close()
        except:
            pass

    # 保存热图  
    if save_heatmap and heatmap_data:  
        logger.info("Generating and saving heatmaps with optimal thresholds...")  
        for data in heatmap_data:  
            save_prediction_heatmaps(  
                predictions=data['predictions'],  
                segmentation_preds=data['seg_pred'],
                targets=data['targets'],  
                images=data['images'],  
                epoch=epoch,  
                organ_names=organ_names,  
                sample_idx=data['sample_name'],  
                base_dir=save_dir,  
                seg_threshold=0.5,
                topk=3, 
                abnormal_threshold=best_thresholds_18,
                disease_names=disease_names_18
            )  

    # 计算平均指标（18种疾病）
    avg_loss = np.mean(loss_list)  
    avg_abnormal_loss = np.mean([np.mean(losses) for losses in abnormal_loss_list if losses])  
    avg_accuracy = np.mean(accuracy_per_disease)  
    avg_precision = np.mean(precision_per_disease)  
    avg_recall = np.mean(recall_per_disease)  
    avg_f1 = np.mean(f1_per_disease)  
    avg_auroc = np.mean(auroc_per_disease)  

    # 打印详细的评估结果
    print("\n" + "="*100)
    print(f"{'Test Epoch ' + str(epoch) + ' - 详细评估结果':^100}")
    print("="*100)
    
    # 打印每种疾病的指标
    print(f"\n{'疾病指标详情':^100}")
    print("-"*100)
    print(f"{'ID':<3} {'Disease Name':<40} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUROC':>7}")
    print("-"*100)
    
    for disease_idx in range(18):
        print(f"{disease_idx:<3} {disease_names_18[disease_idx]:<40} "
              f"{accuracy_per_disease[disease_idx]:>7.4f} "
              f"{precision_per_disease[disease_idx]:>7.4f} "
              f"{recall_per_disease[disease_idx]:>7.4f} "
              f"{f1_per_disease[disease_idx]:>7.4f} "
              f"{auroc_per_disease[disease_idx]:>7.4f}")
    
    print("-"*100)
    
    # 打印平均指标
    print(f"\n{'平均指标（18种疾病）':^100}")
    print("="*100)
    print(f"Average Loss:      {avg_loss:.6f}")
    print(f"Average Accuracy:  {avg_accuracy:.6f}")
    print(f"Average Precision: {avg_precision:.6f}")
    print(f"Average Recall:    {avg_recall:.6f}")
    print(f"Average F1 Score:  {avg_f1:.6f}")
    print(f"Average AUROC:     {avg_auroc:.6f}")
    print("="*100)
    
    # 保存指标到文件
    metrics_file = os.path.join(config.work_dir, "test_results", f"metrics_epoch_{epoch}.txt")
    with open(metrics_file, 'w') as f:
        f.write(f"Test Epoch {epoch} - Evaluation Metrics (18 Diseases)\n")
        f.write(f"Threshold Selection Method: Distance to Left-Top Corner (0,1)\n")
        f.write("="*100 + "\n\n")
        
        f.write("Optimal Thresholds:\n")
        for disease_idx in range(18):
            f.write(f"  {disease_idx:2d}. {disease_names_18[disease_idx]:<40} : {best_thresholds_18[disease_idx]:.4f}\n")
        f.write("\n")
        
        f.write("Per-Disease Metrics:\n")
        f.write("-"*100 + "\n")
        f.write(f"{'ID':<3} {'Disease Name':<40} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUROC':>7}\n")
        f.write("-"*100 + "\n")
        
        for disease_idx in range(18):
            f.write(f"{disease_idx:<3} {disease_names_18[disease_idx]:<40} "
                   f"{accuracy_per_disease[disease_idx]:>7.4f} "
                   f"{precision_per_disease[disease_idx]:>7.4f} "
                   f"{recall_per_disease[disease_idx]:>7.4f} "
                   f"{f1_per_disease[disease_idx]:>7.4f} "
                   f"{auroc_per_disease[disease_idx]:>7.4f}\n")
        
        f.write("-"*100 + "\n\n")
        f.write("Average Metrics (18 Diseases):\n")
        f.write("="*100 + "\n")
        f.write(f"Average Loss:      {avg_loss:.6f}\n")
        f.write(f"Average Accuracy:  {avg_accuracy:.6f}\n")
        f.write(f"Average Precision: {avg_precision:.6f}\n")
        f.write(f"Average Recall:    {avg_recall:.6f}\n")
        f.write(f"Average F1 Score:  {avg_f1:.6f}\n")
        f.write(f"Average AUROC:     {avg_auroc:.6f}\n")
        f.write("="*100 + "\n")
    
    print(f"\n评估指标已保存到: {metrics_file}")
    
    # 记录到tensorboard（如果需要）
    if writer is not None:  
        # 记录总体指标  
        writer.add_scalar('Test/Total_Loss', avg_loss, epoch)  
        writer.add_scalar('Test/Abnormal_Loss', avg_abnormal_loss, epoch)  
        
        # 记录平均指标（18种疾病）
        writer.add_scalar('Test/Avg_Accuracy', avg_accuracy, epoch)  
        writer.add_scalar('Test/Avg_Precision', avg_precision, epoch)  
        writer.add_scalar('Test/Avg_Recall', avg_recall, epoch)  
        writer.add_scalar('Test/Avg_F1', avg_f1, epoch)  
        writer.add_scalar('Test/Avg_AUROC', avg_auroc, epoch)  
        
        # 记录每种疾病的指标  
        for i in range(18):  
            writer.add_scalar(f'Test/Disease_{disease_names_18[i]}/Accuracy', accuracy_per_disease[i], epoch)  
            writer.add_scalar(f'Test/Disease_{disease_names_18[i]}/Precision', precision_per_disease[i], epoch)  
            writer.add_scalar(f'Test/Disease_{disease_names_18[i]}/Recall', recall_per_disease[i], epoch)  
            writer.add_scalar(f'Test/Disease_{disease_names_18[i]}/F1', f1_per_disease[i], epoch)  
            writer.add_scalar(f'Test/Disease_{disease_names_18[i]}/AUROC', auroc_per_disease[i], epoch)  

    logger.info(f"Test epoch {epoch} completed. Average Loss: {avg_loss:.6f}, Average AUROC: {avg_auroc:.6f}")

    return avg_loss