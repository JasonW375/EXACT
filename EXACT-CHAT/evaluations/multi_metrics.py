import argparse
from pycocoevalcap.bleu.bleu import Bleu  
from pycocoevalcap.meteor.meteor import Meteor  
from pycocoevalcap.rouge.rouge import Rouge  
from pycocoevalcap.cider.cider import Cider  
from pycocoevalcap.spice.spice import Spice  
import pandas as pd  
import tqdm  
import json  
import numpy as np  

def compute_scores(gts, res):  
    """  
    Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)  

    :param gts: Dictionary with the image ids and their gold captions,  
    :param res: Dictionary with the image ids ant their generated captions  
    :print: Evaluation score (the mean of the scores of all the instances) for each measure  
    """  

    # Set up scorers  
    scorers = [  
        (Bleu(4), ["BLEU_1"]),  
        (Meteor(), "METEOR"),  
        (Rouge(), "ROUGE_L"),  
        (Cider(), "Cider"),  
       # (Spice(), "spice")  
    ]  
    eval_res = {}  
    # Compute score for each metric  
    for scorer, method in scorers:  
        try:  
            score, scores = scorer.compute_score(gts, res)  
        except TypeError:  
            score, scores = scorer.compute_score(gts, res)  
        if isinstance(method, list):  
            for sc, m in zip(score, method):  
                eval_res[m] = sc  
        else:  
            eval_res[method] = score  
    return eval_res  


def main(ground_truth_path, predictions_path):
    # Read the first JSON file into a list  
    with open(ground_truth_path, 'r') as file1:  
        ground_truth = json.load(file1)  

    # Read the second JSON file into a list   
    with open(predictions_path, 'r') as file2:  
        outputs = json.load(file2)  

    # 为每种问题类型创建单独的评估容器  
    types = ["Multiple choice", "Long answer", "Short answer", "Report generation"]  
    gts_by_type = {type_name: {} for type_name in types}  
    recs_by_type = {type_name: {} for type_name in types}  
    counter_by_type = {type_name: 0 for type_name in types}  
    accession_by_type = {type_name: [] for type_name in types}  

    # 处理所有问题  
    for i in tqdm.tqdm(range(len(ground_truth))):  
        conv_ground = ground_truth[i]["conversations"]  
        conv_out = outputs[i]["conversations_out"]  
        
        for k in range(len(conv_ground)):  
            if conv_ground[k]["from"] == "human":  
                try:  
                    if k+1 < len(conv_ground) and conv_ground[k+1]["from"] == "gpt":  
                        question = conv_ground[k]["value"]  
                        type_name = conv_ground[k]["type"]  
                        
                        # 确定问题类型  
                        if type_name == "free_response" or type_name == "description" or type_name == "conversation":  
                            type_print = "Long answer"  
                        elif type_name == "multiple_choice":  
                            type_print = "Multiple choice"  
                        elif type_name == "report_generation":  
                            type_print = "Report generation"  
                        else:  
                            type_print = "Short answer"  
                        
                        # 为多选题添加特殊处理（保留原有逻辑）  
                        if type_print == "Multiple choice" and "\n" in conv_ground[k+1]["value"]:  
                            continue  
                        
                        # 提取答案  
                        real_answer = conv_ground[k+1]["value"]  
                        output = conv_out[k//2]["answer"]  
                        output = output.replace("<s>","").replace("</s>","").replace("<|eot_id|>","")  
                        
                        # 仅为多选题移除换行符（其他题型可能需要保留格式）  
                        if type_print == "Multiple choice":  
                            output = output.replace("\n","")  
                        
                        # 获取accession number  
                        ac = ground_truth[i]["image"]  
                        ac_list = ac.split("_")  
                        accessionno = ac_list[0] + "_" + ac_list[1] + "_" + ac_list[2]  
                        
                        # 存储答案到对应类型容器  
                        current_counter = counter_by_type[type_print]  
                        gts_by_type[type_print][current_counter] = [real_answer]  
                        recs_by_type[type_print][current_counter] = [output]  
                        accession_by_type[type_print].append(accessionno)  
                        counter_by_type[type_print] += 1  
                except Exception as e:  
                    pass  

    # 输出各类型问题的统计和评估结果  
    print("\n" + "="*80)  
    print("评估结果汇总")  
    print("="*80)  

    for type_name in types:  
        count = counter_by_type[type_name]  
        if count > 0:  
            print(f"\n## {type_name} (共 {count} 个问题)")  
            scores = compute_scores(gts_by_type[type_name], recs_by_type[type_name])  
            print(f"BLEU-1: {scores.get('BLEU_1', 'N/A'):.4f}")  
            print(f"METEOR: {scores.get('METEOR', 'N/A'):.4f}")  
            print(f"ROUGE-L: {float(scores.get('ROUGE_L', 'N/A')):.4f}")  
            print(f"CIDEr: {float(scores.get('Cider', 'N/A')):.4f}")  
        else:  
            print(f"\n## {type_name}: 未找到有效问题")  

    # 输出总体评估结果  
    if sum(counter_by_type.values()) > 0:  
        print("\n## 所有问题类型综合评估")  
        all_gts = {}  
        all_recs = {}  
        counter = 0  
        
        for type_name in types:  
            for idx in gts_by_type[type_name]:  
                all_gts[counter] = gts_by_type[type_name][idx]  
                all_recs[counter] = recs_by_type[type_name][idx]  
                counter += 1  
        
        all_scores = compute_scores(all_gts, all_recs)  
        print(f"BLEU-1: {all_scores.get('BLEU_1', 'N/A'):.4f}")  
        print(f"METEOR: {all_scores.get('METEOR', 'N/A'):.4f}")  
        print(f"ROUGE-L: {float(all_scores.get('ROUGE_L', 'N/A')):.4f}")  
        print(f"CIDEr: {float(all_scores.get('Cider', 'N/A')):.4f}")  


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute NLP metrics (BLEU/METEOR/ROUGE/CIDEr) broken down by question type.')
    parser.add_argument('ground_truth_path', type=str, help='Path to the ground truth VQA JSON file.')
    parser.add_argument('predictions_path', type=str, help='Path to the model predictions JSON file.')
    args = parser.parse_args()
    main(args.ground_truth_path, args.predictions_path)