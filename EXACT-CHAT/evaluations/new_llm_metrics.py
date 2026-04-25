import argparse
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
import pandas as pd
import tqdm
import json


def compute_scores(gts, res):
    """
    Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)

    :param gts: Dictionary with the image ids and their gold captions,
    :param res: Dictionary with the image ids ant their generated captions
    :print: Evaluation score (the mean of the scores of all the instances) for each measure
    """
    scorers = [
        (Bleu(4), ["BLEU_1"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "Cider"),
       # (Spice(), "spice")
    ]
    eval_res = {}
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


def evaluate(ground_truth_path, predictions_path):
    with open(ground_truth_path, 'r') as file1:
        ground_truth = json.load(file1)

    with open(predictions_path, 'r') as file2:
        outputs = json.load(file2)

    gts = {}
    recs = {}
    counter = 0

    for i in tqdm.tqdm(range(len(ground_truth))):
        conv_ground = ground_truth[i]["conversations"]
        conv_out = outputs[i]["conversations_out"]
        for k in range(len(conv_ground)):
            if conv_ground[k]["from"] == "human":
                try:
                    if conv_ground[k+1]["from"] == "gpt":
                        question = conv_ground[k]["value"]
                        type = conv_ground[k]["type"]
                        if type == "free_response" or type == "description" or type == "conversation":
                            type_print = "Long answer"
                        elif type == "multiple_choice":
                            type_print = "Multiple choice"
                        elif type == "report_generation":
                            type_print = "Report generation"
                        else:
                            type_print = "Short answer"

                        if type_print == "Multiple choice":
                            if "\n" not in conv_ground[k+1]["value"]:
                                real_answer = conv_ground[k+1]["value"]
                                output = conv_out[k//2]["answer"]
                                output = output.replace("<s>","").replace("</s>","").replace("<|eot_id|>","").replace("\n","")

                                gts[counter] = [real_answer]
                                recs[counter] = [output]
                                counter += 1
                except:
                    pass

    print(compute_scores(gts, recs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute NLP metrics (BLEU/METEOR/ROUGE/CIDEr) for multiple-choice questions.')
    parser.add_argument('ground_truth_path', type=str, help='Path to the ground truth VQA JSON file.')
    parser.add_argument('predictions_path', type=str, help='Path to the model predictions JSON file.')
    args = parser.parse_args()
    evaluate(args.ground_truth_path, args.predictions_path)