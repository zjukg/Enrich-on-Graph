
# This script is used to evaluate metric results of predictions
import json
import re 
import string
from sklearn.metrics import precision_score
from datasets import load_dataset

def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    # Remove all special symbols "!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~"
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # remove <pad> token:
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s


def match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1

def eval_acc(prediction, answer):
    matched = 0.
    for a in answer:
        if match(prediction, a):
            matched += 1
    if len(answer) == 0:
        return 1
    else:
        return matched / len(answer)

def eval_hit(prediction, answer):
    for a in answer:
        if match(prediction, a):
            return 1
    return 0

def eval_f1(prediction, answer):
    if len(prediction) == 0:
        return 0, 0, 0
    matched = 0
    prediction_str = ' '.join(prediction)
    for a in answer:
        if match(prediction_str, a):
            matched += 1
    precision = matched / len(prediction)
    if len(answer) == 0:
        recall = 1
    else:
        recall = matched / len(answer)
    if precision + recall == 0:
        return 0, precision, recall
    else:
        return 2 * precision * recall / (precision + recall), precision, recall

def extract_topk_prediction(prediction, k=-1):
    results = {}
    for p in prediction:
        if p in results:
            results[p] += 1
        else:
            results[p] = 1
    if k > len(results) or k < 0:
        k = len(results)
    results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    return [r[0] for r in results[:k]]

def eval_result(dataset_path, cal_f1=True, topk = -1):
    dataset_name = "cwq"
    if "cwq" in dataset_path:
        dataset_name = "cwq"
    elif "webqsp" in dataset_path:
        dataset_name = "webqsp"


    # Load results
    acc_list = []
    hit_list = []
    f1_list = []
    precission_list = []
    recall_list = []

    dataset = load_dataset("parquet", data_files={'test': dataset_path})
    for data in dataset["test"]:
        id = data["id"]
        prediction = data['predictions']
        # May need to change to answer field
        answer = data['answer']
        if cal_f1:
            if not isinstance(prediction, list):
                prediction = prediction.split("\n")
            else:
                prediction = extract_topk_prediction(prediction, topk)
            f1_score, precision_score, recall_score = eval_f1(prediction, answer)
            f1_list.append(f1_score)
            precission_list.append(precision_score)
            recall_list.append(recall_score)
            prediction_str = ' '.join(prediction)
            acc = eval_acc(prediction_str, answer)
            hit = eval_hit(prediction_str, answer)
            acc_list.append(acc)
            hit_list.append(hit)
        else:
            acc = eval_acc(prediction, answer)
            hit = eval_hit(prediction, answer)
            acc_list.append(acc)
            hit_list.append(hit)
            
    if len(f1_list) > 0:
        result_str = "Accuracy: " + str(sum(acc_list) * 100 / len(acc_list)) + " Hit: " + str(sum(hit_list) * 100 / len(hit_list)) + " F1: " + str(sum(f1_list) * 100 / len(f1_list)) + " Precision: " + str(sum(precission_list) * 100 / len(precission_list)) + " Recall: " + str(sum(recall_list) * 100 / len(recall_list))
    else:
        result_str = "Accuracy: " + str(sum(acc_list) * 100 / len(acc_list)) + " Hit: " + str(sum(hit_list) * 100 / len(hit_list))
    print(result_str)

    # result_name = f"{dataset_name}_eval_result_top_{topk}.txt" if topk > 0 else f'{dataset_name}_eval_result.txt'
    # eval_result_path = "result/result_eval/" + result_name
    # with open(eval_result_path, 'w') as f:
    #     f.write(result_str)

def eval_result_main(dataset_name=None,cal_f1=True,top_k=-1):
    eval_result(dataset_name,cal_f1,top_k)

if __name__ == "__main__":
    # For test
    dataset_path = "preprocess_datasets/qa_datasets/cwq_gpt-4o-mini-2024-07-18_eog_qa_2025-09-06_17-31-22.parquet"
    eval_result(dataset_path,cal_f1=True,topk=-1)
