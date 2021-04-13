from model import BertMultiTaskModel
from utils import load_label,clean_text
from transformers import BertTokenizer
from sklearn.metrics import classification_report
import torch
import pandas as pd
from tqdm import tqdm

task_num_classes = {'0': 3, '1': 9}
task_id_to_name = {'0': 'short_three', '1': 'short_nine'}
task_name_to_id = {value:key for key,value in task_id_to_name.items()}


model_name_or_path = "data/checkpoint/checkpoint-1-17960-0.7437"
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = BertMultiTaskModel.from_pretrained(model_name_or_path, task_num_classes=task_num_classes,model_path=model_name_or_path)


def one_sentence_pred(task_id:str,text:str)->int:
    output_ids = tokenizer.encode_plus(text, add_special_tokens=True, return_token_type_ids=True,return_attention_mask=True, return_tensors="pt")
    output_ids["task_id"] =task_id

    out = model(**output_ids)
    probs = torch.nn.functional.softmax(out[0][0].detach(), dim=-1)
    pred_labels = probs.detach().argmax(dim=-1).item()
    return pred_labels


def file_predict(task_id:str,input_file:str,label_file:str,output_file:str,output_result:str):
    label2id, id2label = load_label(label_file)
    df = pd.read_csv(input_file, sep='\t', header=None)
    y_true = []
    y_pred = []

    sample_pred_result = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f'Predict {task_id_to_name[task_id]}'):
        text = clean_text(str(row[1]))
        pred_label_id = one_sentence_pred(task_id,text[:510])
        pred_label_str = id2label[pred_label_id]

        sample_pred_result.append([
            str(row[0]),
            str(row[1]),
            str(row[2]),
            pred_label_str,
            1 if pred_label_str == str(row[2]) else 0
        ])

        y_true.append(label2id[str(row[2])])
        y_pred.append(pred_label_id)

    df_result = pd.DataFrame(data=sample_pred_result,columns=["id","text","label","pred","diss"])
    df_result.to_excel(output_file,index=False)

    f1 = classification_report(y_true,y_pred,target_names=list(label2id.keys()))
    with open(output_result, encoding="utf-8", mode="w") as f:
        f.write(f1)
    print(f1)


def three():
    task_id = "0"
    input_file = "data/original_data/short_three_eval0428.csv"
    label_file = "data/original_data/short_three_label.txt"
    output_file = "data/predict/short_three_eval0428_pred2.xlsx"
    output_result = "data/predict/short_three_eval0428_result.txt"

    file_predict(task_id,input_file,label_file,output_file,output_result)


def nine():
    task_id = "1"
    input_file = "data/original_data/short_nine_eval0317.csv"
    label_file = "data/original_data/short_nine_label.txt"
    output_file = "data/predict/short_nine_eval0317_pred.xlsx"
    output_result = "data/predict/short_nine_eval0317_result.txt"

    file_predict(task_id,input_file,label_file,output_file,output_result)


if __name__ == '__main__':
    nine()
    # three()







