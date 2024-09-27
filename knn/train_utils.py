import json
import random
import pandas as pd
import yaml


def read_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file_path)
    return data

def load_config(config_file):
    with open(config_file, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config     

def read_excel(file_path):
    df = pd.read_excel(file_path)
    return df

def read_csv(file_path):
    df = pd.read_csv(file_path)
    return df


def load_attribute_mapping(file_path="/home/yaoxingzhi1/JD_Young/LLM_for_QP/Baseline/train/data/idx2name/attr_dict.txt"):

    with open(file_path, "r", encoding="utf-8") as file:
        id_attribute_mapping = {}
        for line in file:
            parts = line.split("\t")
            id_attribute_mapping[parts[0]] = parts[1]

    return id_attribute_mapping

def load_brand_mapping(file_path="/home/yaoxingzhi1/JD_Young/LLM_for_QP/SFT_Brand_LLM/train/data/idx2name/brand_dict_in.txt"):

    with open(file_path, "r", encoding="utf-8") as file:
        id_brand_mapping = {}
        for line in file:
            parts = line.rstrip().split("\t")
            id_brand_mapping[parts[0]] = parts[1]
    
    return id_brand_mapping

def load_front_category_mapping(file_path="/home/yaoxingzhi1/JD_Young/LLM_for_QP/Baseline/train/data/idx2name/front_cid_dict.txt"):
    with open(file_path, "r", encoding="utf-8") as file:
        # skip first line
        next(file)

        id_category_mapping = {}
        # load line
        for line in file:

            parts = line.split('\t')
            if len(parts)!=9:
                print(parts)
            # 匹配一、二、三级类目
            third_cid, third_category, second_cid, second_category, first_cid, first_category = parts[:6]
            id_category_mapping[third_cid] = third_category
            # id_category_mapping[second_cid] = second_category
            # id_category_mapping[first_cid] = first_category

        # add -1 meaning no category
        id_category_mapping["-1"] = "无"
    return id_category_mapping


def detect_delimiter(file_path):
    delimiters = [',', ';', '\t', '|']
    for delimiter in delimiters:
        try:
            df = pd.read_csv(file_path, delimiter=delimiter, nrows=5)
            print(f"Trying delimiter '{delimiter}':")
            print(df.head())
            return delimiter
        except Exception as e:
            print(f"Delimiter '{delimiter}' failed: {e}")
    return None

def count_lines(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        return len(lines)

def process_line(line):
    parts = line['text'].split('\t')
    if len(parts) == 10:  # 确保每行有正确数量的字段
        return {
            'query': parts[0],
            'contain_brand_intent': parts[1],
            'brand_pred': parts[2],
            'term_list': parts[3],
            'label_list': parts[4],
            'click_brand_ids': parts[5],
            'p_attr_ids': parts[6],
            'n_attr_ids': parts[7],
            'fcid_click_list': parts[8],
            'fcid_expo_list': parts[9]
        }
    else:
        return {}
    
def process_old_line(line):
    parts = line['text'].split('\t')
    if len(parts) == 8:  # 确保每行有正确数量的字段
        return {
            'query': parts[0],
            'term_list': parts[1],
            'label_list': parts[2],
            'click_brand_ids': parts[3],
            'p_attr_ids': parts[4],
            'n_attr_ids': parts[5],
            'fcid_click_list': parts[6],
            'fcid_expo_list': parts[7]
        }
    else:
        return {}

def process_category_line(line):
    parts = line.rstrip().split("\t")
    if len(parts) == 4:
        return {
            'query': parts[0],
            'term_list': parts[1],
            'label_list': parts[2],
            'bert_knn': parts[3]
        }
    
def process_brand_line(line):
    parts = line.rstrip().split("\t")
    if len(parts) == 5:
        return {
            'query': parts[0],
            'term_list': parts[1],
            'label_id': parts[2],
            'label_list': parts[3],
            'bert_knn': parts[4]
        }
