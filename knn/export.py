import torch
from model_src.transformer import BertModel
from transformers import BertTokenizer
from loguru import logger
import argparse
from transformers import HfArgumentParser
from component import DataArguments
from model_src import ExportBertModel
from safetensors.torch import load_file

model_path = "/home/yaoxingzhi1/JD_Young/bert-knn-model-list/knn_model_list/dual_bert/model.safetensors"
tokenizer = BertTokenizer("/home/yaoxingzhi1/JD_Young/fe-qp-models/component/mytokenizers/bert_tokenizer/vocab.txt")

def setup_everything():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_args_file", type=str, default='/home/yaoxingzhi1/knn/args/train_dual_tower.json', help="")
    parser.add_argument("--local_rank", type=int, help="")
    args = parser.parse_args()
    train_args_file = args.train_args_file
    parser = HfArgumentParser((DataArguments))
    data_config = parser.parse_json_file(train_args_file, allow_extra_keys=True)[0]
    feature_config = data_config.FeatureConfig 
    query_model_config = data_config.query_model_config
    doc_model_config = data_config.doc_model_config
    logger.info(data_config)
    return data_config, feature_config, query_model_config, doc_model_config

data_config, feature_config, query_model_config, doc_model_config = setup_everything()

model = ExportBertModel(query_model_config, doc_model_config)
ckpt = load_file(model_path)
missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
logger.info(f">>>>>>>>> missing_keys: {len(missing_keys)}, unexpected_keys:\
    {len(unexpected_keys)}")
logger.info(missing_keys)
logger.info(unexpected_keys)
model.eval()

text1 = "treating tension headaches without medication"
text2 = "Ways to Treat Your Headaches Without Drugs You may not need a doctor's prescription to treat your headaches. You can often find relief in other ways, without medication: Apply an ice pack to the painful part of your head. Try placing it on your forehead, temples, or the back of your neck. Be sure you wrap it in a cloth first to protect your skin."

encoded_input1 = tokenizer(text1, return_tensors='pt', padding=True, truncation=True, max_length=512)
encoded_input2 = tokenizer(text2, return_tensors='pt', padding=True, truncation=True, max_length=512)

input_ids1 = encoded_input1["input_ids"]
attention_mask1 = encoded_input1["attention_mask"]

input_ids2 = encoded_input2["input_ids"]
attention_mask2 = encoded_input2["attention_mask"]

cosine_sim = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
print(cosine_sim)