import faiss
import numpy as np
from tqdm import tqdm
import torch
from transformers import BertTokenizer, HfArgumentParser
# from model_src.transformer.bert.modeling_bert import BertModel
import sys
sys.path.append("/home/yaoxingzhi1/knn")
from model_src import DualTowerBert as MyModel
from component import DualBertDataArguments
import argparse
from loguru import logger
from safetensors.torch import load_file
import json

def setup_everything():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_args_file", type=str, default='/home/yaoxingzhi1/knn/args/train_dual_tower.json', help="")
    parser.add_argument("--local_rank", type=int, help="")
    args = parser.parse_args()
    train_args_file = args.train_args_file
    parser = HfArgumentParser((DualBertDataArguments))
    data_config = parser.parse_json_file(train_args_file, allow_extra_keys=True)[0]
    feature_config = data_config.FeatureConfig 
    query_model_config = data_config.query_model_config
    doc_model_config = data_config.doc_model_config
    logger.info(data_config)
    return data_config, feature_config, query_model_config, doc_model_config

model_name_or_path = "/home/yaoxingzhi1/JD_Young/bert-knn-model-list/knn_model_list/dual_bert/model.safetensors"
tokenizer = BertTokenizer("/home/yaoxingzhi1/knn/component/mytokenizers/bert_tokenizer/vocab.txt")
data_config, feature_config, query_model_config, doc_model_config = setup_everything()
model = MyModel(query_model_config=query_model_config, doc_model_config=doc_model_config)

ckpt = load_file(model_name_or_path)
missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
logger.info(f">>>>>>>>> missing_keys: {len(missing_keys)}, unexpected_keys:\
    {len(unexpected_keys)}")
logger.info(missing_keys)
logger.info(unexpected_keys)

def load_corpus(corpus_path, sep_token, verbose=True):
    corpus = {}
    for line in tqdm(open(corpus_path), mininterval=10, disable=not verbose):
        splits = line.strip().split("\t")
        corpus_id, text_fields = splits[0], splits[1:]
        text = f'{sep_token}'.join((t.strip() for t in text_fields))
        corpus[corpus_id] = text[:10000]
    return corpus

def encode_corpus(corpus):
    return tokenizer(corpus, padding=True, truncation=True, return_tensors="pt")

def encode_query(query):
    return tokenizer(query, padding=True, truncation=True, return_tensors="pt")

def generate_corpus_embeddings(corpus_path, batch_size):
    corpus = load_corpus(corpus_path=corpus_path, sep_token=tokenizer.sep_token)
    model.eval()
    corpus_ids = list(corpus.keys())
    corpus_item = list(corpus.values())
    corpus_embeddings = {}
    for i in tqdm(range(0, len(corpus_item), batch_size)):

        batch_corpus = corpus_item[i: i+batch_size]
        encode_input = encode_corpus(batch_corpus)

        with torch.no_grad():
            corpus_embeddings = model.doc_bert(**encode_input)
            embeddings = corpus_embeddings.pooler_output

            for corpus, embedding in zip(batch_corpus, embeddings):
                corpus_embeddings[corpus] = embedding.cpu().numpy()

    return corpus_ids, corpus_embeddings

# def generate_query_embeddings(queries_path, tokenizer, batch_size):
#     model.eval()

def build_corpus_index(corpus_path, output_dir):
    corpus_ids, corpus_embeddings_dict = generate_corpus_embeddings(corpus_path, batch_size=1024)
    paragraphs = list(corpus_embeddings_dict.keys())
    paragraph_dict = {idx: para for idx,para in enumerate(paragraphs)}

    with open(output_dir, "w", encoding="utf-8") as file:
        json.dump(paragraph_dict, file, ensure_ascii=False, indent=4)
        

    dim = 768
    nlist = 1000
    m = 48
    bits = 8
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, bits)

    corpus_embeddings = np.array(list(corpus_embeddings_dict.values()))
    index.train(corpus_embeddings)
    index.add_with_ids(corpus_embeddings, corpus_ids)

    faiss.write_index(index, "/home/yaoxingzhi1/knn/index/faiss_index.ivfpq")


def search(query, index, k):
    encode_input = encode_query(query)
    query_embedding = model.query_bert(**encode_input)
    distance, retrieval_ids = index.search(query_embedding, k)

    return retrieval_ids

def main():
    corpus_path = "/home/yaoxingzhi1/knn/dataset/collection.tsv"
    output_dir = "/home/yaoxingzhi1/knn/dataset/idx2paragraph.json"
    build_corpus_index(corpus_path, output_dir)

if __name__ == "__main__":
    main()