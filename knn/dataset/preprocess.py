import json
import random

file_path = "/home/yaoxingzhi1/knn/dataset/train_format.jsonl"
outfile_path = "/home/yaoxingzhi1/knn/dataset/train_format_shuffle.jsonl"

# data = []
# with open(file_path, "r", encoding="utf-8") as file:
#     for line in file:
#         data_dict = json.loads(line)
#         query = data_dict["query"]

#         positive_passages = data_dict["positive_passages"]
#         positive_texts = [positive_text["text"] for positive_text in positive_passages]
#         negative_passages = data_dict["negative_passages"]
#         negative_texts = [negative_text["text"] for negative_text in negative_passages]

#         for positive in positive_texts:
#             new_data = {
#                 "query": query,
#                 "positive": positive,
#                 "negative": random.sample(negative_texts, 1)[0]
#             }

#             data.append(new_data)

# with open(outfile_path, "w", encoding="utf-8") as file:
#     for item in data:
#         file.write(json.dumps(item, ensure_ascii=False) + "\n")
data = []
with open(file_path, 'r', encoding="utf-8") as file:
    for line in file:
        data.append(json.loads(line))

random.shuffle(data)

with open(outfile_path, "w", encoding='utf-8') as outfile:
    for item in data:
        outfile.write(json.dumps(item, ensure_ascii=False) + "\n")