import torch
from transformers import BertConfig
import torch.nn.functional as F
from .transformer import BertModel

class ExportBertModel(torch.nn.Module):
    def __init__(self, query_model_config, doc_model_config):
        super(ExportBertModel, self).__init__()
        self.query_bert = BertModel(BertConfig(**query_model_config))
        self.doc_bert = BertModel(BertConfig(**doc_model_config))

    def forward(self, query, query_mask, doc, doc_mask):
        query_outputs = self.query_bert(input_ids=query, attention_mask=query_mask)
        query_pooler_output = query_outputs.pooler_output
        query_embedding = F.normalize(query_pooler_output, p=2, dim=1)

        doc_outputs = self.doc_bert(input_ids=doc, attention_mask=doc_mask)
        doc_pooler_output = doc_outputs.pooler_output
        doc_embedding = F.normalize(doc_pooler_output, p=2, dim=1)

        cosine_sim = torch.matmul(query_embedding, doc_embedding.t())
        return cosine_sim