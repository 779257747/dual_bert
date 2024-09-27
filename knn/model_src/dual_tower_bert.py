import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig
from transformers.modeling_outputs import SequenceClassifierOutput 
from .transformer.bert.modeling_bert import BertModel

class DualTowerBert(nn.Module):
    def __init__(self, query_model_config, doc_model_config):
        super(DualTowerBert, self).__init__()
        self.query_bert = BertModel(BertConfig(**query_model_config))
        self.doc_bert = BertModel(BertConfig(**doc_model_config))
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, query, query_mask,
                positive, positive_mask,
                negative, negative_mask):
        batch_size = query.size(0)
        
        query_output = self.query_bert(input_ids=query, attention_mask=query_mask)
        query_embedding = query_output.pooler_output

        positive_output = self.doc_bert(input_ids=positive, attention_mask=positive_mask)
        positive_embedding = positive_output.pooler_output

        negative_output = self.doc_bert(input_ids=negative, attention_mask=negative_mask)
        negative_embedding = negative_output.pooler_output

        query_embeddings = F.normalize(query_embedding, p=2, dim=1)
        positive_embeddings = F.normalize(positive_embedding, p=2, dim=1)
        negative_embeddings = F.normalize(negative_embedding, p=2, dim=1)

        positive_cosine_sim = torch.matmul(query_embeddings, positive_embeddings.t())
        negative_cosine_sim = torch.matmul(query_embeddings, negative_embeddings.t())
        cosine_sim = torch.cat((positive_cosine_sim, negative_cosine_sim), dim=1)

        device = next(self.query_bert.parameters()).device
        positive_targets = torch.eye(batch_size, device=device)
        negative_targets = torch.zeros((batch_size, batch_size), device=device)
        target_matrix = torch.cat((positive_targets, negative_targets), dim=1)
        
        loss = self.loss_fn(cosine_sim, target_matrix)

        return SequenceClassifierOutput(
            loss=loss,
            logits=cosine_sim,
        )

        