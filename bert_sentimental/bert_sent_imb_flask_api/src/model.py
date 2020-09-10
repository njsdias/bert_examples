import src.config as config
import transformers
import torch.nn as nn


class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1)  # 768 output feature, 1:binary classification

    def forward(self, ids, mask, token_type_ids):
        # get these two outputs from bert
        # out1 (last hidden states, CLS token) is the sequence of hidden states
        # for each every token
        # for all batches. For instance if you say you have 512 tokens
        # (our max length). so you have 512 vectors of size
        # 768 for each batch
        # out2 (bert poller)
        # out1, out2: we are interested only in out2
        # out2: give 738 gor each sample in the batch
        _, o2 = self.bert(ids,
                          attention_mask=mask,
                          token_type_ids=token_type_ids)

        # apply dropout to bert output
        bo = self.bert_drop(o2)

        # pass bert output to a linear layer
        output = self.out(bo)

        return output



