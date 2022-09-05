import transformers
from torch import nn


class BERTurkSentimentAnalyzer(nn.Module):
    def __init__(self, class_count):
        super(BERTurkSentimentAnalyzer, self).__init__()
        self.model = transformers.AutoModel.from_pretrained("dbmdz/bert-base-turkish-cased")
        self.drop = nn.Dropout(p=0.25)
        self.output_layer = nn.Linear(self.model.config.hidden_size, class_count)
        self.sm = nn.Softmax(dim=1)
    
    def forward(self, input_ids, attention_mask):
        last_hidden_state, pooler_output = self.model(input_ids=input_ids, attention_mask=attention_mask).items()
        last_hidden_state = last_hidden_state[1]
        pooler_output = pooler_output[1]

        output = self.drop(pooler_output)
        output = self.output_layer(output)
        return self.sm(output)

class ELECTRASentimentAnalyzer(nn.Module):
    def __init__(self, class_count):
        super(ELECTRASentimentAnalyzer, self).__init__()
        self.model = transformers.AutoModel.from_pretrained("dbmdz/electra-base-turkish-cased-discriminator")
        for (name, param) in self.model.named_parameters():
          if name == "encoder.layer.11.attention.self.query.weight":
            break
          param.requires_grad = False

        self.pooler_layer = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        self.pooler_tanh = nn.Tanh()

        self.drop = nn.Dropout(p=0.25)
        self.output_layer = nn.Linear(self.model.config.hidden_size, class_count)
        self.sm = nn.Softmax(dim=1)
    
    def forward(self, input_ids, attention_mask):
        last_hidden_state = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # Here I mimick the pooler implementation of classic BERT models 
        # to be able obtain required output size
        first_token_tensor = last_hidden_state[:, 0]
        pooler_output = self.pooler_layer(first_token_tensor)
        pooler_output = self.pooler_tanh(pooler_output)

        output = self.drop(pooler_output)
        output = self.output_layer(output)
        return self.sm(output)