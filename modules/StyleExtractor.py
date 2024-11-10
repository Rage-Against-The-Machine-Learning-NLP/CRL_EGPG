import torch
import torch.nn as nn
from transformers import BertModel, RobertaModel, AlbertModel
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRANSFORMERS_CACHE='/gds/hryang/projdata11/hryang/transformers-cache'

class StyleExtractor(nn.Module):
    def __init__(self,bert_type="bert"):
        super().__init__()
        print("using: ", bert_type)
        if bert_type == "bert":
            self.model = BertModel.from_pretrained("bert-base-uncased",output_hidden_states=True,cache_dir=TRANSFORMERS_CACHE)
        elif bert_type == "roberta":
            self.model = RobertaModel.from_pretrained("roberta-base",output_hidden_states=True,cache_dir=TRANSFORMERS_CACHE)
        elif bert_type == "albert":
            self.model = AlbertModel.from_pretrained("albert-base-v2",output_hidden_states=True,cache_dir=TRANSFORMERS_CACHE)

    def forward(self,input):
        attention_mask = (input != 0).float()
        outputs = self.model(input, attention_mask=attention_mask)
        hidden_states = torch.stack(outputs[2], dim=1)
        first_hidden_states = hidden_states[:,:,0,:]
        return first_hidden_states