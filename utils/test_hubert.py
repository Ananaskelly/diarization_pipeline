import torch
from transformers import Wav2Vec2Processor, HubertModel

# processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-base-ls960")
model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
