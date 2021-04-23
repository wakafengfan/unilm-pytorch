import torch
from transformers import AutoConfig

from bojone_snippets import AutoRegressiveDecoder, sequence_padding
from bojone_tokenizers import load_vocab, Tokenizer
from configuration.config import *
from model_unilm import BojoneModelWithPooler

maxlen=32

dict_path = simbert_pt_path / "vocab.txt"
config_path = simbert_pt_path / "bert_config.json"
checkpoint_path = simbert_pt_path / "pytorch_model.bin"

# 加载并精简词表，建立分词器
token_dict = load_vocab(
    dict_path=dict_path,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)


config = AutoConfig.from_pretrained(pretrained_model_name_or_path=config_path)
model = BojoneModelWithPooler(config)
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


class SynonymsGenerator(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps('probas')
    def predict(self, inputs, output_ids, step):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        token_ids = torch.tensor(token_ids, dtype=torch.long, device=device)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long, device=device)
        with torch.no_grad():
            o = model(token_ids, segment_ids, output_type="seq2seq")[:, -1]
            o = torch.softmax(o, dim=-1)
            o = o.cpu().detach().numpy()
        return o

    def generate(self, text, n=1, topk=5):
        token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
        output_ids = self.random_sample([token_ids, segment_ids], n, topk)  # 基于随机采样
        return [tokenizer.decode(ids) for ids in output_ids]


generator = SynonymsGenerator(start_id=None, end_id=tokenizer._token_end_id, maxlen=maxlen)


def gen_synonyms(text, n=100, k=20):
    r = generator.generate(text, n=n)
    r = [_ for _ in r if _!=text]
    r = [text] + r
    X, S = [], []
    for s in r:
        token_ids, segment_ids = tokenizer.encode(s, maxlen=maxlen)
        X.append(token_ids)
        S.append(segment_ids)
    X = torch.tensor(sequence_padding(X), dtype=torch.long, device=device)
    S = torch.tensor(sequence_padding(S), dtype=torch.long, device=device)
    Z = model(X, S)  # [b,768]
    Z = Z.cpu().detach().numpy()
    Z /= (Z**2).sum(axis=1, keepdims=True)**0.5
    argsort = np.dot(Z[1:], -Z[0]).argsort()
    return set([r[i+1] for i in argsort[:k]])


print(gen_synonyms(text="投保需要哪些材料"))












