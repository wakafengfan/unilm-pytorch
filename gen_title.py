import torch
import torch.nn as nn
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge import Rouge
from transformers import AutoConfig

from bojone_snippets import AutoRegressiveDecoder
from bojone_snippets import DataGenerator, sequence_padding
from bojone_tokenizers import load_vocab, Tokenizer
from configuration.config import *
from model_unilm import BojoneModel
from optimizer import create_optimizer_and_scheduler

maxlen = 256
batch_size = 16
epochs = 40

max_grad_norm = 5.0

# 加载数据集
def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            title, content = l.strip().split('\t')
            D.append((title, content))
    return D

data_root_path = common_data_path / "open_dataset" / "csl"
train_data = load_data(str(data_root_path/'train.tsv'))
valid_data = load_data(str(data_root_path/'val.tsv'))


# 加载并精简词表，建立分词器
dict_path = str(bert_wwm_pt_path / "vocab.txt")
config_path = str(bert_wwm_pt_path / "config.json")
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        for is_end, (title, content) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(
                content, title, maxlen=maxlen
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)

            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long)
                batch_segment_ids = torch.tensor(sequence_padding(batch_segment_ids), dtype=torch.long)
                yield batch_token_ids, batch_segment_ids
                batch_token_ids, batch_segment_ids = [], []

train_generator = data_generator(train_data, batch_size)


loss_func = nn.CrossEntropyLoss(reduction="none")
def compute_loss(inputs):
    y_true, y_mask, y_pred = inputs
    y_true = y_true[:, 1:].contiguous()  # 目标token_ids
    y_mask = y_mask[:, 1:].contiguous()  # segment_ids，刚好指示了要预测的部分
    y_pred = y_pred[:, :-1, :].contiguous()  # 预测序列，错开一位
    loss = loss_func(y_pred.view(-1, y_pred.size(-1)), y_true.view(-1))
    loss = loss.view(y_pred.size(0), -1)
    loss = torch.sum(loss * y_mask) / torch.sum(y_mask)
    return loss


# train
config = AutoConfig.from_pretrained(pretrained_model_name_or_path=config_path, keep_tokens=keep_tokens, vocab_size=len(keep_tokens))
tmp_state_dict = torch.load(bert_wwm_pt_path/"pytorch_model.bin", map_location="cpu")
tmp_state_dict['bert.embeddings.word_embeddings.weight'] = torch.index_select(tmp_state_dict['bert.embeddings.word_embeddings.weight'], 0, torch.tensor(keep_tokens, dtype=torch.long))
tmp_state_dict["bert.cls.transform.dense.weight"] = tmp_state_dict["cls.predictions.transform.dense.weight"]
tmp_state_dict["bert.cls.transform.dense.bias"] = tmp_state_dict["cls.predictions.transform.dense.bias"]
tmp_state_dict["bert.cls.transform.LayerNorm.weight"] = tmp_state_dict["cls.predictions.transform.LayerNorm.weight"]
tmp_state_dict["bert.cls.transform.LayerNorm.bias"] = tmp_state_dict["cls.predictions.transform.LayerNorm.bias"]

model = BojoneModel.from_pretrained(pretrained_model_name_or_path=bert_wwm_pt_path, config=config, state_dict=tmp_state_dict)
optimizer, scheduler = create_optimizer_and_scheduler(model, lr=1e-5, num_training_steps=train_generator.steps * epochs)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# generate
class BojoneAutoTitle(AutoRegressiveDecoder):

    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, state=None):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], axis=1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], axis=1)

        token_ids = torch.tensor(token_ids, dtype=torch.long).to(device)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long).to(device)

        with torch.no_grad():
            y_pred = model(token_ids, segment_ids)
        y_pred = torch.softmax(y_pred[:,-1,:].squeeze(1), dim=-1)
        y_pred = y_pred.cpu().detach().numpy()

        return y_pred

    def generate(self, text):
        token_ids, segment_ids, = tokenizer.encode(text, maxlen=maxlen)
        output_ids = self.beam_search([token_ids, segment_ids], topk=1)
        return tokenizer.decode(output_ids)


autotitle = BojoneAutoTitle(start_id=None, end_id=tokenizer._token_end_id, maxlen=32)

rouge = Rouge()
smooth = SmoothingFunction().method1
best_bleu = 0.
def predict_and_evaluate():
    total = 0
    rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0
    for title, content in valid_data:
        total += 1
        title = ' '.join(title).lower()
        pred_title = ' '.join(autotitle.generate(content)).lower()
        if pred_title.strip():
            scores = rouge.get_scores(hyps=pred_title, refs=title)
            rouge_1 += scores[0]['rouge-1']['f']
            rouge_2 += scores[0]['rouge-2']['f']
            rouge_l += scores[0]['rouge-l']['f']
            bleu += sentence_bleu(
                references=[title.split(' ')],
                hypothesis=pred_title.split(' '),
                smoothing_function=smooth
            )
    rouge_1 /= total
    rouge_2 /= total
    rouge_l /= total
    bleu /= total
    return {
        'rouge-1': rouge_1,
        'rouge-2': rouge_2,
        'rouge-l': rouge_l,
        'bleu': bleu,
    }

model.zero_grad()
for e in range(epochs):
    model.train()
    for step, batch in enumerate(train_generator):
        batch = [_.to(device) for _ in batch]
        input_ids, segment_ids = batch
        y_pred = model(input_ids, segment_ids)
        loss = compute_loss((input_ids, segment_ids, y_pred))

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if step % 100 == 0 and step != 0:
            logger.info(f"epoch: {e} - step: {step} - loss: {loss}")

    model.eval()
    eval_dic = predict_and_evaluate()
    logger.info(eval_dic)










