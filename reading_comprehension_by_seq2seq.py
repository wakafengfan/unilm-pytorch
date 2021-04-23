import re

import torch
import torch.nn as nn
from transformers import AutoConfig

from bojone_snippets import DataGenerator, AutoRegressiveDecoder, sequence_padding
from bojone_tokenizers import load_vocab, Tokenizer
from configuration.config import *
from model_unilm import BojoneModel
from optimizer import create_optimizer_and_scheduler
from evaluation.evaluation_sogouQA import is_exact_match_answer, load_qid_answer_expand, \
    cacu_character_level_f

max_p_len = 128
max_q_len = 64
max_a_len = 32
max_qa_len = max_q_len + max_a_len
batch_size = 16
epochs = 8

gradient_accumulation_steps = 1
max_grad_norm = 5.0

config_path = str(bert_wwm_pt_path / "config.json")
dict_path = str(bert_wwm_pt_path / "vocab.txt")

# 标注数据
webqa_data = json.load((mrc_datset_path/"WebQA.json").open())
sogou_data = json.load((mrc_datset_path / 'SogouQA.json').open())

# 保存一个随机序（供划分valid用）
if not os.path.exists('random_order.json'):
    random_order = list(range(len(sogou_data)))
    np.random.shuffle(random_order)
    json.dump(random_order, open('random_order.json', 'w'), indent=4)
else:
    random_order = json.load(open('random_order.json'))

# 划分valid
train_data = [sogou_data[j] for i, j in enumerate(random_order) if i % 3 != 0]
valid_data = [sogou_data[j] for i, j in enumerate(random_order) if i % 3 == 0]
train_data.extend(train_data)
train_data.extend(webqa_data)  # 将SogouQA和WebQA按2:1的比例混合

# 加载并精简词表，建立分词器
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
        """单条样本格式：[CLS]篇章[SEP]问题[SEP]答案[SEP]
        """
        batch_token_ids, batch_segment_ids = [], []
        for is_end, D in self.sample(random):
            question = D['question']
            answers = [p['answer'] for p in D['passages'] if p['answer']]
            passage = np.random.choice(D['passages'])['passage']
            passage = re.sub(u' |、|；|，', ',', passage)
            final_answer = ''
            for answer in answers:
                if all([
                    a in passage[:max_p_len - 2] for a in answer.split(' ')
                ]):
                    final_answer = answer.replace(' ', ',')
                    break
            qa_token_ids, qa_segment_ids = tokenizer.encode(
                question, final_answer, maxlen=max_qa_len + 1
            )
            p_token_ids, p_segment_ids = tokenizer.encode(
                passage, maxlen=max_p_len
            )
            token_ids = p_token_ids + qa_token_ids[1:]
            segment_ids = p_segment_ids + qa_segment_ids[1:]
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
                batch_segment_ids = torch.tensor(sequence_padding(batch_segment_ids), dtype=torch.long, device=device)
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
tmp_state_dict["bert.cls.transform.dense.bias"]   = tmp_state_dict["cls.predictions.transform.dense.bias"]
tmp_state_dict["bert.cls.transform.LayerNorm.weight"] = tmp_state_dict["cls.predictions.transform.LayerNorm.weight"]
tmp_state_dict["bert.cls.transform.LayerNorm.bias"]   = tmp_state_dict["cls.predictions.transform.LayerNorm.bias"]


model = BojoneModel.from_pretrained(pretrained_model_name_or_path=bert_wwm_pt_path, config=config, state_dict=tmp_state_dict)
optimizer, scheduler = create_optimizer_and_scheduler(model, lr=1e-5, num_training_steps=train_generator.steps * epochs)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

class ReadingComprehension(AutoRegressiveDecoder):
    """beam search解码来生成答案
    passages为多篇章组成的list，从多篇文章中自动决策出最优的答案，
    如果没答案，则返回空字符串。
    mode是extractive时，按照抽取式执行，即答案必须是原篇章的一个片段。
    """
    def __init__(self, mode='extractive', **kwargs):
        super(ReadingComprehension, self).__init__(**kwargs)
        self.mode = mode

    def get_ngram_set(self, x, n):
        """生成ngram合集，返回结果格式是:
        {(n-1)-gram: set([n-gram的第n个字集合])}
        """
        result = {}
        for i in range(len(x) - n + 1):
            k = tuple(x[i:i + n])
            if k[:-1] not in result:
                result[k[:-1]] = set()
            result[k[:-1]].add(k[-1])
        return result

    @AutoRegressiveDecoder.wraps(default_rtype='probas', use_states=True)
    def predict(self, inputs, output_ids, states):
        inputs = [i for i in inputs if i[0, 0] > -1]  # 过滤掉无答案篇章
        topk = len(inputs[0])
        all_token_ids, all_segment_ids = [], []
        for token_ids in inputs:  # inputs里每个元素都代表一个篇章
            token_ids = np.concatenate([token_ids, output_ids], 1)
            segment_ids = np.zeros_like(token_ids)
            if states > 0:
                segment_ids[:, -output_ids.shape[1]:] = 1
            all_token_ids.extend(token_ids)
            all_segment_ids.extend(segment_ids)
        padded_all_token_ids = torch.tensor(sequence_padding(all_token_ids), dtype=torch.long, device=device)
        padded_all_segment_ids = torch.tensor(sequence_padding(all_segment_ids), dtype=torch.long, device=device)
        with torch.no_grad():
            probas = model(padded_all_token_ids, padded_all_segment_ids)
            probas = torch.softmax(probas, dim=-1)
        probas = probas.cpu().detach().numpy()
        probas = [
            probas[i, len(ids) - 1] for i, ids in enumerate(all_token_ids)
        ]
        probas = np.array(probas).reshape((len(inputs), topk, -1))
        if states == 0:
            # 这一步主要是排除没有答案的篇章
            # 如果一开始最大值就为end_id，那说明该篇章没有答案
            argmax = probas[:, 0].argmax(axis=1)
            available_idxs = np.where(argmax != self.end_id)[0]
            if len(available_idxs) == 0:  # 所有篇章最大值都是end_id
                scores = np.zeros_like(probas[0])
                scores[:, self.end_id] = 1
                return scores, states + 1
            else:
                for i in np.where(argmax == self.end_id)[0]:
                    inputs[i][:, 0] = -1  # 无答案篇章首位标记为-1
                probas = probas[available_idxs]
                inputs = [i for i in inputs if i[0, 0] > -1]  # 过滤掉无答案篇章
        if self.mode == 'extractive':
            # 如果是抽取式，那么答案必须是篇章的一个片段
            # 那么将非篇章片段的概率值全部置0
            new_probas = np.zeros_like(probas)
            ngrams = {}
            for token_ids in inputs:
                token_ids = token_ids[0]  # [1,s] -> [s]
                sep_idx = np.where(token_ids == tokenizer._token_end_id)[0][0]
                p_token_ids = token_ids[1:sep_idx]
                for k, v in self.get_ngram_set(p_token_ids, states + 1).items():
                    ngrams[k] = ngrams.get(k, set()) | v
            for i, ids in enumerate(output_ids):
                available_idxs = ngrams.get(tuple(ids), set())
                available_idxs.add(tokenizer._token_end_id)
                available_idxs = list(available_idxs)
                new_probas[:, i, available_idxs] = probas[:, i, available_idxs]
            probas = new_probas
        return (probas**2).sum(0) / (probas.sum(0) + 1), states + 1  # 某种平均投票方式

    def answer(self, question, passages, topk=1):
        token_ids = []
        for passage in passages:
            passage = re.sub(u' |、|；|，', ',', passage)
            p_token_ids = tokenizer.encode(passage, maxlen=max_p_len)[0]
            q_token_ids = tokenizer.encode(question, maxlen=max_q_len + 1)[0]
            token_ids.append(p_token_ids + q_token_ids[1:])
        output_ids = self.beam_search(
            token_ids, topk=topk, states=0
        )  # 基于beam search
        return tokenizer.decode(output_ids)


reader = ReadingComprehension(
    start_id=None,
    end_id=tokenizer._token_end_id,
    maxlen=max_a_len,
    mode='extractive'
)

def eval_valid_data(tmp_valid_data, e_num):
    output_dict = []
    load_qid_answer_expand(str(mrc_datset_path / "qid_answer_expand"))
    total = 0
    right = 0
    sum_f = 0.0
    model.eval()
    for i, d in enumerate(iter(tmp_valid_data)):
        q_text = d['question']
        p_texts = [p['passage'] for p in d['passages']]
        a = reader.answer(q_text, p_texts, topk=1)

        competitor_answer = a if a else ""
        qid = d["id"]

        if i < 5:
            logger.info(f"q_text: {q_text} a_text: {a}")

        output_dict.append({
            "qid": str(qid),
            "question": q_text,
            "pred_ans": competitor_answer
        })

        right_flag = is_exact_match_answer(qid, competitor_answer)

        total += 1
        if right_flag == "1":
            right += 1
        max_f, max_f_precision, max_f_recall, max_f_answer = cacu_character_level_f(qid, competitor_answer)
        sum_f += max_f
    logger.info(
        f"epoch: {e_num} Accuracy={1.0 * right / total} {right}/{total}  F1={sum_f / total} {sum_f}/{total}   Final={(1.0 * right / total + sum_f / total) / 2.}")

    json.dump(output_dict, Path(f"P_s2s_pred_answer_{e_num}.json").open("w"), ensure_ascii=False, indent=2)

    return (1.0 * right / total + sum_f / total) / 2.

best_final = -1
model.zero_grad()
for e in range(epochs):
    model.train()
    for step, batch in enumerate(train_generator):

        batch = [_.to(device) for _ in batch]
        logits = model(*batch)
        input_ids, segment_ids = batch
        loss = compute_loss((input_ids, segment_ids, logits))

        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if (step+1) % 100 == 0:
            logger.info(f"epoch: {e} - step: {step+1} - loss: {loss}")

    # evaluation
    final = eval_valid_data(valid_data[:1000], e)
    if final > best_final:
        best_final = final

        model_to_save = model.module if hasattr(model, "module") else model
        torch.save(model_to_save.state_dict(), "rc_model.pt")


# final evaluation
logger.info("begin final evaluation ...")
model = model_to_save
eval_valid_data(valid_data, e)
logger.info("end final evaluation ...")







