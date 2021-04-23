import torch
import torch.nn as nn
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from transformers import AutoConfig

from bojone_snippets import text_segmentate, DataGenerator, sequence_padding, AutoRegressiveDecoder
from bojone_tokenizers import load_vocab, Tokenizer
from configuration.config import *
from model_unilm import BojoneModel
from optimizer import create_optimizer_and_scheduler

# 基本参数
max_p_len = 128
max_q_len = 64
max_a_len = 16
batch_size = 32
epochs = 100

log_steps = 100

# bert配置
dict_path = str(bert_model_path / 'vocab.txt')

# 标注数据
webqa_data = json.load((mrc_datset_path/'WebQA.json').open())
sogou_data = json.load((mrc_datset_path/'SogouQA.json').open())

# 筛选数据
seps, strips = u'\n。！？!?；;，, ', u'；;，, '
data = []
for d in webqa_data + sogou_data:
    for p in d['passages']:
        if p['answer']:
            for t in text_segmentate(p['passage'], max_p_len - 2, seps, strips):
                if p['answer'] in t:
                    data.append((t, d['question'], p['answer']))

del webqa_data
del sogou_data

# 保存一个随机序（供划分valid用）
if not os.path.exists('random_order__all.json'):
    random_order = list(range(len(data)))
    np.random.shuffle(random_order)
    json.dump(random_order, open('random_order__all.json', 'w'), indent=4)
else:
    random_order = json.load(open('random_order__all.json'))

# 划分valid
train_data = [data[j] for i, j in enumerate(random_order) if i % 10 != 0]
valid_data = [data[j] for i, j in enumerate(random_order) if i % 10 == 0]

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
        """单条样本格式：[CLS]篇章[SEP]答案[SEP]问题[SEP]
        """
        batch_token_ids, batch_segment_ids = [], []
        for is_end, (p, q, a) in self.sample(random):
            p_token_ids, _ = tokenizer.encode(p, maxlen=max_p_len + 1)
            a_token_ids, _ = tokenizer.encode(a, maxlen=max_a_len)
            q_token_ids, _ = tokenizer.encode(q, maxlen=max_q_len)
            token_ids = p_token_ids + a_token_ids[1:] + q_token_ids[1:]
            segment_ids = [0] * len(p_token_ids)
            segment_ids += [1] * (len(token_ids) - len(p_token_ids))
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
config = AutoConfig.from_pretrained(pretrained_model_name_or_path=bert_wwm_pt_path, keep_tokens=keep_tokens, vocab_size=len(keep_tokens))
tmp_state_dict = torch.load(bert_wwm_pt_path/"pytorch_model.bin", map_location="cpu")
tmp_state_dict['bert.embeddings.word_embeddings.weight'] = torch.index_select(tmp_state_dict['bert.embeddings.word_embeddings.weight'], 0, torch.tensor(keep_tokens, dtype=torch.long))
tmp_state_dict["bert.cls.transform.dense.weight"] = tmp_state_dict["cls.predictions.transform.dense.weight"]
tmp_state_dict["bert.cls.transform.dense.bias"] = tmp_state_dict["cls.predictions.transform.dense.bias"]
tmp_state_dict["bert.cls.transform.LayerNorm.weight"] = tmp_state_dict["cls.predictions.transform.LayerNorm.weight"]
tmp_state_dict["bert.cls.transform.LayerNorm.bias"] = tmp_state_dict["cls.predictions.transform.LayerNorm.bias"]

model = BojoneModel.from_pretrained(pretrained_model_name_or_path=bert_model_path, config=config, state_dict=tmp_state_dict)
optimizer, scheduler = create_optimizer_and_scheduler(model, lr=1e-5, num_training_steps=train_generator.steps * epochs)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)


class QuestionAnswerGeneration(AutoRegressiveDecoder):
    """随机生成答案，并且通过beam search来生成问题
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)

        token_ids = torch.tensor(token_ids, dtype=torch.long).to(device)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long).to(device)

        with torch.no_grad():
            y_pred = model(token_ids, segment_ids)
        y_pred = torch.softmax(y_pred[:,-1,:].squeeze(1), dim=-1)
        y_pred = y_pred.cpu().detach().numpy()

        return y_pred

    def generate(self, passage, topk=1, topp=0.95):
        token_ids, segment_ids = tokenizer.encode(passage, maxlen=max_p_len)
        a_ids = self.random_sample([token_ids, segment_ids], 1, topp=topp)[0]  # 基于随机采样
        token_ids += list(a_ids)
        segment_ids += [1] * len(a_ids)
        q_ids = self.beam_search([token_ids, segment_ids],
                                 topk=topk)  # 基于beam search
        return tokenizer.decode(q_ids), tokenizer.decode(a_ids)


qag = QuestionAnswerGeneration(
    start_id=None, end_id=tokenizer._token_end_id, maxlen=max_q_len
)

def predict_and_evaluate(e_num):
    s1 = "7月28日，泰国将迎来十世王玛哈·哇集拉隆功国王的68岁诞辰。"
    s2 = "泼水节，亦称宋干节，已有700年的历史，是傣族(中国)、德昂族最盛大的传统节日。"
    s3 = "世界第二高山峰是乔戈里峰,位于中国境内。"
    s4 = "您购买的保险产品由百年人寿保险股份有限公司承保，目前该公司在大连、湖北、 河北、辽宁、北京、河南、黑龙江、安徽、山东、江苏、四川、福建、陕西、内 蒙古、吉林、江西、山西、浙江、广东和重庆地区设有分支机构，本产品在该公 司设有分公司的区域销售"
    s5 = "百年人寿通过保通保险代理有限公司销售保险产品的合同订立均采取电子保单 形式，您在投保成功后 24 小时内，电子保单会发送到您填写的投保人邮箱中， 电子保单与纸质保单具有同等法律效力。"
    for s in [s1, s2, s3, s4, s5]:
        logger.info(f'生成问答: {qag.generate(s)}')

    tmp_valid_data = valid_data[:1000] if e < epochs-1 else valid_data

    total = 0
    rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0
    qa_rouge_1, qa_rouge_2, qa_rouge_l, qa_bleu = 0, 0, 0, 0
    for passage, question, answer in tmp_valid_data:
        total += 1
        pred_q, pred_a = qag.generate(passage, topk=1)

        # 生成问题的评测
        question = ' '.join(question).lower()
        pred_q = ' '.join(pred_q).lower()
        if pred_q.strip():
            scores = rouge.get_scores(hyps=pred_q, refs=question)
            rouge_1 += scores[0]['rouge-1']['f']
            rouge_2 += scores[0]['rouge-2']['f']
            rouge_l += scores[0]['rouge-l']['f']
            bleu += sentence_bleu(
                references=[question.split(' ')],
                hypothesis=pred_q.split(' '),
                smoothing_function=smooth
            )

        # 生成问答对的评测
        qa = ' '.join(question + "。" + answer).lower()
        pred_qa = ' '.join(pred_q + "。" + pred_a).lower()
        if pred_qa.strip():
            qa_scores = rouge.get_scores(hyps=pred_qa, refs=qa)
            qa_rouge_1 += qa_scores[0]['rouge-1']['f']
            qa_rouge_2 += qa_scores[0]['rouge-2']['f']
            qa_rouge_l += qa_scores[0]['rouge-l']['f']
            qa_bleu += sentence_bleu(
                references=[qa.split(' ')],
                hypothesis=pred_qa.split(' '),
                smoothing_function=smooth
            )
    rouge_1 /= total
    rouge_2 /= total
    rouge_l /= total
    bleu /= total

    qa_rouge_1 /= total
    qa_rouge_2 /= total
    qa_rouge_l /= total
    qa_bleu /= total

    result = {
        "rouge-1": rouge_1,
        "rouge-2": rouge_2,
        "rouge-l": rouge_l,
        "bleu": bleu,
        "qa_rouge-1": qa_rouge_1,
        "qa_rouge-2": qa_rouge_2,
        "qa_rouge-l": qa_rouge_l,
        "qa_bleu": qa_bleu
    }

    logger.info(f"enum: {e_num}, rouge-1: {rouge_1}, rouge-2: {rouge_2}, rouge-l: {rouge_l}, bleu: {bleu}")
    logger.info(f"enum: {e_num}, qa_rouge-1: {qa_rouge_1}, qa_rouge-2: {qa_rouge_2}, qa_rouge-l: {qa_rouge_l}, qa_bleu: {qa_bleu}")

    return result

best_final = 1e-10
rouge = Rouge()
smooth = SmoothingFunction().method1
model.zero_grad()
for e in range(epochs):
    model.train()
    for step, batch in enumerate(train_generator):
        # if step > 10: break
        batch = [_.to(device) for _ in batch]
        input_ids, segment_ids = batch
        y_pred = model(input_ids, segment_ids)
        loss = compute_loss((input_ids, segment_ids, y_pred))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % log_steps == 0 and step != 0:
            logger.info(f"epoch: {e} - step: {step}/{train_generator.steps} - loss: {loss}")

    model.eval()
    pred_metric = predict_and_evaluate(e)
    if pred_metric["rouge-l"] > best_final:
        best_final = pred_metric["rouge-l"]

        model_to_save = model.module if hasattr(model, "module") else model
        torch.save(model_to_save.state_dict(), "gen_qa_model.pt")





def predict_to_file(data, filename, topk=1):
    """将预测结果输出到文件，方便评估
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for d in tqdm(iter(data), desc=u'正在预测(共%s条样本)' % len(data)):
            q, a = qag.generate(d[0])
            s = '%s\t%s\t%s\n' % (q, a, d[0])
            f.write(s)
            f.flush()

