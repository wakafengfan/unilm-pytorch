# unilm-pytorch

本库主要在bert4keras基础上，使用pytorch实现基于UniLM+BERT进行seq2seq机器阅读理解、问答对生成、标题生成，以及基于simbert的相似问题生成。

## Environment

- python==3.6.*
- pytorch==1.8
- transformers==4.4.2

## 功能

#### 1.seq2seq机器阅读理解
```bash
python reading_comprehension_by_seq2seq.py
```

数据集：SougoQA + WebQA

评测结果：

| Model        | EM        | F1        | (EM + F1)/2  |
| ------------ | --------- | --------- | ------------ |
| unilm + bert | 0.73093   | 0.81191   | 0.77142      |

#### 2.标题生成
```bash
python gen_title.py
```

数据集：csl

评测结果：

| Model        | Rouge-l   | Rouge-1   | Rouge-2      | Bleu         |
| ------------ | --------- | --------- | ------------ | ------------ |
| unilm + bert | 0.61305   | 0.62182   | 0.51253      | 0.41312      |

#### 3.问答对生成
```bash
python gen_qa_pair.py
```

数据集：SougoQA + WebQA

生成效果：
```bash
"7月28日，泰国将迎来十世王玛哈·哇集拉隆功国王的68岁诞辰。"

生成问答: ('玛哈·哇集拉隆功是哪个国家的国王?', '泰国')
生成问答: ('泰国的十世王王是多少岁', '68')
生成问答: ('泰国王王玛哈·哇集拉隆功的生日是哪一天?', '7月28日')


"泼水节，亦称宋干节，已有700年的历史，是傣族(中国)、德昂族最盛大的传统节日。"

生成问答: ('泼水节是我国哪个少数民族的传统节日？', '傣族')
生成问答: ('泼水节是我国哪一民族一年中最盛大的节日', '傣族')


"世界第二高山峰是乔戈里峰,位于中国境内。"

生成问答: ('世界第二高山峰是什么', '乔戈里峰')


"您购买的保险产品由百年人寿保险股份有限公司承保，目前该公司在大连、湖北、 河北、辽宁、北京、河南、黑龙江、安徽、山东、江苏、四川、福建、陕西、内 蒙古、吉林、江西、山西、浙江、广东和重庆地区设有分支机构，本产品在该公司设有分公司的区域销售"

生成问答: ('百年人寿在哪个城市', '大连')
生成问答: ('百年人寿是什么公司', '保险公司')


"百年人寿通过保通保险代理有限公司销售保险产品的合同订立均采取电子保单形式，您在投保成功后 24 小时内，电子保单会发送到您填写的投保人邮箱中， 电子保单与纸质保单具有同等法律效力。"

生成问答: ('保险代理人代为办理保险业务，承担保险责任的主体是：', '保险人')
生成问答: ('百年人寿是谁销售', '保险代理')
生成问答: ('保险代理人在几小时内填写电子保单', '24小时')

```

#### 4.基于simbert的相似问题生成
```bash
python gen_synonym.py
```

生成效果：
```bash
print(gen_synonyms(text="投保需要哪些材料"))

[
	'个人投保需要哪些材料？',
	'投保人员需要提供哪些材料',
	'投保需要哪些材料？',
	'投保需要准备什么材料',
	'投保保险都需要什么材料',
	'投保需要哪些资料',
	'投保需要提供哪些材料',
	'投保保险需要什么材料',
	'办理保险投保需要哪些材料',
	'个人投保需要什么材料',
	'投保需要什么资料？',
	'投保需要什么材料',
	'投保需要什么材料？',
	'投保保险需要哪些资料',
	'投保保险需要哪些材料？',
	'投保保险都需要哪些材料',
	'投保保险需要哪些材料'
]
```


pytorch版本的simbert权重下载地址：[百度网盘](https://pan.baidu.com/s/1zYBtU21vdNsM1QgMwrbovA) 密码：v8na

### Todo

- 评测seq2seq机器阅读理解模型在Dureader上效果
- 提升标题生成，复现bert4Keras的效果
- 评测问答对生成模型在squad上效果，同微软开源[unilm](https://github.com/microsoft/unilm/tree/master/unilm-v1) 版本进行对比

### 参考：

https://bert4keras.spaces.ac.cn

