# Triaffine-nested-ner
Fusing Heterogeneous Factors with Triaffine Mechanism for Nested Named Entity Recognition [ACL2022 Findings] [Paper link](https://arxiv.org/abs/2110.07480)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/fusing-heterogeneous-factors-with-triaffine/nested-named-entity-recognition-on-ace-2004)](https://paperswithcode.com/sota/nested-named-entity-recognition-on-ace-2004?p=fusing-heterogeneous-factors-with-triaffine)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/fusing-heterogeneous-factors-with-triaffine/nested-named-entity-recognition-on-ace-2005)](https://paperswithcode.com/sota/nested-named-entity-recognition-on-ace-2005?p=fusing-heterogeneous-factors-with-triaffine)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/fusing-heterogeneous-factors-with-triaffine/nested-named-entity-recognition-on-genia)](https://paperswithcode.com/sota/nested-named-entity-recognition-on-genia?p=fusing-heterogeneous-factors-with-triaffine)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/fusing-heterogeneous-factors-with-triaffine/nested-named-entity-recognition-on-tac-kbp)](https://paperswithcode.com/sota/nested-named-entity-recognition-on-tac-kbp?p=fusing-heterogeneous-factors-with-triaffine)


# Environment
All codes are tested under Python 3.7, PyTorch 1.7.0 and Transformers 4.6.1.
Need to install opt_einsum for einsum calculations.
At least 16GB GPU are needed for training.

# Dataset
We only put several samples for each dataset.
You can download full GENIA dataset [here](https://drive.google.com/file/d/1i37ZmJAofKXuOJbq1nG5kqPTM081WfXQ/view?usp=sharing).
There are no direct download links for other datasets.
One need to obtain licences from LDC for getting ACE2004, ACE2005 and KBP2017.
You can email me with LDC licences to obtain processed datasets (yuanzheng.yuanzhen@alibaba-inc.com). 
Please put datas under data/dataset_name, you can also refer word_embed.generate_vocab_embed for data paths.

# Extract word embedding
Please download [cc.en.300.bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz) and [BioWordVec_PubMed_MIMICIII_d200.bin](https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioWordVec_PubMed_MIMICIII_d200.bin) and run *python word_embed.py* to generate required json files. You need to change the path of word embedding.

# Reproduce
To notice, you may need to change the path of bert_name_or_path by yourself.

ace04 + bert
```shell
CUDA_VISIBLE_DEVICES=0 python main.py --version ace04 --model SpanAttModelV3 --bert_name_or_path ../pretraining-models/bert-large-cased/ --learning_rate 1e-5 --batch_size 1 --gradient_accumulation_steps 8 --train_epoch 50 --score tri_affine --truncate_length 192 --word --word_dp 0.2 --word_embed cc --char --pos --use_context --warmup_ratio 0.0 --att_dim 256  --bert_before_lstm --lstm_dim 1024 --lstm_layer 2 --encoder_learning_rate 1e-4 --max_span_count 30 --share_parser --subword_aggr max --init_std 1e-2 --dp 0.1
```

ace05 + bert
```shell
CUDA_VISIBLE_DEVICES=0 python main.py --version ace05 --model SpanAttModelV3 --bert_name_or_path ../pretraining-models/bert-large-cased/ --learning_rate 3e-5 --batch_size 1 --gradient_accumulation_steps 72 --train_epoch 50 --score tri_affine --truncate_length 192 --word --word_dp 0.2 --word_embed cc --char --pos --use_context --warmup_ratio 0.0 --att_dim 256  --bert_before_lstm --lstm_dim 1024 --lstm_layer 2 --encoder_learning_rate 5e-4 --max_span_count 30 --share_parser --subword_aggr max --init_std 1e-2 --dp 0.1
```

genia + biobert
```shell
CUDA_VISIBLE_DEVICES=0 python main.py --version genia91 --model SpanAttModelV3 --bert_name_or_path ../pretraining-models/biobert_v1.1/ --learning_rate 3e-5 --batch_size 1 --gradient_accumulation_steps 48 --train_epoch 15 --score tri_affine --truncate_length 192 --word --word_dp 0.2 --char --pos --use_context --warmup_ratio 0.0  --att_dim 320 --bert_before_lstm --lstm_dim 1024 --lstm_layer 2 --encoder_learning_rate 5e-4 --max_span_count 30 --share_parser --subword_aggr max --init_std 1e-2 --dp 0.2
```

kbp + bert
```shell
CUDA_VISIBLE_DEVICES=0 python main.py --version kbp --model SpanAttModelV3 --bert_name_or_path ../pretraining-models/bert-large-cased/ --learning_rate 1e-5 --batch_size 1 --gradient_accumulation_steps 8 --train_epoch 50 --score tri_affine --truncate_length 192 --word --word_dp 0.2 --word_embed cc --char --pos --use_context --warmup_ratio 0.0 --att_dim 256  --bert_before_lstm --lstm_dim 1024 --lstm_layer 2 --encoder_learning_rate 1e-4 --max_span_count 30 --share_parser --subword_aggr max --init_std 1e-2 --dp 0.1
```

ace04 + albert
```shell
CUDA_VISIBLE_DEVICES=0 python main.py --version ace04 --model SpanAttModelV3 --bert_name_or_path ../pretraining-models/albert-xxlarge-v2/ --learning_rate 1e-5 --batch_size 1 --gradient_accumulation_steps 8 --train_epoch 10 --score tri_affine --truncate_length 192 --word --word_dp 0.2 --word_embed cc --char --pos --use_context --warmup_ratio 0.0 --att_dim 256  --bert_before_lstm --lstm_dim 1024 --lstm_layer 2 --encoder_learning_rate 1e-4 --max_span_count 30 --share_parser --subword_aggr max --init_std 1e-2 --dp 0.1
```

ace05 + albert
```shell
CUDA_VISIBLE_DEVICES=0 python main.py --version ace05 --model SpanAttModelV3 --bert_name_or_path ../pretraining-models/albert-xxlarge-v2/ --learning_rate 1e-5 --batch_size 1 --gradient_accumulation_steps 8 --train_epoch 10 --score tri_affine --truncate_length 192 --word --word_dp 0.2 --word_embed cc --char --pos --use_context --warmup_ratio 0.0 --att_dim 256  --bert_before_lstm --lstm_dim 1024 --lstm_layer 2 --encoder_learning_rate 1e-4 --max_span_count 30 --share_parser --subword_aggr max --init_std 1e-2 --dp 0.1
```

kbp + albert
```shell
CUDA_VISIBLE_DEVICES=0 python main.py --version kbp --model SpanAttModelV3 --bert_name_or_path ../pretraining-models/albert-xxlarge-v2/ --learning_rate 3e-5 --batch_size 1 --gradient_accumulation_steps 72 --train_epoch 10 --score tri_affine --truncate_length 192 --word --word_dp 0.2 --word_embed cc --char --pos --use_context --warmup_ratio 0.0 --att_dim 256  --bert_before_lstm --lstm_dim 1024 --lstm_layer 2 --encoder_learning_rate 5e-4 --max_span_count 30 --share_parser --subword_aggr max --init_std 1e-2  --dp 0.2
```

# Citations
```
@inproceedings{yuan-etal-2022-fusing,
    title = "Fusing Heterogeneous Factors with Triaffine Mechanism for Nested Named Entity Recognition",
    author = "Yuan, Zheng  and
      Tan, Chuanqi  and
      Huang, Songfang  and
      Huang, Fei",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2022",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-acl.250",
    pages = "3174--3186",
    abstract = "Nested entities are observed in many domains due to their compositionality, which cannot be easily recognized by the widely-used sequence labeling framework.A natural solution is to treat the task as a span classification problem.To learn better span representation and increase classification performance, it is crucial to effectively integrate heterogeneous factors including inside tokens, boundaries, labels, and related spans which could be contributing to nested entities recognition.To fuse these heterogeneous factors, we propose a novel triaffine mechanism including triaffine attention and scoring.Triaffine attention uses boundaries and labels as queries and uses inside tokens and related spans as keys and values for span representations.Triaffine scoring interacts with boundaries and span representations for classification.Experiments show that our proposed method outperforms previous span-based methods, achieves the state-of-the-art $F_1$ scores on nested NER datasets GENIA and KBP2017, and shows comparable results on ACE2004 and ACE2005.",
}
```
