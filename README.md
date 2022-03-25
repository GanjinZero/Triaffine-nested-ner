# Triaffine-nested-ner
Fusing Heterogeneous Factors with Triaffine Mechanism for Nested Named Entity Recognition [ACL2022 Findings] [Paper link](https://arxiv.org/abs/2110.07480)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/fusing-heterogeneous-factors-with-triaffine/nested-named-entity-recognition-on-ace-2004)](https://paperswithcode.com/sota/nested-named-entity-recognition-on-ace-2004?p=fusing-heterogeneous-factors-with-triaffine)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/fusing-heterogeneous-factors-with-triaffine/nested-named-entity-recognition-on-ace-2005)](https://paperswithcode.com/sota/nested-named-entity-recognition-on-ace-2005?p=fusing-heterogeneous-factors-with-triaffine)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/fusing-heterogeneous-factors-with-triaffine/nested-named-entity-recognition-on-genia)](https://paperswithcode.com/sota/nested-named-entity-recognition-on-genia?p=fusing-heterogeneous-factors-with-triaffine)


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
@article{Yuan2021FusingHF,
  title={Fusing Heterogeneous Factors with Triaffine Mechanism for Nested Named Entity Recognition},
  author={Zheng Yuan and Chuanqi Tan and Songfang Huang and Fei Huang},
  journal={ArXiv},
  year={2021},
  volume={abs/2110.07480}
}
```
