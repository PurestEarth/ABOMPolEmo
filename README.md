# ABSAPolEmo




## Credits

XLMRForTokenClassification implementation based on [xlm-roberta-ner](https://github.com/mohammadKhalifa/xlm-roberta-ner) by mohammadKhalifa. 


## Example train commands



0. LSTM - 
```
python3 trainLSTM.py \
              --data_dir=./motherfile.json \
              --model LSTM \
              --motherfile \
              --output_dir ./lstm-polemo \
              --max_seq_length=128 \
              --embedding ../pretrained/pretrained_models/elmo-kgr10-e2000000 \
              --epochs 20
```
1. Polish roberta - 
```
python3 train.py \
              --data_dir=./motherfile.json\
              --model POLISH_ROBERTA \
              --output_dir ./polish-roberta-polemo \
              --pretrained ../pretrained/pretrained_models/polish_roberta \
              --epochs 50 \
              --motherfile \
              --wandb True
```
2. Reformer - 
```
python3 train.py \
              --data_dir=./motherfile.json\
              --model REFORMER \
              --output_dir ./reformer-polemo \
              --epochs 50 \
              --motherfile\
              --wandb True
```
3. XLM-roberta
```
python3 train.py \
              --data_dir=./motherfile.json\
              --output_dir ./xlm-polemo \
              --model_name XLMR \
              --pretrained ../pretrained/pretrained_models/roberta_large \
              --epochs 50 \
              --motherfile\
              --wandb True
```
4. HerBERT
```
python3 train.py \
              --data_dir=./motherfile.json\
              --output_dir ./herbert-polemo \
              --model_name HERBERT \
              --pretrained_path allegro/herbert-large-cased \
              --epochs 50 \
              --motherfile\
              --wandb True
```
5. mBERT
```
python3 train.py \
              --data_dir=./motherfile.json\
              --output_dir ./mbert-polemo \
              --model_name BERT_MULTILINGUAL \
              --pretrained_path bert-base-multilingual-cased \
              --epochs 50 \
              --motherfile \
              --g 1\
              --wandb True
```



## Results

|    Model   | F1 | Acc. |  Recall |
|--------|-----------|--------|---|
|  LSTM  | 42.63 ± 0.6 |  41.5  ± 2.1  |  44.03  ± 2.03 |
|  mBERT | 43.65 ± 0.72|  45.76 ± 1.55 | 41.79 ± 1.17   | 
| Polish RoBERTa | 63.73 ± 0.55 |  66.61 ± 1.53  |  61.14 ± 1.30 |
|  XLM-Roberta  | 65.13 ± 0.43 |  66.99 ± 0.99  |  63.38 ± 0.75  |
|  HerBERT    | 65.22 ± 0.29 |  67.56 ± 1.01 |  63.05 ± 0.86 | 