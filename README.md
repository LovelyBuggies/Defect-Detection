# Defect Detection

## Task Definition

Using CodeBERT to embed the codes and to identify whether they contain vulnerabilities.

## Preprocess

1. **Local:** Download dataset from [website](https://drive.google.com/file/d/1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF/view?usp=sharing) to "dataset" folder or run the following command.

```shell
cd dataset
pip install gdown
gdown https://drive.google.com/uc?id=1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF
python preprocess.py
cd ..
```

2. **CoLab:** Preprocess dataset.

```shell
git clone https://github.com/LovelyBuggies/Defect-Detection.git
pip3 install transformers
cd Defect-Detection/
```

## Fine-tune

```shell
cd code
python run.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --do_train \
    --do_eval \
    --do_test \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/valid.jsonl \
    --test_data_file=../dataset/test.jsonl \
    --epoch 5 \
    --block_size 400 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log
```

### Evaluation

```shell
python ../evaluator/evaluator.py -a ../dataset/test.jsonl -p saved_models/predictions.txt
```

{'Acc': 0.6325106295754027}

## Result

The results on the test set are shown as below:

| Methods                                          |    ACC    |
| ------------------------------------------------ | :-------: |
| BiLSTM                                           |   59.37   |
| TextCNN                                          |   60.69   |
| [RoBERTa](https://arxiv.org/pdf/1907.11692.pdf)  |   61.05   |
| [CodeBERT](https://arxiv.org/pdf/2002.08155.pdf) | **63.25** |
