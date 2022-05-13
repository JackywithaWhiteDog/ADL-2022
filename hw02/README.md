# ADL NTU 110 Spring - Homework 2

## Dependencies

- Python version: 3.8.12

Install the dependencies. Since the project is developed under CUDA 11.3, if you need other CUDA version, please check the [Pytorch page](https://pytorch.org/get-started/locally/) and install Pytorch manually.

```shell
pip install -r ./requirements.txt
```

## Training

### Multiple choice

Train the multiple choice model with default setting

```shell
bash ./multi_train.sh
```

### Question answering

Train the question answering model with default setting

```shell
bash ./qa_train.sh
```

## Testing

### Download

Download the models and the cache files.

```shell
bash ./download.sh
```

### Predicting

Apply the inference to the testing file

```shell
bash ./run.sh /path/to/context.json /path/to/test.json /path/to/pred/prediction.csv
```

## BERT for HW01

### Intent classification

Train the intent classification model, evaluate with the validation set, and apply to the testing file with default setting

```shell
bash ./intent.sh
```

### Slot tagging

To run the slot tagging task, install [seqeval](https://pypi.org/project/seqeval/) library.

```
pip install seqeval==1.2.2
```

Train the slot tagging model, evaluate with the validation set, and apply to the testing file with default setting

```shell
bash ./slot.sh
```

## Reference

- Sample code: [huggingface/transformers](https://github.com/huggingface/transformers)
    - [Multiple choice](https://github.com/huggingface/transformers/tree/main/examples/pytorch/multiple-choice)
    - [Question answering](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering)
    - [Intent classification](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification)
    - [Slot tagging](https://github.com/huggingface/transformers/tree/main/examples/pytorch/token-classification)
