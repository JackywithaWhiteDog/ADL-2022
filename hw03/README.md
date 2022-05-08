# ADL NTU 110 Spring - Homework 3

## Dependencies

- Python version: 3.8.12

Install other dependencies. Since the project is developed under CUDA 11.3, if you need other CUDA version, please check the [Pytorch page](https://pytorch.org/get-started/locally/) and install Pytorch manually.

```shell
pip install -r ./requirements.txt
```

### Evaluation

To apply evaluation in validation phase, install [tw_rouge](https://github.com/JackywithaWhiteDog/ADL-2022/tree/main/hw03/tw_rouge)

```shell
pip install -e tw_rouge
```

## Training

Fine tune the mT5 model with default setting

```shell
bash ./train.sh
```

## Testing

Download the models and the cache files.

```shell
bash ./download.sh
```

### Predicting

Apply the inference to the testing file

```shell
bash ./run.sh /path/to/input.jsonl /path/to/output.jsonl
```

## Reference

- Data & Evaluation (ROUGE score with chinese word segmentation): [moooooser999/ADL22-HW3](https://github.com/moooooser999/ADL22-HW3)
