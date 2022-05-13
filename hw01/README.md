# ADL NTU 110 Spring - Homework 1

## Dependencies

python 3.8.12

Install the dependencies. Since the project is developed under CUDA 11.3, if you need other CUDA version, please check the [Pytorch page](https://pytorch.org/get-started/locally/) and install Pytorch manually.

```shell
pip install -r ./requirements.txt
```

## Training

### Intent classification

#### Preprocessing

Prepare embedding vectors, itent-to-index and vocabulary file.

```shell
bash ./intent_preprocess.sh
```

#### Model Training

Train the model with default setting.

```shell
bash ./intent_train.sh
```

### Slot tagging

#### Preprocessing

Prepare embedding vectors, itent-to-index and vocabulary file.

```shell
bash ./slot_preprocess.sh
```

#### Model Training

Train the model with default setting.

```shell
bash ./slot_train.sh
```

## Testing

### Download

Download the models and the cache files.

```shell
bash ./download.sh
```

### Intent classification

Appy intent detection on testing file.

```shell
bash ./intent_cls.sh /path/to/test.json /path/to/pred.csv
```

### Slot tagging

Apply slot tagging on testing file.

```shell
bash ./slot_tag.sh /path/to/test.json /path/to/pred.csv
```

## Reference

- Sample code: [ntu-adl-ta/ADL21-HW1](https://github.com/ntu-adl-ta/ADL21-HW1)
