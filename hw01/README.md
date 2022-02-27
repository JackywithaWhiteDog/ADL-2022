# ADL NTU 110 Spring - Homework 1

## Environment

Install dependencies.

```shell
pip install -r ./requirements.txt
```

## Download

Download the models.

```shell
bash ./download.sh
```

## Intent detection

Appy intent detection on testing file.

```shell
bash ./intent_cls.sh /path/to/test.json /path/to/pred.csv
```

## Slot tagging

Apply slot tagging on testing file.

```shell
bash ./slot_tag.sh /path/to/test.json /path/to/pred.csv
```
