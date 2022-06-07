# PyTorch Knowledge distillation

(Currently under early developments.)

Reproduce experiments for Knowlidge Distillation methods <https://arxiv.org/abs/2006.05525> using PyTorch with [MMclassification](https://github.com/open-mmlab/mmclassification)

## Requirements

- Docker

See Dockerfile for details about development environments.

## Run it

```bash
docker-compose run --rm test /bin/bash
python -m pip install -e .
python tools/train.py "<config file path. see experiments/*.py>"
```

For example, you can enjoy one of the most popular knowledge distillation method for image classification as:

- kd loss: regression on logits. ((Ba, J. & Caruana, R. (2014). Do deep nets really need
to be deep? In: NeurIPS)[<https://papers.nips.cc/paper/2014/hash/ea8fcd92d59581717e06eb187f10666d-Abstract.html>])
- teacher model: ResNet50
- student model: ResNet18
- Dataset: CIFAR-10

In container:

```bash
$ cd /home/kd
$ export PYTHONPATH="`pwd`:$PYTHONPATH"
# Learn large teacher model
$ python tools/train.py experiments/baseline/resnet50.py
# wait a few hours...
# (Optionally) try to learn small student model by oneself
$ python tools/train.py experiments/baseline/resnet18.py
# wait a few hours...
# Then, Learn small student model with mimic teacher model prediction
$ OUTPUT="work_dir/response_based_logts_resnet18"
$ python tools/train.py experiments/response_based/logits_resnet18.py --work-dir $OUTPUT
# check teacher could helps student or not
$ python tools/test.py $OUTPUT/logits_resnet18.py $OUTPUT/latest.pth --out $OUTPUT/test_result.json
```

See more details with `--help` arguments for `train.py`/`test.py`.

## Benchmarks

Ongoing...
