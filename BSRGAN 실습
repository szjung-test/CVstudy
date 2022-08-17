reference : https://github.com/HeaseoChung/Super-resolution


- 사용하는 모델은 학습을 돌릴때 하이드라코어가 필요함
```
pip install hydra-core
```

> Super-resolution>configs>train>PSNR.yaml
- PSNR.yaml 파일에서 내컴퓨터와 데이터셋 연결
```
dataset:
  train_dir: "/workspace/newworld/SRGAN/data/DIV2K_train_HR"
  batch_size: 4
  patch_size : 128
  num_workers: 4
```
> Super-resolution>configs>train>train.yaml
- 사용할 train과 models 수정해준다.
```
hydra:
  run:
    dir: ./outputs/BSRGAN/train/${now:%Y-%m-%d}/${now:%H-%M-%S}

defaults:
  - _self_
  - train: PSNR
  - models: BSRGAN
```
 

```
szjung@esp:/workspace/newworld/Super-resolution/scripts$ CUDA_VISIBLE_DEVICES=1 python trainer.py 
```
