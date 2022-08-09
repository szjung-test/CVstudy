# 컴퓨터 비전을 이용한 프로젝트 정리
- SRGAN code execution(8월3일)
- source : (https://github.com/leftthomas/SRGAN), (https://velog.io/@hyun-wle/SRGAN-%EB%94%B0%EB%9D%BC%ED%95%98%EA%B8%B0)
- [Object Detection 공부 참고 블로그]: https://2hyes.tistory.com/171

#  SRGAN 
- train.py 실행하여 100 epoch 학습
0. 환경 설정
- git, pip3, pytorch 설치된 환경

1. github 코드 다운
```
$ git clone https://github.com/leftthomas/SRGAN.git SRGAN
```
- SRGAN 폴더로 이동
```
cd SRGAN
```
- 폴더 내부 확인
```
ls
```

```
$ cd data
$ sudo apt install unzip
$ wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
$ unzip DIV2K_train_HR.zip
```

### pytorch cuda 설치

[파이토치 쿠다](https://pytorch.org/get-started/locally/).
```
nvidia-smi 로 확인 후 CUDA Version: 11.6
```

- 11.6을 다운, window
![image](https://user-images.githubusercontent.com/93111772/182739121-445aedb4-5dd5-49fd-af0b-779ee021d8df.png)

```
'conda-forge' channel is required for cudatoolkit 11.6
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
```

- cuda 까지 다 설치하고 나면 train.py 학습을 돌린다.

```
python train.py
```

- test_image.py를 통해 SR할 이미지를 돌려본다.
```
python test_image.py --image_name=sz.jpg
```

