# 컴퓨터 비전을 이용한 프로젝트 정리
- SRGAN code execution(8월3일)
- source : (https://github.com/leftthomas/SRGAN), (https://velog.io/@hyun-wle/SRGAN-%EB%94%B0%EB%9D%BC%ED%95%98%EA%B8%B0)
- 블로그에 너무 잘 정리해주셔서 도움이 많이 

# 영상 관련 용어 정리
- 코덱(Codec) : 영상이나 음성 신호를 디지털 신호로 변환하거나 반대로 변환하는 기능을 수행하는 기술, 코더(Coder)와 디코더(Decoder)의 합성어
- MPEG : Moving Picture Experts Group, ISO 및 IEC 산하에서 비디오와 오디오 등 멀티미디어의 표준과 압축기술을 개발을 담당하는 소규모 그룹 
- ex) MPEG1, MPEG2, MPEG4, MPEG7, MPEG21
- MPEG1 : 최초의 비디오와 오디오 표준 압출기술, CD와 같은 매체에서 동영상을 담기 위해 사용, 최대 전송률 약 1.5Mbps이고 최대 해상도가 352x288
- MPEG2 : 디지털 방송이나 DVD와 같은 동영상 압축에 사용되는 손실 압축기술, 오래된 기술이지만 전송률 4~100Mbps, 아직도 디지털 방송에서 사용, Full HD 해상도까지 구현
- MP3 : 음악과 음성 데이터를 압축하는 기술, MP3는 MPEG3 가 아니라 MPEG1의 레이어3을 말하는 것
- MPEG4 : 현재 우리의 일상에서 흔히 사용되는 포맷, 유투브와 같이 인터넷상에 업로드되는 동영상은 대부분, 줄여서 MP4로 표현, 양방향 멀티미디어(동영상, 화상)구현하기 위한 압축 기술, 64Kbps급의 낮은 속도, 높은 압축률을 구현, 고화질 영상의 뛰어난 압축 효율성을 보이는 H.264 코덱과 함께 사용
- MPEG7 : 정보검색을 위한 목적으로 사용되는 압축 표준, 멀티미디어 정보, 콘텐츠를 제작 및 전송, 저장, 유통, 검색 분야에서 사용, 필요한 정보를 찾고자 할 때 데이터베이스에서 쉽게 탐지하고 추출, 
- MPEG21 : 네트워크나 장치에 있는 멀티미디어 자원을 효율적으로 이용하기 위해 개발된 방식, 주로 전자상거래를 위한 목적으로 사용, 사용자가 콘텐츠를 클릭 및 입력하고 표현되는 상호작용 방식 

- 디인터레이스 (Deinterlacing) : 인터레이스(비월주사)방식의 영상을 프로그레시브(순차주사로 바꾸는 과정)
- 인터레이스 (Interlace) : tv가 최초 발명됐을때 부족한 하드웨어 성능으로 인하여 사람들이 원하는 대로 초당 60회 이상 화명을 스캔할 수 없었다. 그래서 하나의 화면을 가로로 마디마디로 자른뒤 처음은 홀수 번째 줄만 스캔을 하고 그 다음은 짝수 번재 줄만 스캔하는 방식으로 초당 60장의 영상을 구현하였는데 이것이 인터레이스다. 스캔된 결과물 하나하나를 필드라고 하며 짝수줄만 스캔한 것을 탑 필드, 홀 수 줄만 스캔한 것을 바텀 필드로 구분, 탑필듣와 바텀필드를 합치면 하나의 프레임이 된다. 프로그레시브 영상은 하나의 화면이 하나의 프레임을 이룬다.

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

