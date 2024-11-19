# Bone Segmentation Project
뼈는 인체의 구조와 기능에 필수적이기 때문에 정확한 뼈 분할은 의료 진단 및 치료 계획 수립에 매우 중요합니다. 본 프로젝트는 딥러닝 기술을 활용하여 뼈 Segmentation 모델을 구현하고자 합니다.

---

## 1. 프로젝트 목표 및 평가 지표
- **프로젝트 목표**: 뼈 분할 모델 개발 및 최적화.
- **적용 분야**: 의료 진단 및 치료 계획.
- **평가 지표**: Dice Score

---

## 2. 하이퍼파라미터 설정 방법
### 2.1. 설정 파일 경로
- **`configs/config.yaml`** 파일에서 하이퍼파라미터를 설정합니다.

### 2.2. 주요 설정 항목
#### Image Resize
이미지 크기를 설정하는 항목입니다. 
```yaml
image_size: &image_size 숫자
ex) image_size: &image_size 1024
```


#### 최대 epochs
```yaml
max_epoch: &max_epoch 숫자 <- 숫자자리에 max_epoch 넣어주시면 됩니다.
ex) max_epoch: &max_epoch 30
```

#### model_name()
모델이름을 설정합니다. 모델 이름은 자동으로 체크포인트 파일 저장 경로에 반영됩니다.
```yaml
model_name: &model_name 모델이름
ex) model_name: &model_name UnetPlusPlus
```
- 예: 모델 이름이 UnetPlusPlus라면 체크포인트는 checkpoints/UnetPlusPlus/ 경로에 저장됩니다.
- 저장 형식: best_epoch_{float}.pt (가장 성능이 좋은 epoch 기준).

그외 하이퍼파라미터는 직접 고쳐서 적으시면 됩니다.


# 3. train 및 test방법
## 3.1 train 방법
`python train.py`
- 학습 실행 시 설정된 하이퍼파라미터에 따라 모델이 학습됩니다.
- 학습 중 가장 성능이 좋은 체크포인트가 자동으로 저장됩니다.

## 3.2 test방법

**1. config.yaml 파일 수정**
테스트하려는 학습 결과 파일명을 설정합니다.

```yaml
test:
  model_file: best_100epoch_0.9999.pt
```
**2. Test 실행**

```bash
python test.py
```
학습된 모델 파일을 이용해 테스트가 자동으로 진행됩니다.


# 4. wandb sweep 사용 방법
## 4.1 초기에 sweep 설정(한번만)
아래 명령어로 sweep 을 시작합니다.
```bash
wandb sweep configs/config_sweep.yaml
```
## 4.2 config.yaml 설정
wandb 설정에서 Sweep 사용 여부를 True로 변경합니다.
```yaml
wandb:
  use_sweep: True
```
## 4.3 wandb 홈페이지에 들어가서 해당 프로젝트에 들어가서 sweep 탭 클릭
sweep configuration -> initialize sweep -> wande agent 커맨드 명령어 줄 것임. 해당 명령어 실행하면 sweep 실행함.

# 5. streamlit 사용 방법
streamlit 폴더에 들어가서
`streamlit run main.py`

# 6. 폴더 구조
```bash
.
├── checkpoints # 데이터 학습된 모델 저장소
├── code
│   ├── __init__.py
│   ├── loader
│   ├── loss_functions
│   ├── models
│   ├── __pycache__
│   ├── requirements.txt
│   ├── scheduler
│   ├── utils
│   └── validataion
├── configs
│   ├── config_sweep.yaml
│   └── config.yaml
├── data
│   ├── meta_data.xlsx
│   ├── test
│   └── train
├── dataset.py
├── EDA
│   ├── EDA_channel.ipynb
│   └── EDA_meta.ipynb
├── output # test.py하고 나서 생기는 csv.
├── README.md
├── streamlit
│   ├── main.py
│   ├── pages
│   └── utils
├── test.py
├── trainer.py
├── train.py
```