# MMSEGMENTATION
- MMSegmentation is an open source semantic segmentation toolbox based on PyTorch. It is a part of the OpenMMLab project.

- The main branch works with PyTorch 1.6+.

[Github](https://github.com/open-mmlab/mmsegmentation)<br>
[Docs](https://mmsegmentation.readthedocs.io/en/latest/)

<br>

## 1. MMSEGMENTATION 기본 사용법

### train
```python
python mmseg_train.py
```
### test
```python
python mmseg_test.py
```

### 공용

- `configs/mmseg_config.yaml`의 값을 받아 사용하지만, `command-line options` 값을 우선적으로 사용하게 됩니다. <br>
  - `--config`, `--model_config`, `--see`, `--size`, `--num_workers`, `--batch_size`, `--epochs`,`...`

<br>



## 2. 하이퍼파라미터 설정 방법
### 2.1. 설정 파일 경로
- `configs/mmseg_config.yaml`에서 하이퍼파라미터를 설정합니다.

### 2.2. 주요 설정 항목
#### Model Config
```yaml
model_config: &model_name model_config.py (mmsegmentation/my_configs 내부에 위치하여야 합니다.)
ex) model_config: &model_name demo_xray.py
```
#### Image Resize
```yaml
image_size: &image_size 숫자
ex) image_size: &image_size 1024
```

#### epochs
```yaml
max_epoch: &max_epoch 숫자 <- 숫자자리에 max_epoch 넣어주시면 됩니다.
ex) max_epoch: &max_epoch 30
```

그외 하이퍼파라미터도 직접 수정해서 적으시면 됩니다.
 
<br>

# 3. model config 작성법
## 3.1 지원하는 모델 찾기
- `mmsegmentation/configs`에서 사용하려는 모델이 있는지 찾습니다.
- 사용하려는 모델에서 어떤 버전을 사용할지 정합니다.

## 3.2 model config 작성하기
- 사용하려는 모델 버전의.py 파일을 확인해보면 `_base_`과 `_base_`에서 어떤 부분이 `overwriting` 되었는지 확인 할 수 있습니다.
- _base_를 추적하면 _base_가 더이상 존재하지 않는 파일들이 있는데, 그곳 내용부터 `my_config/*.py`에 작성해주시고, 역순으로 변경사항을 적용하여 수정해주시면 됩니다.

### 사용 할 버전의 py파일 (ex: swin)
```python
_base_ = [
    'swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py'
]
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_base_patch4_window12_384_20220317-55b0104a.pth'  # noqa
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        pretrain_img_size=384,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12),
    decode_head=dict(in_channels=[128, 256, 512, 1024], num_classes=150),
    auxiliary_head=dict(in_channels=512, num_classes=150))
```
### base file (ex: upernet_swin)
- 기본적으로 최상위 base는 `mmsegmentation/config/_base_` 에 위치합니다.
```python
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
```
<br>

### 학습에 필요한 설정 작성
- model이외에도 `optimizer`, `param_scheduler`, `hook`, `pipeline`, `dataloader`, `evaluator`등을 작성해주시고 `demo_xray.py` 참고하시면 됩니다.
- 1차 적으로 설정값들을 작성해주시고 이후, `mmseg_config.yaml값`으로 일부 값들이 `overwriting` 됩니다.

## 3.3 유의사항
- _base_에서 부터 사용 할 버전의 py파일까지 변경사항을 전부 수정해주시면 됩니다.
- 굳이 base부터 시작하지 않고 _base_ = [...] 을 최상단에 두어도 가능하지만, duplication 문제가 발생하기도해 최상위 base부터 작성하는것을 추천합니다.
- model부에 backbone을 제외한 모든 부분의 type은 아래와 같이 변경해주어야 현재 버전에 문제 없이 돌아갑니다. 변경할 type의 class가 구현되어 있지 않다면, `mmseg/model/decode_heads`에 작성해주시면 됩니다.

```python
type='EncoderDecoder' -> type='EncoderDecoderWithoutArgmax'
type='~~Head' -> type='~~HeadWithoutAccuracy'
```

## 4 참고사항
- Batch size 1은 동작하지 않습니다.
- mmsegmentation/config에 있는 model이 아니여도 mmseg/model/에 있는 backbones, decode_heads등에 구현되어 있는 내용으로 `custom model`도 구현 가능합니다. backbones, decode_heads 또한 없다면 구현하시면 됩니다.
- dataset, loss, ..evaluation/metrics 또한 구현 가능하며, mmsegmentation의 config가 아닌 mmseg내부에서 작성해주시면 됩니다. 

# 5. 폴더 구조
```bash
mmsegmentation
|-- configs
|   |-- _base_
|   |-- many models... ⭐
|-- demo
|-- docker
|-- docs
|-- mmseg
|   |-- __pycache__
|   |-- apis
|   |-- configs        
|   |-- datasets       ⭐
|   |-- engine
|   |-- evaluation     ⭐
|   |-- models         ⭐
|   |-- registry       ⭐
|   |-- structures
|   |-- utils
|   `-- visualization
|-- mmsegmentation.egg-info
|-- my_configs         ⭐
|-- projects
|-- requirements
|-- resources
|-- tests
`-- tools
mmseg_results          ⭐
mmseg_train.py         ⭐
mmseg_test.py          ⭐


```