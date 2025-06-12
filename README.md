# semantic-KG
先把 0515_GE 的資料夾放進 data 然後改名成 raw

```shell

poetry run python -m scripts.data_prepare
```

```shell
poetry run python -m scripts.train
```

```shell
poetry run python -m scripts.inference \
    --image data/images/AD/AD_sub-AD_027_S_6648_task-rest_bold.nii_z001_t003.png \
    --weights model/mcadnnet_mps.pth \
    --extract-activation \
    --activation-output activation
```

### actvation 視覺化順序
```shell
poetry run python -m scripts.group.infer
poetry run python -m scripts.group.acc_nii
poetry run python -m scripts.group.resample
poetry run python -m scripts.group.get_avg_map # optional
poetry run python -m scripts.group.brain_map
poetry run python -m scripts.group.check_map
```