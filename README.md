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
