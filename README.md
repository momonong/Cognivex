# semantic-KG

```shell

poetry run python -m scripts.rename_nii
```

```shell
poetry run python -m scripts.convert_image \
    --input_dir data/packages \
    --output_dir data/images \
    --drop_slices 10 \
    --verbose
```

```shell
# optional
poetry run python -m scripts.populate_fake_nc
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
