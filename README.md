# semantic-KG

```
poetry run python -m scripts.rename_nii
```

```
poetry run python -m scripts.convert_image \
    --input_dir data/packages \
    --output_dir data/images \
    --drop_slices 10 \
    --verbose
```

```
mkdir -p data/images/NC
```

```
poetry run python -m scripts.populate_fake_nc
```

```
poetry run python -m scripts.train
```

```
poetry run python -m scripts.inference \
    --image data/images/AD/sub-01_AD_task-rest_bold.nii_001_001.png \
    --weights model/mcadnnet_mps.pth
```
