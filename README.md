# semantic-KG

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