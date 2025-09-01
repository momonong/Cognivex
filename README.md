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
poetry run python -m scripts.group.act_nii
poetry run python -m scripts.group.resample
poetry run python -m scripts.group.get_avg_act # optional
poetry run python -m scripts.group.brain_map
poetry run python -m scripts.group.check_map
```

C. -N. Jiao et al., "Diagnosis-Guided Deep Subspace Clustering Association Study for Pathogenetic Markers Identification of Alzheimer's Disease Based on Comparative Atlases," in IEEE Journal of Biomedical and Health Informatics, vol. 28, no. 5, pp. 3029-3041, May 2024, doi: 10.1109/JBHI.2024.3372294.
keywords: {Genetics;Brain;Diseases;Functional magnetic resonance imaging;Correlation;Neuroimaging;Feature extraction;Brain imaging genetics;deep subspace clustering;functional connectivity network;diagnosis information;sparse canonical correlation analysis},
