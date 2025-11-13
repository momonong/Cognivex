from nilearn import datasets

aal = datasets.fetch_atlas_aal(version='SPM12')
region_names = aal['labels']

for i, name in enumerate(region_names):
    print(f"{i+1}: {name}")
