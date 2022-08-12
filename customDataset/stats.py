import datumaro as dm
from datumaro.components.operations import compute_ann_statistics
from datumaro.components.hl_ops import merge
from utils import PATH, label_distributions
#%%
coco_train = dm.Dataset.import_from(f'{PATH}/COCO/cvat/relevant_classes/train', 'cvat')
coco_valid = dm.Dataset.import_from(f'{PATH}/COCO/cvat/relevant_classes/valid', 'cvat')

#%%
relevant_coco = merge(coco_train, coco_valid)
#%%
relevant_coco_stats = compute_ann_statistics(relevant_coco)
print(relevant_coco_stats)

#%%
sohas_stats = compute_ann_statistics(dm.Dataset.import_from(f'{PATH}/SOHAS/cvat/sohas_cvat', 'cvat'))
omv_stats = compute_ann_statistics(dm.Dataset.import_from(f'{PATH}/OCMV/omv/manual_labels/cvat', 'cvat'))
dmv_stats = compute_ann_statistics(dm.Dataset.import_from(f'{PATH}/DMV/cvat/original/', 'cvat'))
rifles = compute_ann_statistics(dm.Dataset.import_from(f'{PATH}/RIFLES/cvat/filtered/', 'cvat'))