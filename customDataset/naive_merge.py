# %%
from glob import glob
from utils import merge_datasets, PATH, merge_datasets, split_dataset, label_distributions
import datumaro as dm
from datumaro.components.operations import compute_ann_statistics, IntersectMerge, ExactMerge

# %% create coco sets (test especially)
coco_relevant = dm.Dataset.import_from(f'{PATH}/COCO/cvat/relevant_classes/train', 'cvat')
split_dataset(coco_relevant, name='test', output_path=f'{PATH}/COCO/cvat/relevant_classes_splits', export_type='cvat')

# %% create sohas sets
sohas_auto_train = dm.Dataset.import_from(f'{PATH}/SOHAS/cvat/sohas_auto_splits/train/', 'cvat')
sohas_no_auto_train = sohas_auto_train.transform('remap_labels', mapping={
    'knife': 'knife',
    'pistol': 'pistol',
    'cell phone': 'cell phone'}, default='delete')
# sohas_no_auto_train.transform('remap_labels', mapping={
#     'person': 'person',
#     'car': 'car',
#     'motorcycle': 'motorcycle',
#     'bus': 'bus',
#     'truck': 'truck',
#     'military wheeled vehicle': 'military wheeled vehicle',
#     'military tracked vehicle': 'military wheeled vehicle',
#     'rifle': 'rifle',
#     'knife': 'knife',
#     'pistol': 'pistol',
#     'cell phone': 'cell phone'}, default='keep')
sohas_no_auto_train.select(lambda item: len(item.annotations) != 0)
split_dataset(sohas_no_auto_train, name='test', output_path=f'{PATH}/SOHAS/yolo/sohas_no_auto_splits/',
              export_type='yolo')

sohas_auto_valid = dm.Dataset.import_from(f'/home/thomas/Datasets/SOHAS/cvat/sohas_auto_splits/valid/', 'cvat')
sohas_auto_valid.export(f'{PATH}/SOHAS/yolo/sohas_no_auto_splits/valid', 'yolo', save_images=True)

# %% create rifle sets
rifles_auto_train = dm.Dataset.import_from(f'{PATH}/RIFLES/cvat/autolabel_filtered_rifles_splits/train/', 'cvat')
rifles_no_auto_train = rifles_auto_train.transform('remap_labels', mapping={
    'rifle': 'rifle'})
rifles_no_auto_train.select(lambda item: len(item.annotations) != 0)
split_dataset(rifles_no_auto_train, name='test', output_path=f'{PATH}/RIFLES/yolo/no_auto_splits/', export_type='yolo')

rifles_valid = dm.Dataset.import_from(f'{PATH}/RIFLES/cvat/autolabel_filtered_rifles_splits/valid/', 'cvat')
rifles_valid.export(f'{PATH}/RIFLES/yolo/no_auto_splits/valid', 'yolo', save_images=True)

# %% create military vehicle sets
mil_veh_train = dm.Dataset.import_from(f'{PATH}/MERGED/supervisor/omv_dmv_cvat/train/', 'cvat')
mil_veh_train = mil_veh_train.transform('remap_labels', mapping={
    'military wheeled vehicle': 'military wheeled vehicle',
    'military tracked vehicle': 'military tracked vehicle'})
mil_veh_train.select(lambda item: len(item.annotations) != 0)
split_dataset(mil_veh_train, name='test', output_path=f'{PATH}/MERGED/naive/yolo/omv_dmv/', export_type='yolo')

mil_veh_valid = dm.Dataset.import_from(f'{PATH}/MERGED/supervisor/omv_dmv_cvat/valid', 'cvat')
mil_veh_valid = mil_veh_valid.transform('remap_labels', mapping={
    'military wheeled vehicle': 'military wheeled vehicle',
    'military tracked vehicle': 'military tracked vehicle'})
mil_veh_valid.select(lambda item: len(item.annotations) != 0)
mil_veh_valid.export(f'{PATH}/MERGED/naive/yolo/omv_dmv_cvat/valid', 'yolo', save_images=True)

coco_dataset = f'{PATH}/COCO/cvat/relevant_classes'
sohas_no_auto = f'{PATH}/SOHAS/yolo/sohas_no_auto_splits'
rifles_no_auto = f'{PATH}/RIFLES/yolo/no_auto_splits'
mil_veh_no_auto = f'{PATH}/MERGED/naive/yolo/omv_dmv'
rifles_sohas = f'{PATH}/MERGED/naive/yolo/rifles_sohas'
rifles_sohas_vehicles = f'{PATH}/MERGED/naive/yolo/rifles_sohas_vehicles'

_ = merge_datasets(sohas_no_auto, rifles_no_auto, output=f'{PATH}/MERGED/naive/yolo/rifles_sohas',
                   export=True, export_type='yolo')

#%%

_ = merge_datasets(mil_veh_no_auto, rifles_sohas, output=f'{PATH}/MERGED/naive/yolo/rifles_sohas_vehicles',
                   export=True, export_type='yolo')
#%%
_ = merge_datasets(rifles_sohas_vehicles, coco_dataset, output=f'{PATH}/MERGED/naive/cvat/final_all_coco', export=True,
                   export_type='cvat')

#%%
rifles_sohas = dm.Dataset.import_from(f'{PATH}/MERGED/naive/cvat/rifles_sohas_cvat/valid', 'cvat')
omv_dmv = dm.Dataset.import_from(f'{PATH}/MERGED/naive/cvat/omv_dmv_cvat/valid', 'cvat')
test = IntersectMerge()([rifles_sohas, omv_dmv])
test.export(f'{PATH}/MERGED/naive/cvat/test/valid)','cvat',save_images=True)