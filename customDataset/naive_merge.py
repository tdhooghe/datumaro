# %%
from utils import PATH, merge_datasets, split_dataset

import datumaro as dm
from datumaro.plugins import splitter
from datumaro import Dataset
from datumaro.components.operations import compute_ann_statistics, IntersectMerge, ExactMerge

# %% create coco sets (test especially)
coco_relevant = dm.Dataset.import_from(f'{PATH}/COCO/cvat/relevant_classes_splits/train', 'cvat')
split_dataset(coco_relevant, name='test', output_path=f'{PATH}/COCO/yolo/relevant_classes_splits', export_type='yolo')
#%%
coco_relevant = dm.Dataset.import_from(f'{PATH}/COCO/cvat/relevant_classes_splits/test', 'cvat')
coco_relevant.export(f'{PATH}/COCO/yolo/relevant_classes_splits/test', format='yolo', save_images=True)

# %% create sohas sets
sohas_auto_train = dm.Dataset.import_from(f'{PATH}/SOHAS/cvat/sohas_auto_splits/train/', 'cvat')
sohas_no_auto_train = sohas_auto_train.transform('remap_labels', mapping={
    'knife': 'knife',
    'pistol': 'pistol',
    'cell phone': 'cell phone'}, default='delete')
sohas_no_auto_train.select(lambda item: len(item.annotations) != 0)
split_dataset(sohas_no_auto_train, name='test', output_path=f'{PATH}/SOHAS/yolo/no_auto_splits/',
              export_type='yolo')

sohas_auto_valid = dm.Dataset.import_from(f'/home/thomas/Datasets/SOHAS/cvat/sohas_auto_splits/valid/', 'cvat')
sohas_auto_valid.select(lambda item: len(item.annotations) != 0)
sohas_auto_valid.export(f'{PATH}/SOHAS/yolo/no_auto_splits/valid', 'yolo', save_images=True)

# %% create rifle sets
rifles_auto_train = dm.Dataset.import_from(f'{PATH}/RIFLES/cvat/autolabel_filtered_rifles_splits/train/', 'cvat')
rifles_no_auto_train = rifles_auto_train.transform('remap_labels', mapping={
    'rifle': 'rifle'})
rifles_no_auto_train.select(lambda item: len(item.annotations) != 0)
split_dataset(rifles_no_auto_train, name='test', output_path=f'{PATH}/RIFLES/yolo/no_auto_splits/', export_type='yolo')

rifles_valid = dm.Dataset.import_from(f'{PATH}/RIFLES/cvat/autolabel_filtered_rifles_splits/valid/', 'cvat')
rifles_valid.select(lambda item: len(item.annotations) != 0)
rifles_valid.export(f'{PATH}/RIFLES/yolo/no_auto_splits/valid', 'yolo', save_images=True)

# %% create military vehicle sets
mil_veh_train = dm.Dataset.import_from(f'{PATH}/MERGED/supervisor/omv_dmv_cvat/train/', 'cvat')
# mil_veh_train = mil_veh_train.transform('remap_labels', mapping={
#     'military wheeled vehicle': 'military wheeled vehicle',
#     'military tracked vehicle': 'military tracked vehicle'})
mil_veh_train.select(lambda item: len(item.annotations) != 0)
split_dataset(mil_veh_train, name='test', output_path=f'{PATH}/MERGED/naive/cvat/omv_dmv_test3/', export_type='cvat')

#%%
mil_veh_valid = dm.Dataset.import_from(f'{PATH}/MERGED/supervisor/omv_dmv_cvat/valid', 'cvat')
mil_veh_valid.select(lambda item: len(item.annotations) != 0)  # remove images with no labels
mil_veh_valid.export(f'{PATH}/MERGED/naive/cvat/omv_dmv/valid', 'cvat', save_images=True)

#%%
coco_dataset = f'{PATH}/COCO/cvat/relevant_classes_splits'
coco_sampled_dataset = f'{PATH}/COCO/cvat/relevant_sampled_splits'
sohas_no_auto = f'{PATH}/SOHAS/cvat/no_auto_splits'
rifles_no_auto = f'{PATH}/RIFLES/cvat/no_auto_splits'
mil_veh_no_auto = f'{PATH}/MERGED/naive/cvat/omv_dmv'
# rifles_sohas = f'{PATH}/MERGED/naive/yolo/rifles_sohas'
# rifles_sohas_vehicles = f'{PATH}/MERGED/naive/yolo/rifles_sohas_vehicles'

#%%
inputs = [sohas_no_auto, rifles_no_auto, mil_veh_no_auto, coco_dataset]
_ = merge_datasets(inputs, output=f'{PATH}/FINAL/naive/yolo/naive_merge_nococo', import_type='cvat',
                   export=True, export_type='yolo')
#%%
no_coco_mapping = {
    'military wheeled vehicle': 'military wheeled vehicle',
    'military tracked vehicle': 'military tracked vehicle',
    'rifle': 'rifle',
    'pistol': 'pistol',
    'knife': 'knife',
    'cell phone': 'cell phone'
}
no_coco_inputs = [sohas_no_auto, rifles_no_auto, mil_veh_no_auto]
no_coco_merge = merge_datasets(no_coco_inputs, output=f'{PATH}/FINAL/naive/yolo/naive_merge_nococo', import_type='cvat',
                   export=True, export_type='yolo', mapping=no_coco_mapping)
#%%
dm.Dataset.import_from(f'{PATH}/MERGED/naive/cvat/omv_dmv/train', 'cvat').export(f'{PATH}/MERGED/naive/yolo/omv_dmv/train', 'yolo')

#%% final merge
sohas = f'{PATH}/SOHAS/cvat/sohas_all_splits'
rifles = f'{PATH}/RIFLES/cvat/rifle_all_splits'
relevant_coco = f'{PATH}/COCO/cvat/relevant_classes_splits'
milveh = f'{PATH}/MILVEH/cvat/milveh_all_splits'

naive_inputs = inputs = [milveh, sohas, rifles, relevant_coco]

naive_merge = merge_datasets(naive_inputs, output=f'{PATH}/MERGED/naive/yolo/naive_merge_final', import_type='cvat',
                   export=True, export_type='yolo')





