# %%
import datumaro as dm
from utils import split_dataset, PATH
from datumaro.components.operations import compute_ann_statistics

# images_paths_to_text('/home/thomas/Datasets/sohas_yolo/val')
# images_paths_to_text('/home/thomas/Datasets/sohas_yolo/train')
path = f'{PATH}/SOHAS'
#%%
sohas_dataset = dm.Dataset.import_from(f'{path}/voc/sohas_voc', 'voc')

sohas_dataset_stats = compute_ann_statistics(sohas_dataset)
#%%

sohas_dataset.transform('remap_labels', mapping={'pistol': 'pistol',
                                                 'knife': 'knife',
                                                 'smartphone': 'cell phone'
                                                 },
                        default='delete')
sohas_dataset.select(lambda item: len(item.annotations) != 0)
sohas_transform_stats = compute_ann_statistics(sohas_dataset)

#%%
sohas_dataset.export(f'{path}/cvat/sohas_cvat', 'cvat', save_images=True)

#%%
sohas_auto_labels = dm.Dataset.import_from(f'{path}/cvat/sohas_auto_labels')
sohas_auto_labels_stats = compute_ann_statistics(sohas_auto_labels)

#%% create and export dataset splits for detection
split_stats = split_dataset(sohas_auto_labels, output_path=f'{path}/SOHAS/cvat/sohas_auto_splits', export=True)

#%% remove autolabels
sohas_auto_labels = dm.Dataset.import_from(f'{path}/cvat/sohas_auto_labels')

sohas_no_auto = sohas_auto_labels.transform('remap_labels', mapping={
    'knife': 'knife',
    'pistol': 'pistol',
    'cell phone': 'cell phone'},
                       )
sohas_no_auto.select(lambda item: len(item.annotations) != 0)
split_stats_no_auto = split_dataset(sohas_no_auto, output_path=f'{path}/yolo/sohas_no_auto', export_type='yolo')



