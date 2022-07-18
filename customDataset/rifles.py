import pathlib
import os

from datumaro.components.operations import compute_ann_statistics

from utils import move_files, PATH, split_dataset
import datumaro as dm


#%% extract all images from different folders and put them in one folder
path = "D:/Documents/thomas/rifles_thomas/"
# sources = os.listdir(path)

sub_folders = [f.path for f in os.scandir(path) if f.is_dir()]

for sub_folder in sub_folders:
    move_files(pathlib.Path(sub_folder))

#%%
path = f'{PATH}/RIFLES'
#%%
all_rifles_dataset = dm.Dataset.import_from(f'{path}/cvat/all_images', 'cvat')

all_rifles_dataset_stats = compute_ann_statistics(all_rifles_dataset)
#%%
all_rifles_dataset.select(lambda item: len(item.annotations) != 0)
filtered_rifles_dataset = compute_ann_statistics(all_rifles_dataset)

#%%
all_rifles_dataset.export(f'{path}/cvat/filtered', 'cvat', save_images=True)

#%%
filtered_rifles = dm.Dataset.import_from(f'{path}/cvat/filtered')
filtered_rifles_auto_labels_stats = compute_ann_statistics(filtered_rifles)

#%% create and export dataset splits for detection
export_type = 'yolo'
if export_type == 'yolo':
    output_path = f'{path}/yolo'
elif export_type == 'cvat':
    output_path = f'{path}/cvat'
else:
    output_path = f'{path}/other'

split_stats = split_dataset(filtered_rifles, output_path=f'{output_path}/autolabel_filtered_rifles_splits',
                            export=True, export_type=export_type)
