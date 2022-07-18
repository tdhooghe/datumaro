# %%
import glob
import os

import datumaro as dm

from datumaro.components.operations import compute_ann_statistics
from utils import Sampler, images_paths_to_text, remove_segmentations, PATH, split_dataset, label_dataset

# dataset consists of military trucks, military tanks, military aircraft, military helicopters, civilian cars
# and civilian aircraft


# %% script to create yolo like train.txt file based on images in obj_train_data, to be able to import in datumaro
images_paths_to_text('../datasets/open_civ_mil_vehicles/')

# %% load open source data military trucks, tanks, aircraft, helicopters, and civilian cars and aircraft
civ_mil_dataset = dm.Dataset.import_from(f'{PATH}/OCMV/full/yolo', 'yolo')

# %%
civ_mil_dataset.export(f'{PATH}/OCMV/cvat', 'cvat', save_images=True)

#%%
civ_mil_dataset_cvat = dm.Dataset.import_from(f'{PATH}/OCMV/full/cvat', 'cvat')

#%%
labels = {'military truck': 'military wheeled vehicle', 'military tank': 'military tracked vehicle'}

#%%
omv_dataset, distribution, stats = label_dataset(civ_mil_dataset_cvat, labels)

# %%
omv_dataset.export(f'{PATH}/OCMV/omv/cvat', 'cvat', save_images=True)

#%%
omv_yolo, full_dist, full_stats = label_dataset(civ_mil_dataset_cvat, labels)
omv_yolo.export(f'{PATH}/OCMV/omv/yolo', 'yolo', save_images=True)

#%%
images_paths_to_text(f'{PATH}/OCMV/omv/manual/yolo/obj_train_data')
omv_yolo_preselect = dm.Dataset.import_from(f'{PATH}/OCMV/omv/preselection/yolo', 'yolo')
omv_yolo_preselect.export(f'{PATH}/OCMV/omv/preselection/cvat', 'cvat', save_images=True)

#%%
omv_manual_labels = dm.Dataset.import_from(f'{PATH}/OCMV/omv/preselection/cvat', 'cvat')
omv_manual_labels.select(lambda item: len(item.annotations) != 0)
omv_manual_labels_stats = compute_ann_statistics(omv_manual_labels)
omv_manual_labels.export(f'{PATH}/OCMV/omv/manual/cvat', 'cvat', save_images=True)

