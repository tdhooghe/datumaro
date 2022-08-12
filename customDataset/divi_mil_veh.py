# %%
import datumaro as dm
from datumaro.components.dataset import Dataset
from datumaro.components.operations import compute_ann_statistics
from utils import convert_to_jpg, create_voc_train_txt, remove_xml_attributes, correct_xml_files, split_dataset
import fiftyone as fo
import fiftyone.brain as fob
from utils import PATH

# %%
convert_to_jpg('/home/thomas/PycharmProjects/datasets/original/data_niels/merged/JPEGImages/')

create_voc_train_txt(f'{PATH}/RAW/data_niels/merged/JPEGImages/')

correct_xml_files(f'{PATH}/RAW/data_niels/Annotations/')

# %%
dutch_mil_veh = Dataset.import_from(f'{PATH}/RAW/data_niels/merged', 'voc')
dutch_mil_veh_stats = compute_ann_statistics(dutch_mil_veh)
# %%
dutch_mil_veh.transform('remap_labels', mapping={'BUSHMASTER': 'military wheeled vehicle',
                                                 'FENNEK': 'military wheeled vehicle',
                                                 'FUCHS': 'military wheeled vehicle',
                                                 'GTK-BOXER': 'military wheeled vehicle',
                                                 'CV90': 'military tracked vehicle',
                                                 'LEOPARD1-AVLB': 'military tracked vehicle',
                                                 'LEOPARD2': 'military tracked vehicle',
                                                 'PZH2000': 'military tracked vehicle',
                                                 },
                        default='delete')
dutch_mil_veh.select(lambda item: len(item.annotations) != 0)
dutch_mil_veh_stats_transform = compute_ann_statistics(dutch_mil_veh)
print(dutch_mil_veh_stats_transform['annotations']['labels']['distribution'])
# print(dutch_mil_veh.subsets())

# %% export to yolo format
dutch_mil_veh.export(f'{PATH}/DMV/cvat/original', 'cvat', save_images=True)

# %%
remove_xml_attributes(f'{PATH}/DMV/cvat/original/labels.xml')
# %%
dmv_dataset = dm.Dataset.import_from(f'{PATH}/DMV/cvat/original', 'cvat')
dmv_dataset_splits_stats = split_dataset(dmv_dataset, f'{PATH}/DMV/cvat/original_splits/', export_type='cvat',
                                         export=True)

# %% Look at data in FiftyOne
# Import the dataset
dataset = fo.Dataset.from_dir(
    dataset_dir=f"{PATH}/DMV/cvat/original",
    dataset_type=fo.types.CVATImageDataset,
)
#%%
session = fo.launch_app(dataset)
#%%
print(dataset.first())
fob.compute_uniqueness(dataset)
#%%
rank_view = dataset.sort_by("uniqueness", reverse=True)
session.view = rank_view
# %%
# military_tracked = dataset.match(F("ground_truth.label") == "military tracked vehicle")
# fob.compute_uniqueness(military_tracked)
#
# similar_military_tracked = military_tracked.sort_by("uniqueness", reverse=False)
# session = fo.launch_app(view=similar_military_tracked)






# %% script to find which images are imported (2018) and which are not
# orig_img = '/home/thomas/PycharmProjects/datasets/original/data_niels/merged/JPEGImages/'
# yolo_img = '/home/thomas/PycharmProjects/datasets/adjusted/dutch_mil_veh/obj_train_data/'
#
#
# def cmp_file_lists(dir1, dir2):
#     dir1_filenames = set(f.name for f in Path(dir1).rglob('*'))
#     dir2_filenames = set(f.name for f in Path(dir2).rglob('*'))
#     files_in_dir1_but_not_dir2 = dir1_filenames - dir2_filenames
#     files_in_dir2_but_not_dir1 = dir2_filenames - dir1_filenames
#     return files_in_dir1_but_not_dir2
#
#
# diff = cmp_file_lists(orig_img, yolo_img)
# diff = list(diff)
# diff.sort()
# with open('/home/thomas/PycharmProjects/datasets/adjusted/dutch_mil_veh/diff.csv', 'w') as f:
#     write = csv.writer(f, quoting=csv.QUOTE_ALL)
#     for item in diff:
#         write.writerow([item])
# %%
