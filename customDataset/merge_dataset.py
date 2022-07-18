# %%
from utils import intersect_merge_datasets, PATH, merge_datasets, split_dataset
import datumaro as dm
from datumaro.components.operations import compute_ann_statistics, IntersectMerge, ExactMerge

# %% merge coco and sohas (A)
relevant_coco_samples = f'{PATH}/COCO/cvat/relevant_allsample_splits/'
sohas_auto_labels = f'{PATH}/SOHAS/cvat/sohas_auto_splits/'
# Relevant COCO Samples and Sohas Auto Labels
A_output_directory = f'{PATH}/MERGED/dataset_a_yolo'

A_train_stats, A_valid_stats = intersect_merge_datasets(sohas_auto_labels, relevant_coco_samples, A_output_directory,
                                                        export=True, export_type='yolo')

# %% merge A and rifles (B) duplicate image names DSC_000069.jpg
A = f'{PATH}/MERGED/dataset_a_yolo'
rifles_auto_labels = f'{PATH}/RIFLES/yolo/autolabel_filtered_rifles_splits/'
B_output_directory = f'{PATH}/MERGED/dataset_b'

B_train_stats, B_valid_stats = merge_datasets(A, rifles_auto_labels, B_output_directory, export=True,
                                              export_type='cvat')

# %%
rifles_dataset = dm.Dataset.import_from(rifles_auto_labels, 'cvat')
dataset_a = dm.Dataset.import_from(A, 'cvat')
# %%
rifles_stats = compute_ann_statistics(rifles_dataset)
dataset_a_stats = compute_ann_statistics(dataset_a)

# %%
print(rifles_stats['annotations']['labels']['distribution'])
print(dataset_a_stats['annotations']['labels']['distribution'])

# %%
dmv_dataset = dm.Dataset.import_from(f'{PATH}/DMV/cvat/original', 'cvat')
dmv_dataset_stats = compute_ann_statistics(dmv_dataset)
print(dmv_dataset_stats['annotations']['labels']['distribution'])



# %%
omv_dataset = dm.Dataset.import_from(f'{PATH}/OCMV/omv/manual_labels_incl_persons/cvat/', 'cvat')
omv_dataset_stats = compute_ann_statistics(omv_dataset)
print(omv_dataset_stats['annotations']['labels']['distribution'])
omv_dataset.transform('remap_labels', mapping={
    'military wheeled vehicle': 'military wheeled vehicle',
    'military tracked vehicle': 'military tracked vehicle'},
                      default='delete')

omv_dataset.select(lambda item: len(item.annotations) != 0)
omv_dataset_transform_stats = compute_ann_statistics(omv_dataset)
print(omv_dataset_transform_stats['annotations']['labels']['distribution'])

# %%
# omv_dataset_splits_stats = split_dataset(omv_dataset, f'{PATH}/OCMV/omv/manual_labels_splits/cvat', export_type='cvat',
# export=True)
print(omv_dataset_transform_stats['annotations']['labels']['distribution'])
print(dmv_dataset_stats['annotations']['labels']['distribution'])

# %%
omv_splits_path = f'{PATH}/OCMV/omv/manual_labels_splits/cvat'
dmv_splits_path = f'{PATH}/DMV/cvat/original_splits'

print(compute_ann_statistics(dm.Dataset.import_from(omv_splits_path, 'cvat'))['annotations']['labels']['distribution'])
print(compute_ann_statistics(dm.Dataset.import_from(dmv_splits_path, 'cvat'))['annotations']['labels']['distribution'])

# %%
omv_dmv_train_merged_stats, omv_dmv_valid_merged_stats = \
    merge_datasets(input1=omv_splits_path, input2=dmv_splits_path, output=f'{PATH}/MERGED/omv_dmv', export=True,
                   export_type='cvat')
