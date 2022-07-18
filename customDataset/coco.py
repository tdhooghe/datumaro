# %%
from utils import remove_segmentations, Sampler, split_dataset, PATH, SEED
import datumaro as dm
from datumaro.components.operations import compute_ann_statistics
from datumaro.plugins.sampler.random_sampler import LabelRandomSampler

# %% remove segmentations, as otherwise datumaro imports polygons instead of bounding boxes
remove_segmentations('../datasets/coco/annotations/instances_val2017.json')
remove_segmentations('../datasets/coco/annotations/instances_train2017.json')

# %% import original coco dataset
coco_dataset = dm.Dataset.import_from('../datasets/coco/', 'coco')
coco_dataset_stats = compute_ann_statistics(coco_dataset)
# %% include vehicle classes and remove background images
coco_dataset.transform('remap_labels', mapping={
    'person': 'person',
    'car': 'car',
    'motorcycle': 'motorcycle',
    'bus': 'bus',
    'truck': 'truck',
    'knife': 'knife',
    'cell phone': 'cell phone'},
                       default='delete')

coco_dataset.select(lambda item: len(item.annotations) != 0)
coco_transform_stats = compute_ann_statistics(coco_dataset)
# %%
coco_split_stats = split_dataset(coco_dataset, output_path=f'{PATH}/COCO/cvat/relevant_classes', export_type='cvat')

#%%
coco_dataset.export(f'{PATH}/COCO/cvat/relevant_classes', 'cvat', save_images=True)

# %%
coco_dataset = dm.Dataset.import_from(f'{PATH}/COCO/cvat/relevant_classes')
coco_relevant_stats = compute_ann_statistics(coco_dataset)
# seed = 1234
sampled_coco = coco_dataset.transform(LabelRandomSampler, count=6, label_counts={'person': 1,
                                                                                 'car': 1,
                                                                                 'motorcycle': 9000,
                                                                                 'bus': 6000,
                                                                                 'truck': 10000,
                                                                                 'knife': 8000,
                                                                                 'cell phone': 6500
                                                                                 },
                                      seed=SEED
                                      )
coco_dataset_sampled = compute_ann_statistics(sampled_coco)

# %%
relevant_sampled_stats = split_dataset(sampled_coco, output_path=f'{PATH}/COCO/yolo/relevant_sampled_split', export=True
                                       )
# coco_dataset.transform(Sampler, obj_class='person', num_samples=50000, seed=1234)
# coco_sampled_stats = compute_ann_statistics(coco_dataset)

# %%
stats = split_dataset(coco_dataset, output_path=f'{PATH}/COCO/yolo/relevant_sample_splits', export=True)
