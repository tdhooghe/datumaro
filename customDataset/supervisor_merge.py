# %%
from utils import PATH, merge_datasets, split_dataset

import datumaro as dm
from datumaro.plugins import splitter
from datumaro import Dataset
from datumaro.components.operations import compute_ann_statistics, IntersectMerge, ExactMerge

# %% input locations datasets
sohas_thres = f'{PATH}/SOHAS/cvat/sohas_all_splits_thres'
rifles_thres = f'{PATH}/RIFLES/cvat/rifle_all_splits_thres'
relevant_coco = f'{PATH}/COCO/cvat/relevant_classes_splits'

# %% merge guns and coco without military vehicles
no_mil_vehicles = {
    'military wheeled vehicle': '',
    'military tracked vehicle': ''
}

inputs = [sohas_thres, rifles_thres, relevant_coco]
# %% note mapping_default in function
guns_coco_055 = merge_datasets(inputs, output=f'{PATH}/MERGED/supervisor/cvat/guns_coco_055_excl_mil',
                               import_type='cvat', export=True, export_type='cvat', mapping=no_mil_vehicles,
                               mapping_default='keep')
# %%
guns_coco_021 = merge_datasets(inputs, output=f'{PATH}/MERGED/supervisor/yolo/guns_coco_021_excl_mil',
                               import_type='cvat', export=True, export_type='yolo', mapping=no_mil_vehicles,
                               mapping_default='keep')
# %% export to yolo format
_ = merge_datasets(inputs, output=f'{PATH}/MERGED/supervisor/yolo/guns_coco_055_excl_mil', import_type='cvat',
                   export=True, export_type='yolo', mapping=no_mil_vehicles, mapping_default='keep')

# %% remove pictures from test set without labels
test = dm.Dataset.import_from(f'{PATH}/MILVEH/cvat/milveh_all_splits_055/test')
test.select(lambda item: len(item.annotations) != 0)
test.export(f'{PATH}/MILVEH/cvat/milveh_all_splits_055/test2', 'cvat', save_images=True)

# %% merge guns, military vehicles and coco datasets
milveh_thres = f'{PATH}/MILVEH/cvat/milveh_all_splits_thres'
inputs = [milveh_thres, sohas_thres, rifles_thres, relevant_coco]
guns_milveh_coco_055 = merge_datasets(inputs, output=f'{PATH}/MERGED/supervisor/yolo/guns_milveh_coco_021',
                                      import_type='cvat', export=True, export_type='yolo')
