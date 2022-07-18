#%%
from datumaro.components.operations import compute_ann_statistics
from utils import PATH, merge_datasets
import datumaro as dm
#%%
pistols_path = f'{PATH}/SOHAS/cvat/sohas_auto_splits/'
rifles_path = f'{PATH}/RIFLES/cvat/autolabel_filtered_rifles_splits/'

train, valid = merge_datasets(pistols_path, rifles_path)

#%%
for x in [train, valid]:
    x.transform('remap_labels', mapping={
        'rifle': 'rifle',
        'pistol': 'pistol',
        'knife': 'knife',
        'cell phone': 'cell phone'},
        default='delete')
    x.select(lambda item: len(item.annotations) != 0)
    print(compute_ann_statistics(x)['annotations']['labels']['distribution'])

#%%
train.export(f'{PATH}/MERGED/rifles_sohas_only/train', format='yolo', save_images=True)
valid.export(f'{PATH}/MERGED/rifles_sohas_only/valid', format='yolo', save_images=True)
