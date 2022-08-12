# %%
import json
import os
import pathlib
import shutil
import glob
import random
import datumaro as dm

from datumaro import Dataset
from datumaro.components.extractor import Transform, AnnotationType
from datumaro.components.operations import compute_ann_statistics, IntersectMerge
from datumaro.components.hl_ops import merge
from datumaro.plugins import splitter

PATH = "../../Datasets"
SEED = 1234


class Sampler(Transform):
    def __init__(self, extractor, obj_class, num_samples, seed=1234):
        super().__init__(extractor)
        self._obj_class = obj_class
        self._num_samples = num_samples
        self._seed = seed

    def __iter__(self):
        annotations = 0
        required_quantity = self._num_samples
        person_label_idx = self._extractor.categories()[AnnotationType.label].find(self._obj_class)[0]

        random.seed(self._seed)
        items = random.sample(list(self._extractor), len(list(self._extractor)))

        for item in items:
            new_anns = []
            for ann in item.annotations:
                if hasattr(ann, 'label') and ann.label == person_label_idx:
                    if annotations >= required_quantity:
                        continue
                    else:
                        annotations += 1

                new_anns.append(ann)
            if new_anns:
                yield item.wrap(annotations=new_anns)


def label_distributions(stats=None, dataset=None):
    if dataset:
        stats = compute_ann_statistics(dataset)
    return stats['annotations']['labels']['distribution'], stats


def label_dataset(dataset, labels: dict):
    dataset.transform('remap_labels', mapping=labels, default='delete')
    dataset.select(lambda item: len(item.annotations) != 0)
    distribution, stats = label_distributions(dataset=dataset)
    return dataset, distribution, stats


def split_dataset(dataset, name='valid', output_path=None, train_size=0.9, valid_size=0.1, seed=1234, export=True,
                  export_type='yolo'):
    splits = splitter.Split(dataset=dataset, task='detection',
                            splits=[('train', train_size), (name, valid_size)], seed=seed)
    splits_data = dict()
    splits_stats = dict()
    for split in splits.subsets().keys():
        subset = splits.get_subset(split)
        splits_data[split] = Dataset(subset)
        splits_stats[split] = compute_ann_statistics(subset)
        print(f'{str(split)}: {label_distributions(splits_stats[split])}')

        if export:
            splits_data[split].export(f'{output_path}/{split}', format=export_type, save_images=True)

    return splits_stats


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def merge_datasets(inputs, output=None, export=False, import_type='cvat', export_type='yolo', simple=False,
                   mapping=None, mapping_default='delete'):
    """
    Function that merges train and valid set of two different datasets
    :param inputs:
    :param output:
    :param export:
    :param import_type:
    :param export_type:
    :param simple: default False, True if simple merge is desired
    :param mapping:
    :return:
    """
    # import datasets
    output = f'{output}_cvat' if export_type == 'cvat' else output
    merged_datasets = dict()
    sets = get_immediate_subdirectories(inputs[0])
    for set in sets:
        subsets = []
        for input in inputs:
            subsets.append(dm.Dataset.import_from(f'{input}/{set}', format=import_type))
        if simple:
            merged_dataset = merge(subsets)
        else:
            merged_dataset = IntersectMerge()(subsets)
        merged_datasets[set] = merged_dataset
        stats = compute_ann_statistics(merged_dataset)
        print(f'{set}_stats: {label_distributions(stats)}')

        if export:
            dataset = Dataset(merged_dataset)
            if mapping is not None:
                dataset.transform('remap_labels', mapping=mapping, default='keep')
            dataset.export(f'{output}/{set}', export_type, save_images=True)

    return merged_datasets


def move_files(source_folder: pathlib.Path):
    folder_name = 'extract'
    target_folder = source_folder.joinpath(folder_name)
    target_folder.mkdir(parents=True, exist_ok=True)
    for image_file in source_folder.rglob("*.jpg"):  # recursively find image paths
        if folder_name not in str(image_file):
            shutil.copy(image_file, target_folder.joinpath(image_file.name))


def images_paths_to_text(dataset_split_path: str):
    # list all jpg files in the path
    images = glob.glob(dataset_split_path + '/*.jpg')
    path = pathlib.Path(dataset_split_path)
    child = path.parts[-1]
    parent = path.parent
    file_path = f'{pathlib.Path(str(parent))}/train.txt'

    # write these file into a text file
    with open(file_path, 'w') as tmp:
        for image in images:
            _, image_path = os.path.split(image)
            tmp.write(f'obj_train_data/{image_path}\n')


def remove_segmentations(path: str):
    """
    Function that removes segmentations from coco json files
    :param path:
    :return:
    """
    with open(path) as f:
        d = json.load(f)

    for ann in d['annotations']:
        ann["segmentation"] = []

    with open(path, 'w') as f:
        json.dump(d, f)


# %% script to convert all images in folder to .jpg
def convert_to_jpg(img_path: str):
    """
    Convert all images to lowercase .jpg format such that datumaro is able to recognize the images
    :param img_path:
    :return:
    """
    num_all_files = len(os.listdir(img_path))
    pngs = glob.glob(img_path + '*.png')
    jpegs = glob.glob(img_path + '*.jpeg')
    jpgs_uppercase = glob.glob(img_path + '*.JPG')
    print(jpgs_uppercase)
    for png in pngs:
        im = Image.open(png)
        im.save(os.path.splitext(png)[0] + '.jpg')
        os.remove(png)

    for jpeg in jpegs:
        os.rename(jpeg, os.path.splitext(jpeg)[0] + '.jpg')
        print(jpeg)

    for JPG in jpgs_uppercase:
        os.rename(JPG, os.path.splitext(JPG)[0] + '.jpg')

    num_jpg_files = len(glob.glob(img_path + '*.jpg'))

    if num_all_files == num_jpg_files:
        print(f'All files are .jpg: {len(jpgs_uppercase)} .JPGS, {len(pngs)} pngs and {len(jpegs)} jpegs converted')
    else:
        print(f'{num_all_files - num_jpg_files} are not converted, check the extensions')


def create_voc_train_txt(img_path):
    """
    Function to put all file names without extension from a certain path in train.txt file,
    which is required for the VOC format
    :param img_path:
    :return:
    """
    default_wd = os.getcwd()
    # change working directory to the image directory
    os.chdir(img_path)
    # list all jpg files in the changed working directory
    files = glob.glob('*.jpg')
    files.sort()

    # write these files without extension into a text file
    file_name = '../ImageSets/Main/train.txt'
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'w') as tmp:
        for file in files[:-1]:
            no_ext = os.path.splitext(file)[0]
            tmp.write(no_ext + '\n')
        tmp.write(os.path.splitext(files[-1])[0])

    os.chdir(default_wd)

    if len(files) == len(os.listdir(img_path)):
        print('All files are put in train.txt')
    else:
        print(f'{len(files)} of {len(os.listdir(img_path))} files are put in train.txt, check the extensions')


def correct_xml_files(ann_path: str):
    """
    Recover right paths in folder containing xml files
    :param ann_path:
    :return:
    """

    path = ann_path
    classes = ['BUSHMASTER', 'CV90', 'FENNEK', 'FUCHS', 'GTK-BOXER', 'LEOPARD1-AVLB', 'LEOPARD2', 'PZH2000']
    file_list = os.listdir(path)
    file_list.sort()
    for file in file_list:
        if not file.endswith('.xml'):
            continue
        tree = ET.parse(path + file)
        root = tree.getroot()
        for folder in root.iter('folder'):
            new_folder = 'merged'
            folder.text = str(new_folder)
        for filename in root.iter('filename'):
            new_filename = str(os.path.splitext(filename.text)[0] + '.jpg')
            filename.text = new_filename
        for file_path in root.iter('path'):
            img_path = '../JPEGImages/'
            new_file_path = str(img_path + os.path.splitext(file)[0] + '.jpg')
            file_path.text = os.path.abspath(new_file_path)
        for name in root.iter('name'):
            if name.text not in classes:
                print(f'Unexpected class {name.text} in {file}')
            else:
                continue
        tree.write(path + file)


def remove_xml_attributes(path):
    """
    Remove attributes from a cvat xml file that cause problems in simple merging datasets
    :param path:
    :return:
    """
    tree = ET.parse(path)
    root = tree.getroot()
    for x in root.iter('image'):
        print(x.attrib)
        for y in x.iter('box'):
            for z in y.findall('attribute'):
                y.remove(z)

    tree.write(f'{PATH}/DMV/cvat/original/test.xml')

    for x in root.find('meta'):
        for y in x.find('labels'):
            for z in y.findall('attributes'):
                y.remove(z)

    tree.write(f'{PATH}/DMV/cvat/original/test.xml')
