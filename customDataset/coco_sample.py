import datumaro as dm
from datumaro.components.extractor import Transform, AnnotationType
from datumaro.components.operations import compute_ann_statistics
import random


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
        car_lable_idx = self._extractor.categories()[AnnotationType.label].find('car')[0]
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


def run_sampler():
    full_dataset = dm.Dataset.import_from('../../Datasets/TEST/coco_pistols_7k', 'yolo')
    stats = compute_ann_statistics(full_dataset)
    full_dataset.transform(Sampler, obj_class='person', num_samples=1)
    full_dataset.export(f'../../Datasets/TEST/export_test', 'cvat', save_images=False)
    return stats


if __name__ == "__main__":
    final_stats = run_sampler()
