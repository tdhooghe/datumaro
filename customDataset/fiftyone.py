# %%
import fiftyone as fo
from utils import PATH

for x in fo.list_datasets():
    fo.delete_dataset(x)

dataset_dir = f"{PATH}/SOHAS/cvat/sohas_auto_splits/valid"

# Create the dataset
dataset = fo.Dataset.from_dir(
    dataset_dir=dataset_dir,
    dataset_type=fo.types.CVATImageDataset,
)
session = fo.launch_app(dataset)
# View summary info about the dataset
print(dataset)

# Print the first few samples in the dataset
print(dataset.head())

