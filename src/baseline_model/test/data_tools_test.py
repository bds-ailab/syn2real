import sys
import os

# Adding the src directory to the sys.path to ensure imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_tools import *
import pytest

it = pytest.mark.it
describe = pytest.mark.describe
parametrize = pytest.mark.parametrize
skip = pytest.mark.skip


@describe("Test the training dataset random splitting into train and val")
class TestSplitTrainTest:

    @it("must take a split proportion 0<=prop<=1 otherwise raise a value error")
    @skip
    def test_prop_error(self):
        pass

    @it("must take a valid dataset path otherwise raise a value error")
    @skip
    def test_dataset_path(self):
        pass

    @it(
        "must split the dataset into two complementary train and val sets len(train)+len(val)=len(dataset)"
    )
    @skip
    def test_len_dataset(self):
        pass

    @it(
        "must return two datasets containing only the specified classes in the input parameters"
    )
    @skip
    def test_len_datasets(self):
        pass

    @it(
        "must return a train dataset with num_images_train ~= (1-prop)*total_num_images"
    )
    @skip
    def test_len_train_dataset(self):
        pass

    @it("must return a val dataset with num_images_val ~= prop*total_num_images")
    @skip
    def test_len_val_dataset(self):
        pass


@describe("Test the CustomDataset class : loads images and labels from paths")
class TestCustomDataset:

    @it("must return an int")
    @skip
    def test_len_output(self):
        pass

    @it("must load an image in PIL format")
    @skip
    def test_loaded_image(self):
        pass

    @it("must return int label")
    @skip
    def test_loaded_label(self):
        pass

    @it("must return an image transformed to size 224*224")
    @skip
    def test_image_resize_transform(self):
        pass

    @it("must return an image as tensor")
    @skip
    def test_image_tensor_transform(self):
        pass


@describe("Test the data preprocessing function for ResNet50")
class TestResnet50Preprocessing:

    @it(
        "must return a dataloader of batches with size (BATCH_SIZE, 3, 224, 224), (BATCH_SIZE,)"
    )
    @skip
    def test_batch_shape(self):
        pass

    @it(
        "must return a dataloader with NUM_BATCHES = NUM_IMAGES//BATCH_SIZE (+1 if not divisible)"
    )
    @skip
    def test_dataloader_len(self):
        pass
