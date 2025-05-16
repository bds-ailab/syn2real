import sys
import os

# Adding the src directory to the sys.path to ensure imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model_tools import *
import pytest

it = pytest.mark.it
describe = pytest.mark.describe
parametrize = pytest.mark.parametrize
skip = pytest.mark.skip


@describe("Test pretrained model loading")
class TestModelLoad:

    @it("must return an nn.Module object named ResNet")
    @skip
    def test_model_type(self):
        pass

    @it("must create a model with NUM_CLASSES neurone in the last fc layer")
    @skip
    def test_model_fclayer(self):
        pass

    @it("must have all model's parameters trainable")
    @skip
    def test_model_trainable(self):
        pass

    @it("must return an output tensor with size (BATCH_SIZE, NUM_CLASSES)")
    @skip
    def test_model_output_shape(self):
        pass

    @it("must return this output {} for this input tensor {}")
    @skip
    def test_model_expected_output(self):
        pass


@describe("Test model training")
class TestModelTrain:

    @it("must always have a loss != 0")
    @skip
    def test_train_loss(self):
        pass

    @it(
        "must update (change) the parameters after each step : params_i != params_(i+1)"
    )
    @skip
    def test_train_step(self):
        pass

    @it("must converge in terms of training loss : loss_i < 2*loss_0 ")
    @skip
    def test_train_convergence(self):
        pass


@describe("Test model inference")
class TestModelEval:

    @it("must return the following predictions {} for this mock model and dataloader")
    @skip
    def test_model_inference(self):
        pass
