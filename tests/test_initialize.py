import pytest
import torch
from dummy_model import Model
from torch import nn

from pytorch_saver._model_container import InitializationError, ModelContainer


def test_initialize_model():
    container = ModelContainer()
    model_kwargs = {
        "in_dim": 32,
        "hidden_dim": 32,
        "n_classes": 1,
        "n_layers": 1,
    }
    model_objects = container.initialize(model_class=Model, model_kwargs=model_kwargs)
    assert isinstance(model_objects.model, nn.Module)
    assert model_objects.model.fc1.weight.data.shape == (32, 32)


def test_initialize_model_and_optim():
    container = ModelContainer()
    model_kwargs = {
        "in_dim": 32,
        "hidden_dim": 32,
        "n_classes": 1,
        "n_layers": 1,
    }
    optim_kwargs = {"lr": 2e-4}
    model_objects = container.initialize(
        model_class=Model,
        model_kwargs=model_kwargs,
        optim_class=torch.optim.Adam,
        optim_kwargs=optim_kwargs,
    )
    assert isinstance(model_objects.model, nn.Module)
    assert isinstance(model_objects.optimizer, torch.optim.Adam)
    assert model_objects.optimizer.param_groups[0]["lr"] == 2e-4


def test_initialize_model_and_scheduler():
    container = ModelContainer()
    model_kwargs = {
        "in_dim": 32,
        "hidden_dim": 32,
        "n_classes": 1,
        "n_layers": 1,
    }
    optim_kwargs = {"lr": 2e-4}
    scheduler_kwargs = {
        "T_0": 10,
    }
    model_objects = container.initialize(
        model_class=Model,
        model_kwargs=model_kwargs,
        optim_class=torch.optim.Adam,
        optim_kwargs=optim_kwargs,
        scheduler_class=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
        scheduler_kwargs=scheduler_kwargs,
    )
    assert isinstance(
        model_objects.scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
    )
    assert model_objects.scheduler.T_0 == 10


def test_initialize_missing_model():
    container = ModelContainer()
    model_kwargs = {
        "in_dim": 32,
        "hidden_dim": 32,
        "n_classes": 1,
        "n_layers": 1,
    }
    with pytest.raises(TypeError):
        _ = container.initialize(model_class=None, model_kwargs=model_kwargs)


def test_initialize_missing_optim():
    container = ModelContainer()
    model_kwargs = {
        "in_dim": 32,
        "hidden_dim": 32,
        "n_classes": 1,
        "n_layers": 1,
    }
    optim_kwargs = {"lr": 2e-4}
    scheduler_kwargs = {
        "T_0": 10,
    }
    with pytest.raises(ValueError):
        _ = container.initialize(
            model_class=Model,
            model_kwargs=model_kwargs,
            optim_class=None,
            optim_kwargs=optim_kwargs,
            scheduler_class=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
            scheduler_kwargs=scheduler_kwargs,
        )


def test_initialize_invalid_kwargs():
    container = ModelContainer()
    model_kwargs = {
        "_in_dim": 32,
        "hidden_dim": 32,
        "n_classes": 1,
        "n_layers": 1,
    }
    optim_kwargs = {"lr": 2e-4}
    scheduler_kwargs = {
        "T_0": 10,
    }
    with pytest.raises(InitializationError):
        _ = container.initialize(
            model_class=Model,
            model_kwargs=model_kwargs,
            optim_class=torch.optim.Adam,
            optim_kwargs=optim_kwargs,
            scheduler_class=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
            scheduler_kwargs=scheduler_kwargs,
        )
