import os

import pytest
import torch
from dummy_model import Model

from pytorch_saver._helpers import _remove_saved_file
from pytorch_saver._model_container import InitializationError, ModelContainer


def test_load_model():
    container = ModelContainer()
    model_kwargs = {
        "in_dim": 32,
        "hidden_dim": 32,
        "n_classes": 1,
        "n_layers": 1,
    }
    model_objects = container.initialize(model_class=Model, model_kwargs=model_kwargs)
    file_path = container.save("/tmp", prefix="test")
    container_2 = ModelContainer()
    metadata, loaded_obj = container_2.load(file_path, Model)
    state_dict = model_objects.model.state_dict()
    loaded_state_dict = loaded_obj.model.state_dict()
    for key, value in loaded_state_dict.items():
        assert torch.all(value == state_dict.get(key))
    assert metadata["model_kwargs"] == model_kwargs
    _remove_saved_file(file_path)


def test_load_model_optim_scheduler():
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
    file_path = container.save("/tmp", prefix="test")
    container_2 = ModelContainer()
    metadata, loaded_objs = container_2.load(
        file_path,
        Model,
        torch.optim.Adam,
        torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    )
    model_state_dict = model_objects.model.state_dict()
    loaded_model_state_dict = loaded_objs.model.state_dict()
    optim_state_dict = model_objects.optimizer.state_dict()
    loaded_optim_state_dict = loaded_objs.optimizer.state_dict()
    scheduler_state_dict = model_objects.scheduler.state_dict()
    loaded_scheduler_state_dict = loaded_objs.scheduler.state_dict()
    for key, value in loaded_model_state_dict.items():
        assert torch.all(value == model_state_dict.get(key))
    assert loaded_optim_state_dict["param_groups"] == optim_state_dict["param_groups"]
    assert loaded_scheduler_state_dict == scheduler_state_dict

    assert metadata["model_kwargs"] == model_kwargs
    assert metadata["optim_kwargs"] == optim_kwargs
    assert metadata["scheduler_kwargs"] == scheduler_kwargs
    _remove_saved_file(file_path)


def test_load_invalid_class():
    container = ModelContainer()
    model_kwargs = {
        "in_dim": 32,
        "hidden_dim": 32,
        "n_classes": 1,
        "n_layers": 1,
    }
    optim_kwargs = {"lr": 2e-4}
    _ = container.initialize(
        model_class=Model,
        model_kwargs=model_kwargs,
        optim_class=torch.optim.Adam,
        optim_kwargs=optim_kwargs,
    )
    file_path = container.save("/tmp", prefix="test")
    container_2 = ModelContainer()
    with pytest.raises(ValueError):
        _, _ = container_2.load(file_path, torch.optim.Adam, torch.optim.Adam)
    with pytest.raises(InitializationError):
        _, _ = container_2.load(file_path, Model, Model)
    _remove_saved_file(file_path)


def test_load_missing_class():
    container = ModelContainer()
    model_kwargs = {
        "in_dim": 32,
        "hidden_dim": 32,
        "n_classes": 1,
        "n_layers": 1,
    }
    optim_kwargs = {"lr": 2e-4}
    _ = container.initialize(
        model_class=Model,
        model_kwargs=model_kwargs,
        optim_class=torch.optim.Adam,
        optim_kwargs=optim_kwargs,
    )
    file_path = container.save("/tmp", prefix="test")
    container_2 = ModelContainer()
    _, loaded_objs = container_2.load(file_path, Model)
    assert loaded_objs.optimizer is None
    assert isinstance(loaded_objs.model, Model)
    _remove_saved_file(file_path)


def test_load_custom_metadata():
    container = ModelContainer()
    model_kwargs = {
        "in_dim": 32,
        "hidden_dim": 32,
        "n_classes": 1,
        "n_layers": 1,
    }
    _ = container.initialize(model_class=Model, model_kwargs=model_kwargs)
    file_path = container.save(
        "/tmp",
        prefix="test",
        loss=0.123,
        dataset="fake_data",
        random_data=[1.2, 4.3, 1.9, 0.1],
    )
    container_2 = ModelContainer()
    metadata, _ = container_2.load(file_path, Model)

    assert metadata["loss"] == 0.123
    assert metadata["dataset"] == "fake_data"
    assert metadata["random_data"] == [1.2, 4.3, 1.9, 0.1]
    _remove_saved_file(file_path)
