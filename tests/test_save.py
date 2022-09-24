import json
import os
import pickle
import zipfile

import pytest
import torch
from dummy_model import Model

from pytorch_saver._helpers import _remove_saved_file
from pytorch_saver._model_container import ModelContainer


def test_save_model():
    container = ModelContainer()
    model_kwargs = {
        "in_dim": 32,
        "hidden_dim": 32,
        "n_classes": 1,
        "n_layers": 1,
    }
    model_objects = container.initialize(model_class=Model, model_kwargs=model_kwargs)
    file_path = container.save("/tmp", prefix="test")
    with zipfile.ZipFile(file_path, "r") as zip_file:
        metadata = json.loads(zip_file.read("metadata.json"))
        state_dicts = pickle.loads(zip_file.read("state_dicts.pkl"))
    state_dict = model_objects.model.state_dict()
    for key, value in state_dicts["model_state_dict"].items():
        assert torch.all(value == state_dict.get(key))
    print(metadata)
    assert metadata["model_kwargs"] == model_kwargs
    _remove_saved_file(file_path)


def test_save_model_optim_scheduler():
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
    with zipfile.ZipFile(file_path, "r") as zip_file:
        metadata = json.loads(zip_file.read("metadata.json"))
        state_dicts = pickle.loads(zip_file.read("state_dicts.pkl"))
    model_state_dict = model_objects.model.state_dict()
    for key, value in state_dicts["model_state_dict"].items():
        assert torch.all(value == model_state_dict.get(key))
    optim_state_dict = model_objects.optimizer.state_dict()
    assert (
        state_dicts["optim_state_dict"]["param_groups"]
        == optim_state_dict["param_groups"]
    )
    scheduler_state_dict = model_objects.scheduler.state_dict()
    print(scheduler_state_dict)
    assert state_dicts["scheduler_state_dict"] == scheduler_state_dict

    assert metadata["model_kwargs"] == model_kwargs
    assert metadata["optim_kwargs"] == optim_kwargs
    assert metadata["scheduler_kwargs"] == scheduler_kwargs
    _remove_saved_file(file_path)


def test_save_inference():
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
    file_path = container.save_inference("/tmp", prefix="test")
    with zipfile.ZipFile(file_path, "r") as zip_file:
        metadata = json.loads(zip_file.read("metadata.json"))
        state_dicts = pickle.loads(zip_file.read("state_dicts.pkl"))
    model_state_dict = model_objects.model.state_dict()
    for key, value in state_dicts["model_state_dict"].items():
        assert torch.all(value == model_state_dict.get(key))
    assert "optim_state_dict" not in state_dicts
    assert "scheduler_state_dict" not in state_dicts

    assert metadata["model_kwargs"] == model_kwargs
    assert "optim_kwargs" not in metadata
    assert "scheduler_kwargs" not in metadata
    _remove_saved_file(file_path)


def test_save_model_extra_kwargs():
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
    with zipfile.ZipFile(file_path, "r") as zip_file:
        metadata = json.loads(zip_file.read("metadata.json"))

    assert metadata["loss"] == 0.123
    assert metadata["dataset"] == "fake_data"
    assert metadata["random_data"] == [1.2, 4.3, 1.9, 0.1]

    _remove_saved_file(file_path)


def test_save_without_model():
    container = ModelContainer()
    with pytest.raises(RuntimeError):
        _ = container.save("/tmp", prefix="test")


def test_save_non_serializable_kwarg():
    container = ModelContainer()
    model_kwargs = {
        "in_dim": 32,
        "hidden_dim": 32,
        "n_classes": 1,
        "n_layers": 1,
    }
    _ = container.initialize(model_class=Model, model_kwargs=model_kwargs)
    with pytest.raises(ValueError):
        _ = container.save("/tmp", prefix="test", not_serializable={1, 2, 3})
