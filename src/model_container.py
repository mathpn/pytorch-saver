import io
import json
import pickle
import time
import zipfile
from typing import Any, Dict, NamedTuple, Optional, Tuple

import torch
from torch import nn

Metadata = Dict[str, Any]


class ModelObjects(NamedTuple):
    model: nn.Module
    optimizer: Optional[Any] = None
    scheduler: Optional[Any] = None


class ModelContainer:
    """
    Model container class used to load and save PyTorch training objects and
    associated metadata.

    Attributes
    ----------
    objects_kwargs : dict[str, dict]
        dictionary with keyword arguments used to initialize internal objects.
    model : Any
        model object
    optim : Any
        optimizer object
    scheduler : Any
        scheduler object

    Methods
    -------
    load(self, file_path: str, model_class, optim_class=None, scheduler_class=None):
        Load stored state_dicts and metadata.

    save(self, **kwargs):
        Save internal objects and associated metadata.

    save_inference(self, **kwargs)
        Save only model and associated metadata.
    """

    def __init__(self):
        self.objects_kwargs = {}
        self.model = None
        self.optim = None
        self.scheduler = None

    def initialize(
        self,
        model_class: nn.Module,
        model_kwargs: Dict[str, Any],
        optim_class=None,
        optim_kwargs=None,
        scheduler_class=None,
        scheduler_kwargs=None,
    ) -> ModelObjects:
        """
        Initialize objects required for training or inference.

        Arguments
        ---------
        model_class : Any
            model class to create model object
        model_kwargs : dict[str, Any]
            dictionary with keyword arguments to initialize model object
        optim_class: Optional[Any]
            optional optimizer class to create optimizer object
        optim_kwargs : Optional[dict[str, Any]]
            optional dictionary with keyword arguments to initialize optimizer object
        scheduler_class: Optional[Any]
            scheduler class to create scheduler object
        scheduler_kwargs : Optional[dict[str, Any]]
            optional dictionary with keyword arguments to initialize scheduler object

        Returns
        -------
            objects (dict): dictionary with created objects
                keys: model and, optionally, optim and scheduler
        """
        self.objects_kwargs = {
            "model_kwargs": model_kwargs,
            "optim_kwargs": optim_kwargs,
            "scheduler_kwargs": scheduler_kwargs,
        }
        model_kwargs = model_kwargs or {}
        optim_kwargs = optim_kwargs or {}
        scheduler_kwargs = scheduler_kwargs or {}
        objects = {}
        self.model = model_class(**model_kwargs)
        objects["model"] = self.model
        if optim_class:
            self.optim = optim_class(self.model.parameters(), **optim_kwargs)
            objects["optim"] = self.optim
        if scheduler_class and optim_class:
            self.scheduler = scheduler_class(self.optim, **scheduler_kwargs)
            objects["scheduler"] = self.scheduler
        return objects

    def load(
        self,
        file_path: str,
        model_class: nn.Module,
        optim_class: Optional[Any] = None,
        scheduler_class: Optional[Any] = None,
    ) -> Tuple[Metadata, ModelObjects]:
        """
        Load stored state_dicts and metadata.

        Arguments
        ---------
        model_class : Any
            model class to create model object
        optim_class: Optional[Any]
            optimizer class to create optimizer object
        scheduler_class: Optional[Any]
            scheduler class to create scheduler object

        Returns
        -------
            metadata_dict (dict): dictionary with loaded metadata
            objects (dict): dictionary with loaded objects
        """
        with zipfile.ZipFile(file_path, "r") as zip_file:
            metadata_dict = json.loads(zip_file.read("metadata.json"))
            state_dicts = pickle.loads(zip_file.read("state_dicts.pkl"))
        objects = self.initialize(
            model_class,
            metadata_dict["model_kwargs"],
            optim_class,
            metadata_dict.get("optim_kwargs"),
            scheduler_class,
            metadata_dict.get("scheduler_class"),
        )
        self._load_state_dicts(state_dicts)
        return metadata_dict, objects

    def save(self, **kwargs) -> None:
        """
        Save internal objects and associated metadata.

        Internal objects state_dicts are pickled.
        Keyword arguments used to initialize the objects are stored
        in metadata.json.

        Any additional kwargs passed to this function are stored
        in metadata.json and must be JSON-serializable.

        Returns: None
        """
        state_dicts = self._create_state_dicts()
        metadata_dict = self._create_metadata_dict(**kwargs)
        self._save_zip(metadata_dict, state_dicts)

    def save_inference(self, **kwargs) -> None:
        """
        Save only model and associated metadata.

        The model state_dict is pickled.
        Keyword arguments used to initialize the model are stored
        in metadata.json.

        Any additional kwargs passed to this function are stored
        in metadata.json and must be JSON-serializable.

        Returns: None
        """
        state_dicts = self._create_state_dicts(inference=True)
        metadata_dict = self._create_metadata_dict(**kwargs)
        self._save_zip(metadata_dict, state_dicts)

    def _create_metadata_dict(self, **kwargs) -> dict[str, Any]:
        for key, value in kwargs.items():
            if not _is_json_serializable(value):
                raise ValueError(
                    f"At least one provided argument is not JSON-serializable: {key} of type {type(value)}"
                )
        metadata_dict = {k: v for k, v in self.objects_kwargs.items() if v}
        if overlap := set(metadata_dict).intersection(set(kwargs)):
            raise ValueError(f"Provided keyword arguments contain reserved keywords: {sorted(overlap)}")
        metadata_dict = {**metadata_dict, **kwargs}
        metadata_dict["timestamp"] = int(time.time())
        return metadata_dict

    def _create_state_dicts(self, inference: bool = False) -> dict[str, dict]:
        if self.model is None:
            raise RuntimeError("Model must be initialized or loaded before saving.")
        state_dicts = {"model_state_dict": self.model.state_dict()}
        if not inference and self.optim:
            state_dicts["optim_state_dict"] = self.optim.state_dict()
        if not inference and self.scheduler:
            state_dicts["scheduler_state_dict"] = self.scheduler.state_dict()
        return state_dicts

    def _save_zip(self, metadata_dict, state_dicts) -> None:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            zip_file.writestr("metadata.json", json.dumps(metadata_dict, indent=2))
            zip_file.writestr("state_dicts.pkl", pickle.dumps(state_dicts))

        with open("checkpoint.zip", "wb") as zip_file:
            zip_file.write(buffer.getvalue())

    def _load_state_dicts(self, state_dicts: dict[str, dict]) -> None:
        self.model.load_state_dict(state_dicts["model_state_dict"])
        optim_state_dict = state_dicts.get("optim_state_dict")
        if optim_state_dict and self.optim:
            self.optim.load_state_dict(optim_state_dict)
        scheduler_state_dict = state_dicts.get("scheduler_state_dict")
        if scheduler_state_dict and self.scheduler:
            self.optim.load_state_dict(optim_state_dict)


def _is_json_serializable(obj) -> bool:
    """check if an object is JSON-serializable."""
    try:
        json.dumps(obj)
        return True
    except (TypeError, OverflowError):
        return False
