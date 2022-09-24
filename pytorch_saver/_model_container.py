import io
import json
import pickle
import time
import zipfile
from typing import Any, Dict, NamedTuple, Optional, Tuple

from torch import nn

from pytorch_saver._helpers import _is_json_serializable

Metadata = Dict[str, Any]


class ModelObjects(NamedTuple):
    model: nn.Module
    optimizer: Optional[Any] = None
    scheduler: Optional[Any] = None


class InitializationError(Exception):
    pass


def _initialize(name: str, object_class, *args, **kwargs):
    try:
        obj = object_class(*args, **kwargs)
    except Exception as exc:
        raise InitializationError(f"failed to initialize {name}") from exc
    return obj


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

    save(self, folder_path: str, **kwargs):
        Save internal objects and associated metadata.

    save_inference(self, folder_path: str, **kwargs)
        Save only model and associated metadata.
    """

    def __init__(self):
        self.__objects_kwargs = {}
        self.__model = None
        self.__optim = None
        self.__scheduler = None

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
        self._check_classes(model_class, optim_class, scheduler_class)

        self.__objects_kwargs = {
            "model_kwargs": model_kwargs,
            "optim_kwargs": optim_kwargs,
            "scheduler_kwargs": scheduler_kwargs,
        }
        model_kwargs = model_kwargs or {}
        optim_kwargs = optim_kwargs or {}
        scheduler_kwargs = scheduler_kwargs or {}
        self.__model = _initialize("model", model_class, **model_kwargs)
        if optim_class:
            self.__optim = _initialize(
                "optim", optim_class, self.__model.parameters(), **optim_kwargs
            )
        if scheduler_class and optim_class:
            self.__scheduler = _initialize(
                "scheduler", scheduler_class, self.__optim, **scheduler_kwargs
            )
        objects = ModelObjects(
            model=self.__model, optimizer=self.__optim, scheduler=self.__scheduler
        )
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
        self._check_classes(model_class, optim_class, scheduler_class)

        with zipfile.ZipFile(file_path, "r") as zip_file:
            metadata_dict = json.loads(zip_file.read("metadata.json"))
            state_dicts = pickle.loads(zip_file.read("state_dicts.pkl"))
        objects = self.initialize(
            model_class,
            metadata_dict["model_kwargs"],
            optim_class,
            metadata_dict.get("optim_kwargs"),
            scheduler_class,
            metadata_dict.get("scheduler_kwargs"),
        )
        self._load_state_dicts(state_dicts)
        return metadata_dict, objects

    def save(self, folder_path: str, prefix: Optional[str] = None, **kwargs) -> None:
        """
        Save internal objects and associated metadata.

        Internal objects state_dicts are pickled.
        Keyword arguments used to initialize the objects are stored
        in metadata.json.

        Any additional kwargs passed to this function are stored
        in metadata.json and must be JSON-serializable.

        Arguments
        ---------
        folder_path : str
            path to folder to save the checkpoint file
        prefix : str
            prefix string used to identify the saved file. Default is no prefix
        **kwargs
            extra keyword arguments are stored in metadata.json inside the saved ZIP

        Returns
        -------
        file_path : str
            file path to saved ZIP file
        """
        state_dicts = self._create_state_dicts()
        metadata_dict = self._create_metadata_dict(**kwargs)
        file_path = self._save_zip(
            folder_path, prefix, "checkpoint", metadata_dict, state_dicts
        )
        return file_path

    def save_inference(
        self, folder_path: str, prefix: Optional[str] = None, **kwargs
    ) -> None:
        """
        Save only model and associated metadata.

        The model state_dict is pickled.
        Keyword arguments used to initialize the model are stored
        in metadata.json.

        Any additional kwargs passed to this function are stored
        in metadata.json and must be JSON-serializable.

        Arguments
        ---------
        folder_path : str
            path to folder to save the checkpoint file
        prefix : str
            prefix string used to identify the saved file. Default is no prefix
        **kwargs
            extra keyword arguments are stored in metadata.json inside the saved ZIP

        Returns
        -------
        file_path : str
            file path to saved ZIP file
        """
        state_dicts = self._create_state_dicts(inference=True)
        metadata_dict = self._create_metadata_dict(inference=True, **kwargs)
        file_path = self._save_zip(
            folder_path, prefix, "inference", metadata_dict, state_dicts
        )
        return file_path

    def _check_classes(self, model_class, optim_class, scheduler_class) -> None:
        if not issubclass(model_class, nn.Module):
            raise ValueError("model_class must be an instance of torch.nn.Module")

        if scheduler_class is not None and optim_class is None:
            raise ValueError("optimizer must be initialized to use a scheduler")

    def _create_metadata_dict(
        self, *, inference: bool = False, **kwargs
    ) -> Dict[str, Any]:
        for key, value in kwargs.items():
            if not _is_json_serializable(value):
                raise ValueError(
                    f"At least one provided argument is not JSON-serializable: {key} of type {type(value)}"
                )
        if inference:
            metadata_dict = {"model_kwargs": self.__objects_kwargs["model_kwargs"]}
        else:
            metadata_dict = {k: v for k, v in self.__objects_kwargs.items() if v}
        metadata_dict["timestamp"] = int(time.time())
        if overlap := set(metadata_dict).intersection(set(kwargs)):
            raise ValueError(
                f"Provided keyword arguments contain reserved keywords: {sorted(overlap)}"
            )
        metadata_dict = {**metadata_dict, **kwargs}
        return metadata_dict

    def _create_state_dicts(self, inference: bool = False) -> Dict[str, Dict]:
        if self.__model is None:
            raise RuntimeError("Model must be initialized or loaded before saving.")
        state_dicts = {"model_state_dict": self.__model.state_dict()}
        if not inference and self.__optim:
            state_dicts["optim_state_dict"] = self.__optim.state_dict()
        if not inference and self.__scheduler:
            state_dicts["scheduler_state_dict"] = self.__scheduler.state_dict()
        return state_dicts

    def _save_zip(
        self, folder_path: str, prefix: str, file_name: str, metadata_dict, state_dicts
    ) -> str:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            zip_file.writestr("metadata.json", json.dumps(metadata_dict, indent=2))
            zip_file.writestr("state_dicts.pkl", pickle.dumps(state_dicts))

        # TODO use timestamp from dictionary
        if prefix:
            file_path = f"{folder_path}/{prefix}_{file_name}_{int(time.time())}.zip"
        else:
            file_path = f"{folder_path}/{file_name}_{int(time.time())}.zip"

        with open(file_path, "wb") as zip_file:
            zip_file.write(buffer.getvalue())
        return file_path

    def _load_state_dicts(self, state_dicts: dict[str, dict]) -> None:
        self.__model.load_state_dict(state_dicts["model_state_dict"])
        optim_state_dict = state_dicts.get("optim_state_dict")
        if optim_state_dict and self.__optim:
            self.__optim.load_state_dict(optim_state_dict)
        scheduler_state_dict = state_dicts.get("scheduler_state_dict")
        if scheduler_state_dict and self.__scheduler:
            self.__optim.load_state_dict(optim_state_dict)
