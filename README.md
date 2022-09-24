# pytorch-saver
Simple helper to save and load PyTorch models.

## Why use this package to save and load models?

PyTorch [suggests](https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_models_for_inference.html) two standard ways of saving and loading models: either saving their state_dict or saving the pickled model object itself.

Both methods have drawbacks:

- Saving the state_dict is very flexible, however we loose all the arguments used to create the model, the optimizer and (optionally) the scheduler;
- Saving a pickled snapshot solves this issue, but it's not flexible at all. Even minor changes to the model class can break the unpickling process and the arguments used to define the object are still obscured behind the objects themselves.
  
Therefore, the goal of this package is to provide a pratical way of creating models and associated objects, saving, and loading them without headaches. Also, any additional metadata should be included in the saved file.

## Installing

Clone the repository and go inside its folder:

    cd pytorch-saver

Install with pip:

    pip install .


## How to use it

### Initializing objects

Import ModelContainer and create a new container instance.

    from pytorch_saver import ModelContainer
    container = ModelContainer()

This is the only part of the pipeline that breaks with Python conventions. Since we need to store all arguments used to create the objects as to recreate them, they are created through the initialize method.

Pass all the classes and dictionaries with all keyword arguments to initialize them to the initialize method.

    model_objects = container.initialize(
        Model,
        model_kwargs,
        torch.optim.Adam,
        optim_kwargs
    )

Model objects is a NamedTuple with three attributes: model, optimizer and scheduler. Access these objects (if created though initialize) and train your model.

### Saving checkpoints

Use the save method to save checkpoints:

    container.save("./", "tutorial")

This will save a checkpoint to "./tutorial_checkpoint_TIMESTAMP.zip", where TIMESTAMP is the current Unix timestamp in seconds.

Any additional keyword arguments provided will be saved as model metadata, as long as they are JSON-serializable.

    container.save("./", "tutorial", loss=0.55, epoch=5)

If you only want to save the model (ignoring optimizer and scheduler), use the save_inference method.

### Loading saved files

Use the load method to load checkpoints:

    from pytorch_saver import ModelContainer
    container = ModelContainer()
    metadata, objs = container.load(file_path)

"metadata" is a dictionary with the arguments used to initialize all objects, the timestamp, and any additional arguments passed to the saved method when saving this file.

"objs" is a NamedTuple with the same structure as the one returned by the initialize method.
