# mlflow_with_lightning
Experimenting with MLflow framework with PyTorch Lightning

A simple guide:
Use the `lightning` library to abstract away your deep learning model. You can check the documentation [here](https://lightning.ai/docs/pytorch/stable/) and [here](https://github.com/Lightning-AI/pytorch-lightning). What you need to do is the following:

1. Create your deep learning model like you would with any deep learning model using `pytorch` by defining the forward function and some important attributes.
2. Define your `LightningModule`, which wraps the `pytorch` model.
3. Define the following functions: `forward`, `training_step` and `configure_optimizers`. Some optional functions include `validation_step` and `test_step` etc.
   
With these and the the correct data, you can fit your model easily. A neat thing about `lightning` is that it works very nicely with `mlflow`. Here is a nice [documentation](https://mlflow.org/docs/latest/python_api/mlflow.pytorch.html) of how you can log your `pytorch` experiment.

Once you have logged successfully logged your experiments with `mlflow` go to your project's root directory and run the following command.
```console
mlflow server --host 127.0.0.1 --port 8080
```
