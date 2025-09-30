import os

from codetiming import Timer
from open_spiel.python.algorithms.alpha_zero import model as az_model

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))


class AlphaZeroModel:
    """ Singleton to load an AlphaZero model from disk on first use and retain it in memory. """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with Timer(text="AZ.model_load took {:0.4f} seconds"):
                cls._instance = az_model.Model.from_checkpoint(
                    str(os.path.join(MODELS_DIR, "az_model/checkpoint--1")))
        return cls._instance
