from echo.src.base_objective import BaseObjective
import numpy as np
from .models import QuantizedTransformer


class Objective(BaseObjective):
    def __init__(self, config, metric="val_loss"):
        BaseObjective.__init__(self, config, metric)

    def train(self, trial, conf):
        return
