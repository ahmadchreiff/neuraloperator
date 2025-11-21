__version__ = "2.0.0"

from .models import TFNO, FNO, get_model
from .data import datasets, transforms
from . import mpu
from .training import Trainer, PhysicsWeightScheduler
from .losses import (
    LpLoss,
    H1Loss,
    BurgersEqnLoss,
    ICLoss,
    NavierStokesEqnLoss,
    WeightedSumLoss,
    Aggregator,
    Relobralo,
    SoftAdapt,
    FourierDiff,
    non_uniform_fd,
    FiniteDiff,
)
