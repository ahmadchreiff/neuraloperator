from typing import Any, List, Optional

from zencfg import ConfigBase
from .distributed import DistributedConfig
from .models import ModelConfig, FNO_Medium2d
from .opt import OptimizationConfig, PatchingConfig
from .wandb import WandbConfig


class NavierStokesOptConfig(OptimizationConfig):
    """
    Default:
    n_epochs: int = 600
    learning_rate: float = 3e-4
    training_loss: str = "h1"
    weight_decay: float = 1e-4
    scheduler: str = "StepLR"
    step_size: int = 100
    gamma: float = 0.5
    """
    n_epochs: int = 4
    learning_rate: float = 1e-3
    training_loss: str = "l2"
    weight_decay: float = 1e-4
    scheduler: str = "StepLR"
    step_size: int = 5
    gamma: float = 0.7


class NavierStokesDatasetConfig(ConfigBase):
    """
    Default:
    folder: str = "~/data/navier_stokes/"
    batch_size: int = 8
    n_train: int = 10000
    train_resolution: int = 128
    n_tests: List[int] = [2000]
    test_resolutions: List[int] = [128]
    test_batch_sizes: List[int] = [8]
    encode_input: bool = True
    encode_output: bool = True
    """
    folder: str = "~/data/navier_stokes/"
    batch_size: int = 8
    n_train: int = 100
    train_resolution: int = 128
    n_tests: List[int] = [50]
    test_resolutions: List[int] = [128]
    test_batch_sizes: List[int] = [8]
    encode_input: bool = True
    encode_output: bool = True
 
 
class NavierStokesPhysicsLossConfig(ConfigBase):
    enabled: bool = False
    viscosity: float = 0.001
    dx: float = 1.0
    dy: float = 1.0
    dt: float = 1.0
    advection_weight: float = 1.0
    diffusion_weight: float = 1.0
    forcing_weight: float = 0.0
    use_vorticity_form: bool = True
    derivative_mode: str = "spectral"
    denormalize: bool = True
    initial_weight: float = 0.0
    max_weight: float = 1.0
    warmup_epochs: int = 50
    weight_schedule: str = "none"


class Default(ConfigBase):
    n_params_baseline: Optional[Any] = None
    verbose: bool = True
    distributed: DistributedConfig = DistributedConfig()
    model: ModelConfig = FNO_Medium2d()
    opt: OptimizationConfig = NavierStokesOptConfig()
    data: NavierStokesDatasetConfig = NavierStokesDatasetConfig()
    physics_loss: NavierStokesPhysicsLossConfig = NavierStokesPhysicsLossConfig()
    patching: PatchingConfig = PatchingConfig()
    wandb: WandbConfig = WandbConfig()
