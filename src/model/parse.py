from .spec import ModelSpec
from .residual_diffusion import ResidualDiffusionModule
from .anchored_residual_diffusion import AnchoredResidualDiffusionModule
from .vm import VelocityModule
from .straightpcf import CoupledVelocityModule, StraightPCFModule

def get_model(model_config, **kwargs) -> ModelSpec:
    MAP = {
        'VelocityModule': VelocityModule,
        'ResidualDiffusionModule': ResidualDiffusionModule,
        'AnchoredResidualDiffusionModule': AnchoredResidualDiffusionModule,
        'CoupledVelocityModule': CoupledVelocityModule,
        'StraightPCFModule': StraightPCFModule,
    }
    __target__ = model_config['__target__']
    del model_config['__target__']
    assert __target__ in MAP, f"expect: [{','.join(MAP.keys())}], found: {__target__}"
    return MAP[__target__](model_config=model_config, **kwargs)
