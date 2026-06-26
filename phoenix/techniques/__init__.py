"""Phoenix technique catalogue."""

from .cycling import CyclingModule
from .cv import CVModule
from .dcir import DCIRModule
from .degradation import DegradationModule
from .dqdv import DQDVModule
from .dvdq import DVDQModule
from .eis import EISModule
from .gitt import GITTModule
from .ici import CurrentInterruptionModule
from .ocv import OCVModule
from .parameter_perturbation import ParameterPerturbationModule
from .pitt import PITTModule
from .protocol_sensitivity import ProtocolSensitivityModule
from .rate_capability import RateCapabilityModule

TECHNIQUE_MODULES = {
    "Cycling": CyclingModule,
    "Rate capability": RateCapabilityModule,
    "CV": CVModule,
    "dQ/dV": DQDVModule,
    "dV/dQ": DVDQModule,
    "DCIR": DCIRModule,
    "ICI": CurrentInterruptionModule,
    "GITT": GITTModule,
    "PITT": PITTModule,
    "EIS": EISModule,
    "OCV": OCVModule,
    "Degradation": DegradationModule,
}

__all__ = [
    "CyclingModule",
    "RateCapabilityModule",
    "CVModule",
    "DQDVModule",
    "DVDQModule",
    "DCIRModule",
    "CurrentInterruptionModule",
    "GITTModule",
    "PITTModule",
    "EISModule",
    "OCVModule",
    "DegradationModule",
    "ParameterPerturbationModule",
    "ProtocolSensitivityModule",
    "TECHNIQUE_MODULES",
]
