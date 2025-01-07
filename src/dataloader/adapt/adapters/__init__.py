from dataloader.adapt.base import Adapter
from dataloader.adapt.adapters.meddec import MedDecAdapter
from dataloader.adapt.adapters.snomed import SnomedAdapter
from dataloader.adapt.adapters.mdace import MdaceAdapter
from dataloader.adapt.adapters.nbme import NbmeAdapter
from dataloader.adapt.adapters.mimiciv import MimicIvForTrainingAdapter, MimicIvInferenceAdapter

KNOWN_ADAPTERS: list[type[Adapter]] = [
    NbmeAdapter,
    MedDecAdapter,
    SnomedAdapter,
    MdaceAdapter,
    MimicIvForTrainingAdapter,
    MimicIvInferenceAdapter,
]


def get_adapter_by_name(adapter_name: str):
    # Ensure the adapter name matches an object in globals
    return globals().get(adapter_name)
