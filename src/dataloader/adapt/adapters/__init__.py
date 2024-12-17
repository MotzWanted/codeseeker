from dataloader.adapt.adapters.meddec import MedDecAdapter
from dataloader.adapt.adapters.snomed import SnomedAdapter
from dataloader.adapt.adapters.mdace import MdaceAdapter
from dataloader.adapt.base import Adapter
from dataloader.adapt.adapters.nbme import NbmeAdapter

KNOWN_ADAPTERS: list[type[Adapter]] = [NbmeAdapter, MedDecAdapter, SnomedAdapter, MdaceAdapter]
