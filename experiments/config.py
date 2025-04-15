from functools import partial
import hashlib
import pathlib
import typing as typ
import datasets
import pydantic
from pydantic_settings import BaseSettings, SettingsConfigDict
from throughster.factory import create_interface

DUMP_FOLDER = pathlib.Path("~/research/codeseeker/experiments").expanduser()


class BaseArguments(BaseSettings):
    """Args for the script."""

    experiment_id: str
    experiment_name: str

    provider: str = "vllm"  # "azure" | "vllm" | "mistral"
    api_base: str
    deployment: str
    endpoint: typ.Literal["chat/completions", "completions"] = "completions"
    use_cache: bool = True  # whether to cache on request level

    dataset: str
    seed: str

    num_workers: int
    batch_size: int

    model_config = SettingsConfigDict(cli_parse_args=True, frozen=True)

    @pydantic.computed_field
    def _deployment_name(self) -> str:
        """Get the model name."""
        return self.deployment.split("/")[-1]

    @pydantic.computed_field
    def _hash(self) -> str:
        """Create unique identifier for the arguments"""
        model_dict = self.model_dump(
            exclude={"_hash", "_deployment_name", "experiment_folder", "experiment_name", "seed", "fewshots"}
        )
        return hashlib.md5(str(model_dict).encode()).hexdigest()

    @pydantic.computed_field
    def experiment_folder(self) -> str:
        """Get the experiment name."""
        return f"{self.experiment_name}/{self._deployment_name}"

    @pydantic.computed_field
    def _seeds(self) -> list[int]:
        """Get the seeds."""
        return [int(x) for x in self.seed.split(":")] if ":" in self.seed else [int(self.seed)]

    @pydantic.computed_field
    def _datasets(self) -> list[str]:
        """Get the datasets."""
        return [x for x in self.dataset.split(":")] if ":" in self.dataset else [self.dataset]

    @pydantic.computed_field
    def _prompts(self) -> list[str]:
        """Get the prompts."""
        return [x for x in self.prompt_name.split(":")] if ":" in self.prompt_name else [self.prompt_name]


def _get_dataset(dset: datasets.Dataset | datasets.DatasetDict) -> datasets.Dataset:
    """Get a `datasets.Dataset`."""
    if isinstance(dset, datasets.Dataset):
        return dset
    return next(iter(dset.values()))


def _init_client_fn(
    provider: str, api_base: str, endpoint: str, deployment: str, use_cache: bool, **kwargs
) -> typ.Callable:
    return partial(
        create_interface,
        provider=provider,
        api_base=api_base,
        endpoint=endpoint,
        model_name=deployment,
        use_cache=use_cache,
        cache_dir=str(pathlib.Path(f"~/.cache/throughster/{deployment}").expanduser()),
    )
