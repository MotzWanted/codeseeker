import datasets
import dataloader
from dataloader.base import DatasetConfig
from trie.base import Trie
from trie.icd import ICD10Trie

from loguru import logger


def format_dataset(
    dataset: datasets.Dataset | datasets.DatasetDict,
    trie: Trie,
    debug: bool = False,
) -> datasets.Dataset:
    """Format the dataset."""
    if isinstance(dataset, datasets.DatasetDict):
        dataset = datasets.concatenate_datasets(list(dataset.values()))

    unique_codes: set[str] = set()
    for codes in dataset["targets"]:
        unique_codes.update(codes)
    dataset = dataset.map(
        lambda row: {
            **row,
            "targets": [code for code in row["targets"] if code in trie.lookup],
        }
    )

    filtered_codes: set[str] = set()
    for codes in dataset["targets"]:
        filtered_codes.update(codes)

    # print difference between unique_codes and filtered_codes
    filtered_codes = unique_codes - filtered_codes
    if filtered_codes:
        logger.warning(
            f"Number of filtered codes ({len(filtered_codes)}): `{filtered_codes}`"
        )
    if debug:
        return dataset.select(range(10))
    return dataset


xml_trie = ICD10Trie.from_cms(year=2022)
xml_trie.parse()

dset_config: DatasetConfig = DatasetConfig(
    **dataloader.DATASET_CONFIGS["mdace-icd10cm"]
)
dset_config.options.prep_map_kws = {
    "num_proc": 4,
}
dset = dataloader.load_dataset(dset_config)
dset = format_dataset(dset, xml_trie)

all_main_terms = xml_trie.get_all_main_terms()
unique_codes: set[str] = set()
for term in all_main_terms:
    if "Floppy" in term.title:
        print("here")
    term_codes = xml_trie.get_all_term_codes(term.id)
    unique_codes.update(term_codes)

mdace_codes: set[str] = set()
for codes in dset["targets"]:
    mdace_codes.update(codes)

# estimate recall
recall = len(mdace_codes.intersection(unique_codes)) / len(mdace_codes)
print(f"Recall: {recall:.2%}")

# list codes from mdace that are not in the trie
missing_codes = mdace_codes - unique_codes
print(f"Missing codes: {len(missing_codes)}")
