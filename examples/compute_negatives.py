from collections import defaultdict
import json
import typing
import pydantic
import rich
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import dataloader
from dataloader.base import DatasetConfig
import datasets
import pandas as pd
import torch

from dataloader.constants import PROJECT_ROOT
from trie.icd import ICD10Trie

OUTPUT_DIR = PROJECT_ROOT / "data" / "medical-coding-systems" / "negatives"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def compute_negatives_from_codes(
    codes: list[str],
    trie: ICD10Trie,
    model_name: str,
    top_k: int = 200,
    batch_size: int = 128,
) -> dict[str, list[str]]:
    # Load model if string identifier
    model = SentenceTransformer(model_name)

    # Prepare dataset
    descriptions = [trie[code].description for code in codes]
    dataset = datasets.Dataset.from_dict({"codes": codes, "descriptions": descriptions})

    # Embedding function
    def embed_batch(batch: dict[str, list[typing.Any]]) -> dict[str, list[typing.Any]]:
        embeddings = model.encode(
            batch["descriptions"], convert_to_tensor=True, normalize_embeddings=True
        )
        return {**batch, "embeddings": embeddings.tolist()}

    # Apply embeddings
    dataset = dataset.map(
        embed_batch,
        batched=True,
        batch_size=batch_size,
        desc="Computing embeddings",
    )

    # Compute similarity matrix
    embeddings = torch.tensor(dataset["embeddings"])
    similarity_matrix = embeddings @ embeddings.T

    similarity_df = pd.DataFrame(similarity_matrix.numpy(), index=codes, columns=codes)

    approx_size = similarity_df.memory_usage().sum() / 1024 / 1024
    rich.print(f"Estimated size of similarity matrix: {approx_size:.2f} MB")

    # Build negatives dictionary
    negatives = defaultdict(list)
    for code in tqdm(codes, desc="Computing negatives"):
        top_similar = similarity_df[code].nlargest(top_k + 1).index
        for similar_code in top_similar:
            if similar_code != code:
                negatives[code].append(similar_code)

    return dict(negatives)


class Arguments(pydantic.BaseModel):
    embedding_model: str = "pritamdeka/S-PubMedBert-MS-MARCO"
    coding_system: int = 2022


def main(args: Arguments):
    xml_trie = ICD10Trie.from_cms(year=args.coding_system)
    xml_trie.parse()
    mdace_config: DatasetConfig = DatasetConfig(
        **dataloader.DATASET_CONFIGS["mdace-icd10cm"]
    )
    mdace = dataloader.load_dataset(mdace_config)
    mdace = datasets.concatenate_datasets(mdace.values())  # type: ignore

    unique_codes: set[str] = set()
    for codes in mdace["targets"]:
        unique_codes.update(codes)
    mdace = mdace.map(
        lambda row: {
            **row,
            "targets": [code for code in row["targets"] if code in xml_trie.lookup],
        }
    )

    filtered_codes = {code for codes in mdace["targets"] for code in codes}

    # print difference between unique_codes and filtered_codes
    rich.print(f"Number of filtered codes: {len(unique_codes - filtered_codes)}")

    cm_codes = xml_trie.get_root_leaves(root="cm")
    unique_cm_codes = set(code.name for code in cm_codes)
    codes = list(filtered_codes | unique_cm_codes)

    negatives = compute_negatives_from_codes(
        codes=codes,
        trie=xml_trie,
        model_name=args.embedding_model,
        top_k=200,
        batch_size=128,
    )

    output_dir = PROJECT_ROOT / "data" / "medical-coding-systems" / "negatives"
    output_dir.mkdir(exist_ok=True, parents=True)

    with open(output_dir / f"{args.coding_system}_negatives.json", "w") as f:
        json.dump(negatives, f, indent=2)


if __name__ == "__main__":
    main(Arguments())
