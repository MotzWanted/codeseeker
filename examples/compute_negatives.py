from collections import defaultdict
import json
import pydantic
from openai import OpenAI
from tqdm import tqdm
import dataloader
from dataloader.base import DatasetConfig
import datasets
import pandas as pd
import torch

from dataloader.constants import PROJECT_ROOT
from trie.icd import ICD10Trie

_MEDICAL_CODING_SYSTEMS_DIR = PROJECT_ROOT / "data/medical-coding-systems/icd"


class Arguments(pydantic.BaseModel):
    embedding_provider: str = "vllm"
    embedding_host: str = "localhost"
    embedding_port: int = 6538
    embedding_model: str = "BAAI/bge-multilingual-gemma2"
    coding_system: str = "icd10cm_2022"


class EmbeddingClient:
    def __init__(self, endpoint: str, model_name: str):
        self.endpoint = endpoint
        self.model_name = model_name
        self.client = None

    def get_client(self):
        self.client = OpenAI(base_url=self.endpoint, api_key="EMPTY")

    def embed(self, text: str | list[str]) -> list[list[float]]:
        if self.client is None:
            self.get_client()

        if isinstance(text, str):
            text = [text]

        responses = self.client.embeddings.create(  # type: ignore
            input=text,
            model=self.model_name,
        )
        return [r.embedding for r in responses.data]


def main(args: Arguments):
    client = EmbeddingClient(
        f"http://{args.embedding_host}:{args.embedding_port}/v1/", args.embedding_model
    )

    xml_trie = ICD10Trie.from_cms(year=2022)
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

    filtered_codes: set[str] = set()
    for codes in mdace["targets"]:
        filtered_codes.update(codes)

    # print difference between unique_codes and filtered_codes
    print("Number of filtered codes:", len(unique_codes - filtered_codes))

    cm_codes = xml_trie.get_root_leaves(root="cm")
    unique_cm_codes = set(code.name for code in cm_codes)
    codes = list(filtered_codes | unique_cm_codes)
    descriptions = [xml_trie[code].description for code in codes]

    dataset = datasets.Dataset.from_dict({"codes": codes, "descriptions": descriptions})

    dataset = dataset.map(
        lambda x: {"embeddings": client.embed(x["descriptions"])},
        batched=True,
        batch_size=1024,
        load_from_cache_file=False,
    )

    embeddings = dataset["embeddings"]
    embeddings = [torch.tensor(emb) for emb in embeddings]
    embeddings = [emb / emb.norm(dim=-1, keepdim=True) for emb in embeddings]
    similarities = torch.stack(embeddings) @ torch.stack(embeddings).transpose(0, 1)

    similarity_matrix = pd.DataFrame(similarities.numpy(), index=codes, columns=codes)
    assert similarity_matrix.values.diagonal().all() == 1

    print(similarity_matrix.head(5))

    approx_size = similarity_matrix.memory_usage().sum() / 1024 / 1024
    print(f"Estimated size of similarity matrix: {approx_size:.2f} MB")

    # Create a dictionary of negatives
    # Each code should a lists of 1000 tuples
    # where each tuple is (code, description) of the 1000 codes with the highest similarity
    # and save to a json file
    output_dir = PROJECT_ROOT / "data" / "medical-coding-systems" / "negatives"
    output_dir.mkdir(exist_ok=True, parents=True)
    negatives = defaultdict(list)
    for code in tqdm(codes):
        top_similarities = similarity_matrix[code].nlargest(200)
        for similar_code, _ in top_similarities.items():
            if similar_code == code:
                continue
            negatives[code].append(similar_code)
    with open(output_dir / f"{args.coding_system}_negatives.json", "w") as f:
        json.dump(negatives, f, indent=2)


if __name__ == "__main__":
    main(Arguments())
