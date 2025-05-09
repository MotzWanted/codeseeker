import typing
import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer
from models.plmicd import PLMICDModel


class PLMICDLocateAgent:

    def __init__(
        self,
        pretrained_model_path: str,
        device: str = "cpu",
        top_k: int = 1000,
        note_max_length: int = 4000,
        *args,
        **kwargs,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_path + "/tokenizer"
        )
        self.model = PLMICDModel.from_pretrained(pretrained_model_path + "/model")
        self.model.eval()
        self.device = device
        self.model.to(device)  # type: ignore
        self.id2label = self.model.config.id2label
        self.top_k = top_k
        self.note_max_length = note_max_length
        super().__init__(*args, **kwargs)

    def __call__(
        self, batch: dict[str, list[typing.Any]], *args, **kwargs
    ) -> dict[str, list[typing.Any]]:
        if "notes" not in batch:
            raise ValueError("Batch must contain 'notes' key.")
        raw_inputs: list[list[str]] = batch["notes"]

        # Tokenize the data
        tokenized_inputs = self.tokenizer(
            raw_inputs,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            max_length=self.note_max_length,
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**tokenized_inputs)

        # Shape: (batch_size, num_labels)
        logits = outputs.logits.sigmoid()
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        # Convert the top_k indices to labels
        # for item in batch... for index in item. Shape: (batch_size, top_k)
        top_k_codes = [
            [self.id2label[int(idx.item())] for idx in indices]
            for indices in top_k_indices
        ]

        return {
            **batch,
            f"recall@{self.top_k}": top_k_codes,
        }
