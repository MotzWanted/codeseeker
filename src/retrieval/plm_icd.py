import typing
import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer
from models.plmicd import PLMICDModel


class PLMICDRetriever:

    def __init__(
        self,
        pretrained_model_path: str,
        valid_labels: list[str] | None = None,
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
        self.id2label = typing.cast(dict[str, str], self.model.config.id2label)
        self.top_k = top_k
        self.note_max_length = note_max_length
        mask = torch.ones(len(self.id2label), dtype=torch.bool)
        if valid_labels:
            valid_set = set(valid_labels)
            for k, v in self.id2label.items():
                if v in valid_set:
                    continue
                mask[int(k)] = False
        # move mask to correct device once
        self.valid_label_mask = mask.to(self.device)
        super().__init__(*args, **kwargs)

    def __call__(
        self, batch: dict[str, list[typing.Any]], *args, **kwargs
    ) -> dict[str, list[typing.Any]]:
        if "note" not in batch:
            raise ValueError("Batch must contain 'note' key.")
        raw_inputs: list[list[str]] = batch["note"]

        # Tokenize the data
        tokenized_inputs = self.tokenizer(
            raw_inputs,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            max_length=self.note_max_length,
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**tokenized_inputs)["logits"]
            # mask out invalid labels by setting logits to a large negative
            logits = logits.masked_fill(
                ~self.valid_label_mask.unsqueeze(0), float("-1e9")
            )
            # Shape: (batch_size, num_labels)
            logits = logits.sigmoid()

        _, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        # Convert the top_k indices to labels
        # for item in batch... for index in item. Shape: (batch_size, top_k)
        top_k_codes = [
            [self.id2label[str(idx.item())] for idx in indices]  # type: ignore
            for indices in top_k_indices
        ]

        return {
            **batch,
            "codes": top_k_codes,
        }
