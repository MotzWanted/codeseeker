import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.configuration_auto import AutoConfig
from typing import Any, Dict, Optional


class PLMICDConfig(PretrainedConfig):
    is_composition = True

    def __init__(
        self,
        num_classes: int,
        chunk_size: int,
        pad_token_id: int,
        id2label: Dict[str, str],
        label2id: Dict[str, int],
        encoder: PreTrainedModel | PretrainedConfig | None = None,
        selection_threshold: float | None = None,
        output_activation: str = "sigmoid",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.num_classes = num_classes
        self.chunk_size = chunk_size
        self.pad_token_id = pad_token_id
        self.id2label = id2label
        self.label2id = label2id
        self.selection_threshold = selection_threshold
        self.output_activation = output_activation


class PLMICDModel(PreTrainedModel):
    config_class = PLMICDConfig

    def __init__(
        self,
        config: Optional[PLMICDConfig] = None,
        encoder: Optional[PreTrainedModel] = None,
    ):
        if config is None or (config.encoder is None and encoder is None):
            raise ValueError(
                "Either an encoder configuration or an encoder has to be provided."
            )

        if config.encoder is None:
            if encoder is None:
                raise ValueError("Both config.encoder and encoder cannot be None")
            config.encoder = encoder.config

        # initialize with config
        super().__init__(config)

        # Use pretrained encoder if provided, otherwise initialize a new encoder from config
        if encoder is None:
            encoder_model_type = config.encoder.pop("model_type")
            encoder_config_dict = config.encoder.to_dict()
            self.config.encoder = AutoConfig.for_model(
                encoder_model_type, **encoder_config_dict
            )
            encoder = AutoModel.from_config(
                self.config.encoder, add_pooling_layer=False
            )
            if encoder is None:
                raise ValueError(
                    f"Encoder model type {encoder_model_type} not found in AutoModel."
                )
            encoder.eval()

        self.encoder = encoder

        # make sure that the individual model's config refers to the shared config
        # so that the updates to the config will be synced
        self.encoder.config = self.config.encoder

        self.label_wise_attention = LabelCrossAttention(
            input_size=config.encoder.hidden_size, num_classes=config.num_classes
        )

        if config.output_activation == "sigmoid":
            self.output_activation_fn = torch.sigmoid
        elif config.output_activation == "softmax":
            self.output_activation_fn = torch.softmax

    def save_pretrained(self, save_directory, **kwargs):
        """Save configuration and model's state_dict to the directory."""
        self.config.save_pretrained(save_directory)
        torch.save(self.state_dict(), f"{save_directory}/pytorch_model.bin")

    @classmethod
    def from_pretrained(cls, save_directory, *model_args, **kwargs):
        config = PLMICDConfig.from_pretrained(save_directory)
        if not isinstance(config, PLMICDConfig):
            raise ValueError(f"Config loaded is not a PLMICDConfig: {type(config)}")
        model = cls(config, *model_args, **kwargs)
        state_dict = torch.load(
            f"{save_directory}/pytorch_model.bin",
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state_dict)

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()
        return model

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        chunk_input_ids = self.split_input_into_chunks(
            input_ids, self.config.pad_token_id
        )
        chunk_attention_mask = self.split_input_into_chunks(attention_mask, 0)
        batch_size, num_chunks, chunk_size = chunk_input_ids.size()

        # Remove chunks that are entirely padding.
        relevant_chunk_indices = (
            (chunk_input_ids != self.config.pad_token_id)
            .any(dim=-1)
            .view(-1)
            .nonzero(as_tuple=True)[0]
        )
        relevant_chunk_input_ids = torch.index_select(
            chunk_input_ids.view(-1, chunk_size), 0, relevant_chunk_indices
        )
        relevant_chunk_attention_mask = torch.index_select(
            chunk_attention_mask.view(-1, chunk_size), 0, relevant_chunk_indices
        )

        if self.encoder is None:
            raise ValueError("Encoder is not initialized")

        outputs = self.encoder(
            input_ids=relevant_chunk_input_ids,
            attention_mask=relevant_chunk_attention_mask,
        )[0]

        # Set outputs for padding chunks to zero vector
        outputs_with_padding_chunks = torch.zeros(
            (batch_size * chunk_input_ids.size(1), chunk_size, outputs.size(-1)),
            device=self.encoder.device,
            dtype=outputs.dtype,
        )
        out = outputs_with_padding_chunks.index_copy_(
            0, relevant_chunk_indices, outputs
        )

        return out.view(batch_size, num_chunks * chunk_size, -1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        loss_reduction: str = "mean",
        **kwargs,
    ) -> dict[str, Any]:
        input_ids = input_ids.to(self.encoder.device)
        attention_mask = attention_mask.to(self.encoder.device)

        hidden_output = self.encode(input_ids, attention_mask)
        logits, attention = self.label_wise_attention(
            hidden_output, attention_mask, return_attention=return_attention
        )

        # Ensure the attention tensor covers only the original sequence length.
        if attention is not None:
            attention = attention[:, :, : input_ids.size(1)]

        loss = None
        if targets is not None:
            targets = targets.to(self.encoder.device)
            loss = F.binary_cross_entropy_with_logits(
                logits, targets, reduction=loss_reduction
            )

        return {"logits": logits, "loss": loss, "attention": attention}

    def predict(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        logits = self(input_ids, attention_mask)["logits"]
        return self.output_activation_fn(logits)  # type: ignore

    def split_input_into_chunks(
        self, input_sequence: torch.Tensor, pad_index: int
    ) -> torch.Tensor:
        """Split input into chunks of chunk_size.
        Args:
            input_sequence (torch.Tensor): input sequence to split (batch_size, seq_len)
            pad_index (int): padding index
        Returns:
            torch.Tensor: reshaped input (batch_size, num_chunks, chunk_size)
        """
        batch_size = input_sequence.size(0)
        # pad input to be divisible by chunk_size
        input_sequence = nn.functional.pad(
            input_sequence,
            (
                0,
                self.config.chunk_size
                - input_sequence.size(1) % self.config.chunk_size,
            ),
            value=pad_index,
        )
        return input_sequence.view(batch_size, -1, self.config.chunk_size)


class LabelCrossAttention(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.weights_k = nn.Linear(input_size, input_size, bias=False)
        self.label_representations = nn.Parameter(
            torch.rand(num_classes, input_size), requires_grad=True
        )
        self.weights_v = nn.Linear(input_size, input_size)
        self.output_linear = nn.Linear(input_size, 1)
        self.layernorm = nn.LayerNorm(input_size)
        self.num_classes = num_classes
        self._init_weights(mean=0.0, std=0.03)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attention: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Label Cross Attention mechanism
        Args:
            x (torch.Tensor): [batch_size, seq_len, input_size]
        Returns:
            tuple[torch.Tensor, Optional[torch.Tensor]]: [batch_size, num_classes], [batch_size, num_classes, seq_len]
        """

        # pad attention masks with 0 such that it has the same sequence length as x
        attention_mask = nn.functional.pad(
            attention_mask, (0, x.size(1) - attention_mask.size(1)), value=0
        )
        attention_mask = attention_mask.to(torch.bool)

        # repeat attention masks for each class
        attention_mask = attention_mask.unsqueeze(1).repeat(1, self.num_classes, 1)

        V = self.weights_v(x)
        K = self.weights_k(x)
        Q = self.label_representations

        if not return_attention:
            # This option is faster and requires less memory due to the use of flash attention
            y = torch.nn.functional.scaled_dot_product_attention(
                Q, K, V, scale=1, attn_mask=attention_mask
            )  # [batch_size, num_classes, input_size]
            attention = None
        else:
            # This option is slower and requires more memory due to the use of full attention
            # But lets you retrieve the attention weights
            scores = torch.matmul(Q, K.transpose(-2, -1))
            if attention_mask is not None:
                scores = scores.masked_fill(~attention_mask, float("-inf"))
            attention = torch.nn.functional.softmax(scores, dim=-1)
            y = torch.matmul(attention, V)  # [batch_size, num_classes, input_size]

        y = self.layernorm(y)
        # [batch_size, num_classes, input_size]

        output = self.output_linear(y).squeeze(-1)
        # [batch_size, num_classes]
        return output, attention

    def _init_weights(self, mean: float = 0.0, std: float = 0.03) -> None:
        """
        Initialise the weights
        Args:
            mean (float, optional): Mean of the normal distribution. Defaults to 0.0.
            std (float, optional): Standard deviation of the normal distribution. Defaults to 0.03.
        """

        self.weights_k.weight = torch.nn.init.normal_(self.weights_k.weight, mean, std)
        self.weights_v.weight = torch.nn.init.normal_(self.weights_v.weight, mean, std)
        self.label_representations = torch.nn.init.normal_(
            self.label_representations, mean, std
        )
        self.output_linear.weight = torch.nn.init.normal_(
            self.output_linear.weight, mean, std
        )
