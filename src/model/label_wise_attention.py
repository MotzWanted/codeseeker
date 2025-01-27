import torch
import torch.nn as nn


class LabelCrossAttention(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.weights_k = nn.Linear(input_size, input_size, bias=False)
        self.label_representations = torch.nn.Parameter(torch.rand(num_classes, input_size), requires_grad=True)
        self.weights_v = nn.Linear(input_size, input_size)
        self.output_linear = nn.Linear(input_size, 1)
        self.layernorm = nn.LayerNorm(input_size)
        self.num_classes = num_classes
        self._init_weights(mean=0.0, std=0.03)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Label Cross Attention mechanism

        Args:
            x (torch.Tensor): [batch_size, seq_len, input_size]

        Returns:
            torch.Tensor: [batch_size, num_classes]
        """
        # pad attention masks with 0 such that it has the same sequence lenght as x
        attention_mask = torch.nn.functional.pad(attention_mask, (0, x.size(1) - attention_mask.size(1)), value=0)
        attention_mask = attention_mask.to(torch.bool)
        # repeat attention masks for each class
        attention_mask = attention_mask.unsqueeze(1).repeat(1, self.num_classes, 1)

        V = self.weights_v(x)
        K = self.weights_k(x)
        Q = self.label_representations

        # This option is faster and requires less memory due to the use of flash attention
        y = torch.nn.functional.scaled_dot_product_attention(
            Q, K, V, scale=1, attn_mask=attention_mask
        )  # [batch_size, num_classes, input_size]
        y = self.layernorm(y)
        # [batch_size, num_classes, input_size]

        output = self.output_linear(y).squeeze(-1)
        # [batch_size, num_classes]
        return output

    def _init_weights(self, mean: float = 0.0, std: float = 0.03) -> None:
        """
        Initialise the weights

        Args:
            mean (float, optional): Mean of the normal distribution. Defaults to 0.0.
            std (float, optional): Standard deviation of the normal distribution. Defaults to 0.03.
        """

        self.weights_k.weight = torch.nn.init.normal_(self.weights_k.weight, mean, std)
        self.weights_v.weight = torch.nn.init.normal_(self.weights_v.weight, mean, std)
        self.label_representations = torch.nn.init.normal_(self.label_representations, mean, std)
        self.output_linear.weight = torch.nn.init.normal_(self.output_linear.weight, mean, std)
