from overrides import overrides
import torch
from torch.nn import Dropout, Linear

from allennlp.common.registrable import Registrable
from allennlp.nn.util import masked_softmax, weighted_sum


class MultiHeadAttention(torch.nn.Module, Registrable):
    # pylint: disable=line-too-long
    """
    This class implements the key-value scaled dot product attention mechanism
    detailed in the paper `Attention is all you Need
    <https://www.semanticscholar.org/paper/Attention-Is-All-You-Need-Vaswani-Shazeer/0737da0767d77606169cbf4187b83e1ab62f6077>`_ .

    The attention mechanism is a weighted sum of a projection V of the inputs, with respect
    to the scaled, normalised dot product of Q and K, which are also both linear projections
    of the input. This procedure is repeated for each attention head, using different parameters.

    Parameters
    ----------
    num_heads : ``int``, required.
        The number of attention heads to use.
    input_dim : ``int``, required.
        The size of the last dimension of the input tensor.
    attention_dim ``int``, required.
        The total dimension of the query and key projections which comprise the
        dot product attention function. Must be divisible by ``num_heads``.
    values_dim : ``int``, required.
        The total dimension which the input is projected to for representing the values,
        which are combined using the attention. Must be divisible by ``num_heads``.
    output_projection_dim : ``int``, optional (default = None)
        The dimensionality of the final output projection. If this is not passed
        explicitly, the projection has size `input_size`.
    attention_dropout_prob : ``float``, optional (default = 0.1).
        The dropout probability applied to the normalised attention
        distributions.
    normalize : ``bool`` (default = True)
        Whether returned attentions are normalized


    Output:

    - values, attentions, attentions_raw: shape ``((batch_size, values_dim), (batch_size, num_heads, timesteps))``.
    """
    def __init__(self,
                 num_heads: int,
                 input_dim: int,
                 attention_dim: int,
                 values_dim: int,
                 output_projection_dim: int = None,
                 attention_dropout_prob: float = 0.1,
                 normalization_function: str = 'softmax',
                 normalize: bool = True) -> None:
        super().__init__()

        self._num_heads = num_heads
        self._input_dim = input_dim
        self._output_dim = output_projection_dim or input_dim
        self._attention_dim = attention_dim
        self._values_dim = values_dim
        self._normalize = normalize
        self._normalization_function = normalization_function
        valid_normalization_functions = ['softmax', 'sigmoid']
        if normalization_function not in valid_normalization_functions:
            raise ValueError(f"The normalization_function ({normalization_function}) is not in "
                             f"list of valid functions: {valid_normalization_functions}.")

        if attention_dim % num_heads != 0:
            raise ValueError(f"Key size ({attention_dim}) must be divisible by the number of "
                             f"attention heads ({num_heads}).")

        if values_dim % num_heads != 0:
            raise ValueError(f"Value size ({values_dim}) must be divisible by the number of "
                             f"attention heads ({num_heads}).")

        self._query_projection = Linear(input_dim, attention_dim)
        self._key_projection = Linear(input_dim, attention_dim)
        self._value_projection = Linear(input_dim, values_dim)

        self._scale = (input_dim // num_heads) ** 0.5
        self._output_projection = Linear(values_dim, self._output_dim)
        self._attention_dropout = Dropout(attention_dropout_prob)

    def get_input_dim(self):
        return self._input_dim

    def get_output_dim(self):
        return self._output_dim

    @overrides
    def forward(self,
                vector: torch.Tensor,
                matrix: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:

        num_heads = self._num_heads

        batch_size, timesteps, _ = matrix.size()

        if mask is None:
            mask = matrix.new_ones(batch_size, timesteps)

        # Shape (batch_size, attention_dim)
        query = self._query_projection(vector)
        # Shape (batch_size, timesteps, attention_dim)
        keys = self._key_projection(matrix)
        # Shape (batch_size, timesteps, values_dim)
        values = self._value_projection(matrix)

        # Shape (num_heads * batch_size, timesteps, values_dim / num_heads)
        values_per_head = values.view(batch_size, timesteps, num_heads, int(self._values_dim/num_heads))
        values_per_head = values_per_head.transpose(1, 2).contiguous()
        values_per_head = values_per_head.view(batch_size * num_heads, timesteps, int(self._values_dim/num_heads))

        # Shape (num_heads * batch_size, attention_dim / num_heads)
        query_per_head = query.view(batch_size * num_heads, int(self._attention_dim/num_heads))

        # Shape (num_heads * batch_size, timesteps, attention_dim / num_heads)
        keys_per_head = keys.view(batch_size, timesteps, num_heads, int(self._attention_dim/num_heads))
        keys_per_head = keys_per_head.transpose(1, 2).contiguous()
        keys_per_head = keys_per_head.view(batch_size * num_heads, timesteps, int(self._attention_dim/num_heads))

        # shape (num_heads * batch_size, 1, timesteps)
        scaled_similarities = torch.bmm(query_per_head.unsqueeze(1) / self._scale, keys_per_head.transpose(1, 2))

        mask_repeated = mask.repeat(1, num_heads).view(batch_size * num_heads, timesteps).float()
        # shape (num_heads * batch_size, 1, timesteps)
        # Normalise the distributions, using the same mask for all heads.
        if self._normalization_function == "softmax":
            attention_unnormalized = scaled_similarities * mask_repeated.unsqueeze(1)
            attention_full = masked_softmax(scaled_similarities,
                                           mask_repeated,
                                           memory_efficient=True)
        else:
            attention_unnormalized = torch.sigmoid(scaled_similarities) * mask_repeated.unsqueeze(1)
            attention_full = attention_unnormalized / (attention_unnormalized.sum(dim=2, keepdim=True) + 1e-13)

        attention = self._attention_dropout(attention_full)

        # Take a weighted sum of the values with respect to the attention
        # distributions for each element in the num_heads * batch_size dimension.
        # shape (num_heads * batch_size, 1, values_dim/num_heads)
        outputs = weighted_sum(values_per_head, attention).squeeze(1)

        outputs = outputs.view(batch_size, self._values_dim)

        # Project to final output dimension.
        # shape (batch_size, output_dim)
        outputs = self._output_projection(outputs)
        returned_attentions = attention_full if self._normalize else attention_unnormalized
        returned_attentions = returned_attentions.view(batch_size, num_heads, timesteps)
        attention_raw = scaled_similarities.view(batch_size, num_heads, timesteps)

        return outputs, returned_attentions, attention_raw
