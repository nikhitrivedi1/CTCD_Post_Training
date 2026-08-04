"""Projection strategies for dimension alignment between teacher and model.

Used when teacher and model have different hidden dimensions.
For example: BERT-large (1024) -> DistilBERT (768) requires projection.

For BERT-base (768) -> DistilBERT (768), use IdentityProjection (no transform).
"""

import torch
import torch.nn as nn


class IdentityProjection:
    """
    No projection - pass through unchanged.

    Use when teacher and model have matching dimensions.
    Example: BERT-base (768) -> DistilBERT (768)
    """

    def __init__(self, dim: int | None = None):
        """
        Initialize identity projection.

        Args:
            dim: Hidden dimension (stored but not used for transform)
        """
        self.output_dim = dim

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Pass through unchanged."""
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass through unchanged."""
        return x


class LinearProjection(nn.Module):
    """
    Linear projection for dimension alignment.

    Projects teacher hidden states to match model dimension.
    Example: BERT-large (1024) -> DistilBERT (768)

    This is a learnable projection - the weights are trained during distillation.
    """

    def __init__(self, input_dim: int, output_dim: int):
        """
        Initialize linear projection.

        Args:
            input_dim: Teacher hidden dimension
            output_dim: Model hidden dimension
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input to output dimension."""
        # Cast input to match projector dtype (handles fp16 training)
        x = x.to(self.proj.weight.dtype)
        return self.proj(x)


class MLPProjection(nn.Module):
    """
    MLP projection with hidden layer for more expressive transform.

    Two-layer MLP: input_dim -> hidden_dim -> output_dim

    Use when simple linear projection is insufficient or when you want
    the projection to learn a more complex transformation.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
        activation: str = "relu",
    ):
        """
        Initialize MLP projection.

        Args:
            input_dim: Teacher hidden dimension
            output_dim: Model hidden dimension
            hidden_dim: Hidden layer dimension (default: average of input/output)
            activation: Activation function (relu, gelu, tanh)
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        hidden_dim = hidden_dim or (input_dim + output_dim) // 2

        # Select activation
        if activation == "relu":
            act = nn.ReLU()
        elif activation == "gelu":
            act = nn.GELU()
        elif activation == "tanh":
            act = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), act, nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input through MLP."""
        # Cast input to match projector dtype (handles fp16 training)
        x = x.to(self.proj[0].weight.dtype)
        return self.proj(x)


# Registry for factory lookup
PROJECTION_REGISTRY = {
    "identity": IdentityProjection,
    "linear": LinearProjection,
    "mlp": MLPProjection,
}


def get_projection(
    name: str, input_dim: int | None = None, output_dim: int | None = None, **kwargs
):
    """
    Factory function to get a projection strategy by name.

    Args:
        name: Strategy name (identity, linear, mlp)
        input_dim: Teacher hidden dimension
        output_dim: Model hidden dimension
        **kwargs: Additional arguments (hidden_dim for mlp)

    Returns:
        Projection strategy instance
    """
    if name not in PROJECTION_REGISTRY:
        raise ValueError(
            f"Unknown projection: {name}. Available: {list(PROJECTION_REGISTRY.keys())}"
        )

    if name == "identity":
        return IdentityProjection(dim=output_dim)
    elif name == "linear":
        if input_dim is None or output_dim is None:
            raise ValueError("linear projection requires input_dim and output_dim")
        return LinearProjection(input_dim, output_dim)
    elif name == "mlp":
        if input_dim is None or output_dim is None:
            raise ValueError("mlp projection requires input_dim and output_dim")
        return MLPProjection(input_dim, output_dim, **kwargs)

    return PROJECTION_REGISTRY[name]()


def create_projection_if_needed(
    teacher_dim: int, model_dim: int, projection_type: str = "linear", **kwargs
):
    """
    Create projection only if dimensions differ.

    Args:
        teacher_dim: Teacher hidden dimension
        model_dim: Model hidden dimension
        projection_type: Type of projection if needed (linear, mlp)
        **kwargs: Additional arguments for projection

    Returns:
        IdentityProjection if dims match, else specified projection type
    """
    if teacher_dim == model_dim:
        return IdentityProjection(dim=model_dim)

    return get_projection(projection_type, input_dim=teacher_dim, output_dim=model_dim, **kwargs)
