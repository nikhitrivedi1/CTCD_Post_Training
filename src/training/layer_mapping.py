"""Layer mapping strategies for teacher-student correspondence.

Maps teacher layers to student layers for feature and attention distillation.
Used when teacher has more layers than student (e.g., BERT-12 -> DistilBERT-6).
"""

from typing import List, Tuple


class UniformMappingStrategy:
    """
    Map every Nth teacher layer to student layers.

    Example (12 teacher -> 6 student):
        Teacher layers: [1, 3, 5, 7, 9, 11] -> Student layers: [0, 1, 2, 3, 4, 5]
        (every 2nd teacher layer)
    """

    def map(self, teacher_layers: int, student_layers: int) -> List[Tuple[int, int]]:
        """Return list of (teacher_idx, student_idx) pairs."""
        if student_layers == 0:
            return []
        step = teacher_layers // student_layers
        return [(i * step + step - 1, i) for i in range(student_layers)]

    def validate(self, teacher_layers: int, student_layers: int) -> bool:
        """Check if mapping is valid for given layer counts."""
        return teacher_layers >= student_layers and student_layers > 0


class LastNMappingStrategy:
    """
    Map last N teacher layers to student layers.

    Example (12 teacher -> 6 student):
        Teacher layers: [6, 7, 8, 9, 10, 11] -> Student layers: [0, 1, 2, 3, 4, 5]
        (last 6 teacher layers)

    Rationale: Later layers often contain more task-specific information.
    """

    def map(self, teacher_layers: int, student_layers: int) -> List[Tuple[int, int]]:
        """Return list of (teacher_idx, student_idx) pairs."""
        if student_layers == 0:
            return []
        offset = teacher_layers - student_layers
        return [(offset + i, i) for i in range(student_layers)]

    def validate(self, teacher_layers: int, student_layers: int) -> bool:
        """Check if mapping is valid for given layer counts."""
        return teacher_layers >= student_layers and student_layers > 0


class FirstLastMappingStrategy:
    """
    Emphasize first and last layers, interpolate middle.

    Example (12 teacher -> 6 student):
        Teacher layers: [0, 2, 4, 7, 9, 11] -> Student layers: [0, 1, 2, 3, 4, 5]
        (first, last, and evenly spaced middle)

    Rationale: First layers capture low-level features, last layers capture
    high-level semantics. Both are important for knowledge transfer.
    """

    def map(self, teacher_layers: int, student_layers: int) -> List[Tuple[int, int]]:
        """Return list of (teacher_idx, student_idx) pairs."""
        if student_layers == 0:
            return []
        if student_layers == 1:
            return [(teacher_layers - 1, 0)]
        if student_layers == 2:
            return [(0, 0), (teacher_layers - 1, 1)]

        # First and last, then evenly spaced middle
        indices = [0]
        middle_count = student_layers - 2
        step = (teacher_layers - 1) / (student_layers - 1)
        for i in range(1, middle_count + 1):
            indices.append(int(i * step))
        indices.append(teacher_layers - 1)

        return [(indices[i], i) for i in range(student_layers)]

    def validate(self, teacher_layers: int, student_layers: int) -> bool:
        """Check if mapping is valid for given layer counts."""
        return teacher_layers >= student_layers and student_layers > 0


class CustomMappingStrategy:
    """
    User-defined explicit layer mapping.

    Example:
        CustomMappingStrategy(
            teacher_indices=[1, 3, 5, 7, 9, 11],
            student_indices=[0, 1, 2, 3, 4, 5]
        )

    Use when you want precise control over which teacher layers
    correspond to which student layers.
    """

    def __init__(self, teacher_indices: List[int], student_indices: List[int]):
        """
        Initialize with explicit mapping.

        Args:
            teacher_indices: List of teacher layer indices (0-indexed)
            student_indices: List of student layer indices (0-indexed)
        """
        if len(teacher_indices) != len(student_indices):
            raise ValueError(
                f"teacher_indices ({len(teacher_indices)}) and student_indices "
                f"({len(student_indices)}) must have same length"
            )
        self.teacher_indices = teacher_indices
        self.student_indices = student_indices
        self.mapping = list(zip(teacher_indices, student_indices))

    def map(self, teacher_layers: int, student_layers: int) -> List[Tuple[int, int]]:
        """Return the pre-configured mapping."""
        return self.mapping

    def validate(self, teacher_layers: int, student_layers: int) -> bool:
        """Check if all indices are within bounds."""
        return all(
            0 <= t < teacher_layers and 0 <= s < student_layers
            for t, s in self.mapping
        )


# Registry for factory lookup
LAYER_MAPPING_REGISTRY = {
    "uniform": UniformMappingStrategy,
    "last_n": LastNMappingStrategy,
    "first_last": FirstLastMappingStrategy,
    "custom": CustomMappingStrategy,
}


def get_layer_mapping(name: str, **kwargs):
    """
    Factory function to get a layer mapping strategy by name.

    Args:
        name: Strategy name (uniform, last_n, first_last, custom)
        **kwargs: Arguments for custom mapping (teacher_indices, student_indices)

    Returns:
        Layer mapping strategy instance
    """
    if name not in LAYER_MAPPING_REGISTRY:
        raise ValueError(
            f"Unknown layer mapping: {name}. "
            f"Available: {list(LAYER_MAPPING_REGISTRY.keys())}"
        )

    if name == "custom":
        return CustomMappingStrategy(**kwargs)
    return LAYER_MAPPING_REGISTRY[name]()
