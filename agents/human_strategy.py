"""把人类长局经验转换为可解释的盘面结构指标。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class HumanStructure:
    empty_cols: int
    inversions: int
    ordered_ratio: float
    top_pairs: int
    low_bottom_cols: int
    low_value_cols: int
    score: float


def human_structure_from_stacks(stacks):
    """从栈顶在前的列数组评估人类长局结构指标。"""
    empty_cols = sum(not stack for stack in stacks)
    inversions = 0
    comparable_pairs = 0
    top_pairs = 0
    low_bottom_cols = 0
    low_value_cols = 0
    nonempty_cols = 0
    for stack in stacks:
        if not stack:
            continue
        nonempty_cols += 1
        comparable_pairs += len(stack) * (len(stack) - 1) // 2
        inversions += sum(
            upper > lower
            for i, upper in enumerate(stack)
            for lower in stack[i + 1:]
        )
        top_pairs += int(len(stack) >= 2 and stack[0] == stack[1])
        low_bottom_cols += int(stack[-1] <= 2)
        low_value_cols += int(any(value <= 6 for value in stack))

    ordered_ratio = 1.0 - inversions / max(1, comparable_pairs)
    empty_reserve = min(2, empty_cols) / 2
    pair_ratio = min(3, top_pairs) / 3
    bottom_quality = 1.0 - low_bottom_cols / max(1, nonempty_cols)
    low_concentration = 1.0 - max(0, low_value_cols - 1) / max(1, len(stacks) - 1)
    score = (
        0.40 * ordered_ratio
        + 0.25 * bottom_quality
        + 0.15 * pair_ratio
        + 0.15 * empty_reserve
        + 0.05 * low_concentration
    )
    return HumanStructure(
        empty_cols=empty_cols,
        inversions=inversions,
        ordered_ratio=ordered_ratio,
        top_pairs=top_pairs,
        low_bottom_cols=low_bottom_cols,
        low_value_cols=low_value_cols,
        score=score,
    )


def human_structure(game):
    """评估列内顺序、空列、对子、底部锚点和低级元素集中度。"""
    return human_structure_from_stacks(game.stacks)
