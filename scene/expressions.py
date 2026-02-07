"""Shared utilities for safe math expression evaluation."""

import math
from typing import Dict


# Safe math namespace for eval expressions
_SAFE_MATH = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "sqrt": math.sqrt, "abs": abs, "pi": math.pi,
    "min": min, "max": max, "log": math.log, "exp": math.exp,
}


def safe_eval(expression: str, variables: Dict) -> float:
    """Evaluate a math expression with whitelisted functions and variables.

    Args:
        expression: String expression to evaluate (e.g., "sin(t * pi)")
        variables: Dictionary of variables available in the expression

    Returns:
        Result of evaluation as a float

    Raises:
        ValueError: If expression cannot be evaluated
    """
    ns = dict(_SAFE_MATH)
    ns.update(variables)
    try:
        return float(eval(expression, {"__builtins__": {}}, ns))
    except Exception as e:
        raise ValueError(f"Failed to evaluate expression '{expression}': {e}")
