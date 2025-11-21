"""
Generic utility tools shared across agents.
"""

from __future__ import annotations

from typing import List, Sequence

from langchain_core.tools import tool

CALCULATOR_TOOL_NAME = "calculator"


def canonical_operation(operation: str) -> str:
    """
    Normalize and validate the requested arithmetic operation.
    """
    if not operation:
        return ""
    op = operation.strip().lower()
    alias_map = {
        "add": "add",
        "addition": "add",
        "sum": "add",
        "+": "add",
        "subtract": "subtract",
        "subtraction": "subtract",
        "minus": "subtract",
        "-": "subtract",
        "multiply": "multiply",
        "multiplication": "multiply",
        "product": "multiply",
        "*": "multiply",
        "divide": "divide",
        "division": "divide",
        "/": "divide",
    }
    return alias_map.get(op, "")


def execute_operation(operation: str, operands: Sequence[float]) -> float:
    """
    Run the arithmetic operation against the provided operands.
    """
    if operation == "add":
        return float(sum(operands))

    if operation == "subtract":
        total = float(operands[0])
        for operand in operands[1:]:
            total -= float(operand)
        return total

    if operation == "multiply":
        total = float(operands[0])
        for operand in operands[1:]:
            total *= float(operand)
        return total

    if operation == "divide":
        total = float(operands[0])
        for operand in operands[1:]:
            if operand == 0:
                raise ValueError("division by zero")
            total /= float(operand)
        return total

    raise ValueError(f"unsupported operation '{operation}'")


@tool(CALCULATOR_TOOL_NAME)
def calculator(operation: str, operands: List[float]) -> dict:
    """
    Perform addition, subtraction, multiplication, or division on numeric operands.

    Args:
        operation: Operation to run (add, subtract, multiply, divide). Aliases accepted: +, -, *, /.
        operands: List of numbers (at least two) to apply the operation on.

    Returns:
        Dict containing the canonicalized operation, original operands, and computed result.
    """
    if operands is None:
        raise ValueError("operands are required")
    op = canonical_operation(operation)
    if not op:
        raise ValueError("operation is required and must be one of add, subtract, multiply, divide")
    if len(operands) < 2:
        raise ValueError("operands must contain at least two numbers")

    result = execute_operation(op, operands)
    return {
        "operation": op,
        "operands": operands,
        "result": result,
    }


# Convenience export for agent wiring.
COMMON_TOOLS = [calculator]
