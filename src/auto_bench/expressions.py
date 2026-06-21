from __future__ import annotations

import ast
import math
import operator
import re
from collections.abc import Callable, Mapping
from typing import Any, cast

from auto_bench.errors import ProtocolError

EXPR_RE = re.compile(r"^\$\{([^{}]+)\}$")
INTERPOLATION_RE = re.compile(r"\$\{([^{}]+)\}")


def slug(value: object) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9._=-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "value"


_FUNCTIONS = {
    "min": min,
    "max": max,
    "int": int,
    "str": str,
    "ceil": math.ceil,
    "floor": math.floor,
    "slug": slug,
}

_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_,
}

_COMPARE_OPS: dict[type[ast.cmpop], Callable[[Any, Any], bool]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}


def render_value(value: Any, context: Mapping[str, Any], path: str) -> Any:
    if isinstance(value, str):
        match = EXPR_RE.fullmatch(value)
        if match:
            return eval_expr(match.group(1), context, path)
        return INTERPOLATION_RE.sub(
            lambda m: str(eval_expr(m.group(1), context, path)), value
        )
    if isinstance(value, list):
        return [
            render_value(item, context, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        return {
            key: render_value(item, context, f"{path}.{key}" if path else str(key))
            for key, item in value.items()
        }
    return value


def eval_expr(source: str, context: Mapping[str, Any], path: str) -> Any:
    try:
        parsed = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise ProtocolError(f"{path}: invalid expression {source!r}") from exc
    return _eval(parsed.body, context, path)


def _eval(node: ast.AST, context: Mapping[str, Any], path: str) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in context:
            return context[node.id]
        raise ProtocolError(f"{path}: unknown name {node.id!r}")
    if isinstance(node, ast.Attribute):
        root, attrs = _attribute_path(node)
        if root not in {"metadata", "trtllm"}:
            raise ProtocolError(f"{path}: invalid reference root {root!r}")
        return _resolve_path(context[root], attrs, path)
    if isinstance(node, ast.BinOp):
        bin_op = _BIN_OPS.get(type(node.op))
        if bin_op is None:
            raise ProtocolError(f"{path}: unsupported arithmetic operator")
        return bin_op(
            _eval(node.left, context, path), _eval(node.right, context, path)
        )
    if isinstance(node, ast.UnaryOp):
        unary_op = _UNARY_OPS.get(type(node.op))
        if unary_op is None:
            raise ProtocolError(f"{path}: unsupported unary operator")
        return unary_op(_eval(node.operand, context, path))
    if isinstance(node, ast.BoolOp):
        values = [_eval(value, context, path) for value in node.values]
        if isinstance(node.op, ast.And):
            result = True
            for value in values:
                result = result and value
            return result
        if isinstance(node.op, ast.Or):
            result = False
            for value in values:
                result = result or value
            return result
        raise ProtocolError(f"{path}: unsupported boolean operator")
    if isinstance(node, ast.Compare):
        left = _eval(node.left, context, path)
        for op_node, comparator in zip(node.ops, node.comparators, strict=True):
            compare_op = _COMPARE_OPS.get(type(op_node))
            if compare_op is None:
                raise ProtocolError(f"{path}: unsupported comparison operator")
            right = _eval(comparator, context, path)
            if not compare_op(left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
            raise ProtocolError(f"{path}: unsupported function call")
        if node.keywords:
            raise ProtocolError(f"{path}: keyword arguments are not supported")
        args = [_eval(arg, context, path) for arg in node.args]
        function = cast(Callable[..., Any], _FUNCTIONS[node.func.id])
        return function(*args)
    if isinstance(node, ast.List):
        return [_eval(item, context, path) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_eval(item, context, path) for item in node.elts)
    raise ProtocolError(f"{path}: unsupported expression syntax")


def _attribute_path(node: ast.Attribute) -> tuple[str, list[str]]:
    attrs: list[str] = []
    current: ast.AST = node
    while isinstance(current, ast.Attribute):
        attrs.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        raise ProtocolError("invalid attribute reference")
    attrs.reverse()
    return current.id, attrs


def _resolve_path(value: Any, attrs: list[str], path: str) -> Any:
    current = value
    traversed: list[str] = []
    for attr in attrs:
        traversed.append(attr)
        if not isinstance(current, Mapping) or attr not in current:
            dotted = ".".join(traversed)
            raise ProtocolError(f"{path}: reference path {dotted!r} does not exist")
        current = current[attr]
    if isinstance(current, Mapping) and set(current) == {"sweep"}:
        raise ProtocolError(f"{path}: reference points to unresolved sweep object")
    return current
