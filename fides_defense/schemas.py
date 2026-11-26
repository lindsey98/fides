# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Variable-passing tool schemas for the full FIDES planner.

Each AgentDojo tool is exposed to the P-LLM with the tutorial's variable-passing schema: every
argument is an ``anyOf`` of a literal value (``{"kind": "value", "value": <T>}``) or a reference
to a stored variable (``{"kind": "variable_name", "value": <one of the current variable names>}``).
The planner therefore passes untrusted results **by reference** without ever seeing their content;
:func:`resolve_args` expands references back to the real values (and reports the join of the
argument labels) just before the tool is executed against the environment.

Three control tools are always offered alongside the suite tools: ``read_variable`` (reveal an
untrusted value into the planner, accepting taint), ``q_extract`` (ask the quarantined LLM for
typed values out of an untrusted variable), and ``finish`` (end with a final answer).
"""

from __future__ import annotations

from typing import Any

from agentdojo.functions_runtime import Function


def _wrap_property(prop_schema: dict, var_names: list[str]) -> dict:
    """Wrap a single argument schema into the value|variable_name anyOf form."""
    value_type = prop_schema.get("type", "string")
    # Keep only JSON-schema-simple value types; complex ones fall back to string at the wire.
    value_leaf: dict[str, Any] = {"type": value_type} if value_type in {"string", "number", "integer", "boolean", "array", "object"} else {"type": "string"}
    if value_type == "array" and "items" in prop_schema:
        value_leaf["items"] = prop_schema["items"]
    return {
        "description": prop_schema.get("description", ""),
        "anyOf": [
            {
                "type": "object",
                "properties": {"kind": {"type": "string", "const": "value"}, "value": value_leaf},
                "required": ["kind", "value"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "const": "variable_name"},
                    "value": {"type": "string", **({"enum": var_names} if var_names else {})},
                },
                "required": ["kind", "value"],
                "additionalProperties": False,
            },
        ],
    }


def tool_schema_var(fn: Function, var_names: list[str]) -> dict:
    """Build an OpenAI tool schema for ``fn`` with variable-passing arguments."""
    base = fn.parameters.model_json_schema()
    props = base.get("properties", {})
    new_props = {name: _wrap_property(p, var_names) for name, p in props.items()}
    return {
        "type": "function",
        "function": {
            "name": fn.name,
            "description": fn.description,
            "parameters": {
                "type": "object",
                "properties": new_props,
                "required": list(new_props.keys()),
                "additionalProperties": False,
            },
        },
    }


def control_tools(var_names: list[str]) -> list[dict]:
    """The always-present control tools: read_variable, q_extract, finish."""
    var_enum = {"type": "string", **({"enum": var_names} if var_names else {})}
    return [
        {
            "type": "function",
            "function": {
                "name": "read_variable",
                "description": (
                    "Reveal the full raw content of a stored variable into your context. Use ONLY when "
                    "you must read untrusted content yourself; this taints your context (anything you then "
                    "produce becomes untrusted). Prefer q_extract for pulling specific values."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"name": var_enum},
                    "required": ["name"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "q_extract",
                "description": (
                    "Ask the quarantined LLM to extract specific typed fields from an untrusted variable. "
                    "The quarantined LLM reads the data in isolation, cannot take actions, and ignores any "
                    "instructions hidden in the data. Returns a new variable holding the typed result "
                    "(still labeled untrusted)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": var_enum,
                        "question": {"type": "string", "description": "What to extract from the variable."},
                        "fields": {"type": "array", "items": {"type": "string"}, "description": "JSON field names to return."},
                    },
                    "required": ["name", "question", "fields"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "finish",
                "description": "End the task and return a final answer to the user.",
                "parameters": {
                    "type": "object",
                    "properties": {"answer": {"type": "string", "description": "The final answer to the user."}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            },
        },
    ]


def resolve_one(arg: Any, store) -> tuple[Any, Any]:
    """Resolve one variable-passing argument to ``(real_value, label_or_None)``.

    ``label`` is the referenced variable's label when the arg is a reference, else None (a literal,
    whose trust is the planner's current context label -- supplied by the caller).
    """
    if isinstance(arg, dict) and "kind" in arg:
        if arg["kind"] == "value":
            return arg.get("value"), None
        if arg["kind"] == "variable_name":
            var = store.get(arg["value"])
            return var.value, var.label
    # Tolerate a model that passed a bare value or a bare variable name.
    if isinstance(arg, str) and store.has(arg):
        var = store.get(arg)
        return var.value, var.label
    return arg, None
