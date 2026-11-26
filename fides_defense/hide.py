# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Node-level hiding of a tool result (the paper's Hide).

The paper hides only the parts of a value whose label is *stricter* than the current context:
it recursively walks the value and replaces nodes labeled untrusted with variables, while the
trusted parts stay visible to the planner. Labels are field-level (a field attacked by an
injection task is untrusted -- e.g. an email ``body`` -- but the trusted metadata around it, like
the sender address, remains visible).

:func:`hide_result` walks a tool result and returns a redacted rendering in which each untrusted
field is replaced by a fresh variable handle (so the planner can pass it by reference or send it to
the quarantined LLM), while trusted fields are shown inline. A bare-string result (web page, file,
review text) has no field structure and is hidden whole.
"""

from __future__ import annotations

from typing import Any

from fides_defense.core import Label
from fides_defense.policies import field_is_untrusted
from fides_defense.varstore import VarStore, describe_value

_MAX_LIST = 12
_MAX_LEAF = 200


def _to_plain(x: Any) -> Any:
    """Convert BaseModels (recursively) to plain JSON-ish dict/list/scalars."""
    from pydantic import BaseModel

    if isinstance(x, BaseModel):
        return x.model_dump(mode="json")
    if isinstance(x, list):
        return [_to_plain(i) for i in x]
    if isinstance(x, dict):
        return {k: _to_plain(v) for k, v in x.items()}
    return x


def hide_result(suite: str, result: Any, store: VarStore, untrusted_label: Label) -> str:
    """Render ``result`` for the planner, hiding untrusted fields as fresh variables.

    Returns the redacted rendering string. Side effect: creates one variable per hidden node in
    ``store`` (untrusted, content hidden), so the planner can reference it. A bare-string / scalar
    result from an untrusted source is hidden whole.
    """
    data = _to_plain(result)

    def hide_node(v: Any, origin: str) -> str:
        var = store.put(v, untrusted_label, origin=origin, show_value=False)
        return f"<{var.name}: {describe_value(v)}, UNTRUSTED>"

    def walk(v: Any, key: str | None, top: bool) -> str:
        # A field whose key is untrusted: hide the whole subtree as one variable.
        if key is not None and field_is_untrusted(suite, key):
            return hide_node(v, origin=key)
        if isinstance(v, dict):
            return "{" + ", ".join(f"{k}: {walk(val, k, False)}" for k, val in list(v.items())[:30]) + "}"
        if isinstance(v, list):
            shown = [walk(i, None, False) for i in v[:_MAX_LIST]]
            more = f", …(+{len(v) - _MAX_LIST} more)" if len(v) > _MAX_LIST else ""
            return "[" + ", ".join(shown) + more + "]"
        # A bare leaf. At the top level (no enclosing field name) it is untrusted content with no
        # field structure -> hide whole. Otherwise it is trusted metadata -> show it.
        if top:
            return hide_node(v, origin="result")
        s = str(v)
        return s if len(s) <= _MAX_LEAF else s[:_MAX_LEAF] + "…"

    return walk(data, None, True)
