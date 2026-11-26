# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Labeled variable memory for the full FIDES planner.

Every tool result is stored here as a :class:`Variable` carrying the *real* value (used when
resolving a later tool call's arguments and when scoring the AgentDojo environment) plus its
information-flow :class:`~fides_defense.core.Label`. The privileged planner (P-LLM) is shown
only a **handle** for each variable -- its name, a short type/shape descriptor, and its label
-- never the raw content of an untrusted value. This is the tutorial's variable-passing idea:
untrusted data stays out of the planner's context and flows by reference, so an injected
instruction hidden in a tool result can never reach the planner to be followed.

Trusted values may optionally be inlined into the handle (the planner is allowed to see its
own / trusted data); untrusted values are always shown as an opaque handle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fides_defense.core import Label, label_is_untrusted


def describe_value(value: Any, max_len: int = 200) -> str:
    """A short, structural description of a value (type + shape), with no untrusted content."""
    if isinstance(value, str):
        return f"str(len={len(value)})"
    if isinstance(value, (list, tuple)):
        inner = type(value[0]).__name__ if value else "empty"
        return f"list(len={len(value)}, of={inner})"
    if isinstance(value, dict):
        return f"dict(keys={sorted(value.keys())[:8]})"
    if isinstance(value, bool):
        return f"bool({value})"
    if isinstance(value, (int, float)):
        return f"{type(value).__name__}({value})"
    s = str(value)
    return f"{type(value).__name__}" + (f"({s[:max_len]})" if len(s) <= max_len else f"(len={len(s)})")


@dataclass
class Variable:
    name: str
    value: Any                     # the real value (resolved into tool calls; never shown raw if hidden)
    label: Label
    origin: str = ""               # e.g. "read_inbox" or "q_extract" -- for the handle / logs
    show_value: bool | None = None  # show the value inline in the handle; default: only if trusted.
    render: str | None = None      # explicit handle body (node-level-hidden structure: trusted fields
    #                                visible, untrusted fields shown as sub-variable handles).
    # q_extract results are untrusted but show_value=True: the quarantined LLM already reduced the
    # raw content to constrained typed fields, so they are safe to show as DATA -- and any USE of
    # them still goes by labeled reference, so sinks remain policy-governed.

    @property
    def untrusted(self) -> bool:
        return label_is_untrusted(self.label)

    @property
    def _shown(self) -> bool:
        return self.show_value if self.show_value is not None else (not self.untrusted)

    def handle(self) -> str:
        """The one-line view the P-LLM sees: name, shape, label, and (when shown) the value."""
        tag = "UNTRUSTED" if self.untrusted else "TRUSTED"
        shape = describe_value(self.value)
        base = f"{self.name} = {self.origin or 'value'}  # {shape}, {tag}"
        if self.render is not None:
            base += f"\n    {self.render}"
        elif self._shown:
            v = str(self.value)
            base += f"\n    value: {v if len(v) <= 500 else v[:500] + ' …(truncated)'}"
        else:
            base += "\n    (untrusted content hidden; use q_extract to pull typed values, or read_variable to reveal)"
        return base


@dataclass
class VarStore:
    _vars: dict[str, Variable] = field(default_factory=dict)
    _counter: int = 0

    def new_name(self) -> str:
        self._counter += 1
        return f"r{self._counter}"

    def put(self, value: Any, label: Label, origin: str = "", name: str | None = None,
            show_value: bool | None = None, render: str | None = None) -> Variable:
        name = name or self.new_name()
        var = Variable(name=name, value=value, label=label, origin=origin, show_value=show_value, render=render)
        self._vars[name] = var
        return var

    def get(self, name: str) -> Variable:
        if name not in self._vars:
            raise KeyError(f"Unknown variable: {name}")
        return self._vars[name]

    def has(self, name: str) -> bool:
        return name in self._vars

    def names(self) -> list[str]:
        return list(self._vars)

    def handles(self) -> str:
        """The full variable table shown to the P-LLM (handles only)."""
        if not self._vars:
            return "(no variables yet)"
        return "\n".join(v.handle() for v in self._vars.values())
