# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Actions, labeled context, and the policy protocol.

Faithful port of the tutorial's planning-loop primitives (``Action`` / ``Query`` /
``ToolCall`` / ``Response``), the ``PolicyViolation`` exception, and the dynamic
taint-tracking "labeled trace" that a developer-defined ``Policy`` inspects.

The context label is a product of the standard integrity lattice {Trusted, Untrusted}
and the standard confidentiality lattice {Low, High} -- both defined in the tutorial.
The loop joins the label of every tool result into a running ``current_label`` and, after
each action, evaluates the ``Policy`` over the labeled trace; a ``Policy`` raises
``PolicyViolation`` to deterministically stop the action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from fides_defense.lattice import ConfidentialityLabel, IntegrityLabel, ProductLabel

# The context label: (integrity, confidentiality). Mirrors the tutorial's product of the
# standard integrity and confidentiality lattices.
Label = ProductLabel  # ProductLabel[IntegrityLabel, ConfidentialityLabel]


def trusted_public() -> Label:
    """Bottom label: trusted and public (the label of the user's own instruction)."""
    return ProductLabel(IntegrityLabel.trusted(), ConfidentialityLabel.low())


def untrusted_public() -> Label:
    return ProductLabel(IntegrityLabel.untrusted(), ConfidentialityLabel.low())


def label_is_untrusted(label: Label) -> bool:
    return isinstance(label.left, IntegrityLabel) and label.left.is_untrusted


# --- Actions (tutorial: "Automatic Planning Loop") --------------------------------------


@dataclass
class Action:
    pass


@dataclass
class Query(Action):
    """Ask the model for the next step (a model turn)."""

    # In the AgentDojo adapter the messages/tools live in the pipeline state, so this is a
    # marker action; kept for parity with the tutorial's Planning Loop.
    note: str = ""


@dataclass
class ToolCall(Action):
    """Execute one tool call requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class Response(Action):
    """Conclude the loop with a final response to the user."""

    response: str


# --- Policies (tutorial: "Dynamic Taint-Tracking") --------------------------------------


class PolicyViolation(Exception):
    """Raised by a policy to deterministically stop a policy-violating action."""

    def __init__(self, reason: str):
        super().__init__(f"Policy violation: {reason}")
        self.reason = reason


# A policy is a predicate over the labeled trace of actions. It may raise PolicyViolation.
# The trace entries are (action, label) pairs, where `label` is the context label at the
# time the action was taken (so a ToolCall's label reflects everything already in context).
Policy = Callable[[list[tuple[Action, Label]]], None]
