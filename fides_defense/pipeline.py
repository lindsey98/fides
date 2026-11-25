# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""AgentDojo adapter for the FIDES dynamic taint-tracking planning loop.

``FidesToolsLoop`` is the AgentDojo ``BasePipelineElement`` equivalent of the tutorial's
``LabeledPlanningLoop``. It replaces the stock ``ToolsExecutionLoop([ToolsExecutor(), llm])``
with a loop that, for every tool call the model requests:

  1. forms a ``ToolCall`` action and appends ``(action, current_label)`` to a labeled trace;
  2. evaluates the ``Policy`` over the trace -- a ``PolicyViolation`` deterministically blocks
     the call (the model gets a refusal tool result instead of the real one);
  3. otherwise executes the tool via AgentDojo's runtime, then **joins** the result's label
     (UNTRUSTED iff the tool is an untrusted source for the suite) into ``current_label``.

The model turn ("Query" action) is delegated to the wrapped AgentDojo LLM element, exactly
as the tutorial's loop delegates to the OpenAI client. Blocking before execution (rather than
after) is what gives FIDES its deterministic guarantee: a policy-violating sink never runs.
"""

from __future__ import annotations

from ast import literal_eval
from collections.abc import Sequence
from typing import Any

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import (
    ChatMessage,
    ChatToolResultMessage,
    FunctionCall,
    text_content_block_from_string,
)

from fides_defense.core import (
    Label,
    Policy,
    PolicyViolation,
    ToolCall,
    trusted_public,
)
from fides_defense.lattice import ConfidentialityLabel, IntegrityLabel, ProductLabel
from fides_defense.policies import is_untrusted_source


def _is_string_list(s: str) -> bool:
    try:
        return isinstance(literal_eval(s), list)
    except (ValueError, SyntaxError):
        return False


def _result_label(suite: str, tool_name: str) -> Label:
    """Source-based label for a tool result (mirrors the tutorial's per-source labeling)."""
    integrity = IntegrityLabel.untrusted() if is_untrusted_source(suite, tool_name) else IntegrityLabel.trusted()
    return ProductLabel(integrity, ConfidentialityLabel.low())


class FidesToolsLoop(BasePipelineElement):
    def __init__(
        self,
        llm: BasePipelineElement,
        suite_name: str,
        policy: Policy,
        max_iters: int = 15,
    ) -> None:
        self.llm = llm
        self.suite_name = suite_name
        self.policy = policy
        self.max_iters = max_iters

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        # The user's own instruction is the trusted, public bottom of the lattice.
        current_label: Label = trusted_public()
        trace: list[tuple[Any, Label]] = []
        events: list[dict[str, Any]] = extra_args.setdefault("fides_events", [])

        for _ in range(self.max_iters):
            if len(messages) == 0:
                break
            last = messages[-1]
            if last["role"] != "assistant":
                break
            tool_calls = last.get("tool_calls")
            if not tool_calls:
                break  # Response action: the loop concludes.

            tool_results: list[ChatToolResultMessage] = []
            for tool_call in tool_calls:
                result_msg = self._handle_tool_call(
                    tool_call, runtime, env, trace, current_label, events
                )
                tool_results.append(result_msg)
                # Join the executed result's label into the context (skip blocked calls).
                if not (result_msg.get("error") or "").startswith("PolicyViolation"):
                    current_label = current_label.join(_result_label(self.suite_name, tool_call.function))

            messages = [*messages, *tool_results]

            # "Query" action: ask the model for the next step.
            query, runtime, env, messages, extra_args = self.llm.query(
                query, runtime, env, messages, extra_args
            )

        return query, runtime, env, messages, extra_args

    def _handle_tool_call(
        self,
        tool_call: FunctionCall,
        runtime: FunctionsRuntime,
        env: Env,
        trace: list[tuple[Any, Label]],
        current_label: Label,
        events: list[dict[str, Any]],
    ) -> ChatToolResultMessage:
        name = tool_call.function
        args = dict(tool_call.args)

        # Coerce stringified list args, matching AgentDojo's ToolsExecutor.
        for k, v in args.items():
            if isinstance(v, str) and _is_string_list(v):
                args[k] = literal_eval(v)

        action = ToolCall(id=tool_call.id or "", name=name, arguments=args)
        trace.append((action, current_label))

        # --- Policy check (before execution: this is the deterministic enforcement) ---
        try:
            self.policy(trace)
        except PolicyViolation as pv:
            events.append(
                {"type": "blocked", "tool": name, "label": repr(current_label), "reason": pv.reason}
            )
            return ChatToolResultMessage(
                role="tool",
                content=[
                    text_content_block_from_string(
                        f"[FIDES policy block] Refused to execute `{name}`: {pv.reason}. "
                        "This action was stopped by the information-flow-control policy."
                    )
                ],
                tool_call_id=tool_call.id,
                tool_call=tool_call,
                error=f"PolicyViolation: {pv.reason}",
            )

        # --- Execute the tool via AgentDojo's runtime ---
        result, error = runtime.run_function(env, name, args)
        events.append(
            {
                "type": "executed",
                "tool": name,
                "label": repr(current_label),
                "result_label": repr(_result_label(self.suite_name, name)),
                "error": error,
            }
        )
        return ChatToolResultMessage(
            role="tool",
            content=[text_content_block_from_string(tool_result_to_str(result))],
            tool_call_id=tool_call.id,
            tool_call=tool_call,
            error=error,
        )


def make_fides_pipeline(
    llm: BasePipelineElement,
    llm_name: str,
    suite_name: str,
    policy: Policy,
    defense: str,
    system_message: str | None = None,
    max_iters: int = 15,
) -> BasePipelineElement:
    """Build the AgentDojo pipeline for FIDES (``defense="fides"``) or the undefended
    original model (``defense="none"``).

    FIDES: ``[SystemMessage, InitQuery, llm, FidesToolsLoop(llm, policy)]`` -- same shape as
    the stock pipeline but with the taint-tracking/policy loop swapped in for the plain
    tools loop. ``none`` reproduces the stock AgentDojo pipeline (no IFC, no policy).
    """
    from agentdojo.agent_pipeline import (
        AgentPipeline,
        InitQuery,
        SystemMessage,
        ToolsExecutionLoop,
        ToolsExecutor,
    )
    from agentdojo.agent_pipeline.agent_pipeline import load_system_message

    sys_msg = system_message if system_message is not None else load_system_message(None)
    elements = [SystemMessage(sys_msg), InitQuery(), llm]

    if defense == "fides":
        elements.append(FidesToolsLoop(llm, suite_name, policy, max_iters=max_iters))
    else:  # "none": undefended original model
        elements.append(ToolsExecutionLoop([ToolsExecutor(), llm], max_iters=max_iters))

    pipeline = AgentPipeline(elements)
    pipeline.name = llm_name
    return pipeline
