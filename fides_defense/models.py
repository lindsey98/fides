# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Model routing: map a ``--model`` name onto an AgentDojo LLM pipeline element.

FIDES (per the tutorial) relies on the model's **native** tool-calling API with strict
JSON schemas, so every backend here returns an AgentDojo LLM element that speaks native
function calling. Routing mirrors the sibling baselines (CaMeL / DRIFT) so a model name
resolves identically across harnesses:

    gpt-*                     -> OpenAILLM            (OPENAI_API_KEY[, OPENAI_BASE_URL])
    gemini-*                  -> GoogleLLM (Vertex), or OpenAILLM if OPENAI_BASE_URL set
    anthropic:* | claude*     -> AnthropicLLM         (ANTHROPIC_API_KEY)
    org/model (has a slash)   -> OpenAILLM @ OpenRouter(OPENROUTER_API_KEY)
    local:* | bare name       -> OpenAILLM @ LOCAL_API_BASE (native tools), or prompted
                                 LocalLLM if FIDES_LOCAL_PROMPTED=1
"""

from __future__ import annotations

import logging
import os

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement

logger = logging.getLogger("fides")


def build_llm(model_name: str) -> BasePipelineElement:
    """Return the AgentDojo LLM pipeline element for ``model_name``.

    The pipeline keeps ``model_name`` as its name, so AgentDojo's attacks address the
    right victim model where the id is recognized and fall back to a generic name
    otherwise (the fork's ``get_model_name_from_pipeline`` degrades gracefully).
    """
    import openai

    name = model_name.lower()

    # Explicit OpenAI-compatible route: `openai:<name>` forces the OpenAI client (which
    # honors OPENAI_API_KEY + OPENAI_BASE_URL) for ANY model name -- use this for a custom
    # OpenAI endpoint whose served model is not named `gpt-*`.
    if model_name.startswith("openai:"):
        from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM

        return OpenAILLM(openai.OpenAI(), model_name.split("openai:", 1)[-1])

    if model_name.startswith("gpt-"):
        from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM

        return OpenAILLM(openai.OpenAI(), model_name)  # honors OPENAI_API_KEY / OPENAI_BASE_URL

    if model_name.startswith("gemini-"):
        if os.getenv("OPENAI_BASE_URL"):
            from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM

            return OpenAILLM(openai.OpenAI(), model_name)
        from google import genai

        from agentdojo.agent_pipeline.llms.google_llm import GoogleLLM

        client = genai.Client(
            vertexai=True, project=os.getenv("GCP_PROJECT"), location=os.getenv("GCP_LOCATION")
        )
        return GoogleLLM(model_name, client)

    if model_name.startswith("anthropic:") or name.startswith("claude"):
        import anthropic

        from agentdojo.agent_pipeline.llms.anthropic_llm import AnthropicLLM

        model = model_name.split("anthropic:", 1)[-1]
        return AnthropicLLM(anthropic.Anthropic(), model)

    if model_name.startswith("local:") or "/" not in model_name:
        model = model_name.split("local:", 1)[-1]
        base_url = os.getenv("LOCAL_API_BASE", "http://localhost:8000/v1")
        key = os.getenv("LOCAL_API_KEY", "EMPTY")
        import httpx

        client = openai.OpenAI(api_key=key, base_url=base_url, http_client=httpx.Client(trust_env=False))
        if os.getenv("FIDES_LOCAL_PROMPTED") == "1":
            # Prompted tool-call path for servers without native tool support.
            from agentdojo.agent_pipeline.llms.local_llm import LocalLLM

            fmt = os.getenv("LOCAL_TOOL_CALL_FORMAT") or ("python" if "gemma" in model.lower() else "json")
            return LocalLLM(client, model, tool_call_format=fmt)
        from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM

        return OpenAILLM(client, model)

    # Has a slash and no cloud prefix -> OpenRouter (org/model).
    from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM

    client = openai.OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1"
    )
    return OpenAILLM(client, model_name)


def build_raw_client(model_name: str):
    """Return ``(openai_compatible_client, model_id)`` for the full FIDES planner.

    The full planner drives the client directly (variable-passing tool schemas + structured
    outputs), which needs an OpenAI-compatible endpoint. Supported routes: ``openai:<name>``,
    ``gpt-*`` (OPENAI_API_KEY[/OPENAI_BASE_URL]), ``org/model`` (OpenRouter), and
    ``local:<name>`` / a bare served name (LOCAL_API_BASE). Anthropic/Google native models are
    not supported here -- use ``--planner labeled_basic`` for those, or route them through an
    OpenAI-compatible gateway (``openai:<name>`` + OPENAI_BASE_URL).
    """
    import openai

    name = model_name.lower()
    if model_name.startswith("openai:"):
        return openai.OpenAI(), model_name.split("openai:", 1)[-1]
    if model_name.startswith("gpt-"):
        return openai.OpenAI(), model_name
    if model_name.startswith("gemini-") and os.getenv("OPENAI_BASE_URL"):
        return openai.OpenAI(), model_name
    if model_name.startswith(("anthropic:", "claude")) or name.startswith("gemini-"):
        raise ValueError(
            f"--planner full needs an OpenAI-compatible endpoint; `{model_name}` is a native "
            "Anthropic/Google model. Use --planner labeled_basic, or route it through an "
            "OpenAI-compatible gateway as openai:<name> with OPENAI_BASE_URL set."
        )
    if model_name.startswith("local:") or "/" not in model_name:
        import httpx

        model = model_name.split("local:", 1)[-1]
        client = openai.OpenAI(
            api_key=os.getenv("LOCAL_API_KEY", "EMPTY"),
            base_url=os.getenv("LOCAL_API_BASE", "http://localhost:8000/v1"),
            http_client=httpx.Client(trust_env=False),
        )
        return client, model
    # org/model -> OpenRouter
    return openai.OpenAI(api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1"), model_name
