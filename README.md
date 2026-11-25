# Securing AI Agents with Information-Flow Control

This repository contains a Jupyter notebook to accompany the academic paper "Securing AI Agents with Information-Flow Control".
The notebook is structured as a tutorial to walk readers through the concepts introduced in the paper.
This illustrates a practical implementation of mechanisms to deterministically enforce security policies in agentic tasks and grounds the concepts into executable code that readers can experiment with.

[![arXiv](https://img.shields.io/badge/arXiv-2505.23643-00ff00.svg)](https://arxiv.org/abs/2505.23643)

## Setup Instructions

The notebook has been tested against GPT-4o and GPT-4.1 using the Azure OpenAI Chat Completions API, but could be easily adapted to use an OpenAI endpoint instead.
The code sets up the model endpoint in the client using a `.env` configuration file containing definitions for `AZURE_ENDPOINT`, `API_VERSION` and `AZURE_DEPLOYMENT`.
Please copy the provided `.env.example` file to `.env` and edit as appropriate.
The notebook authenticates to the Azure OpenAI Service using Microsoft Entra ID, but could be simply adapted to use key-based authentication if preferred.

For convenience, we provide a fully-evaluated notebook to illustrate the intended output, but we encourage readers to set up their own endpoint to experiment hands-on and reproduce our the outputs.

## Running FIDES as an AgentDojo / AgentDyn defense baseline

Beyond the tutorial, this repository ships a runnable **defense baseline** that applies the
tutorial's information-flow-control (IFC) planning loop over the
[AgentDojo](https://github.com/ethz-spylab/agentdojo) benchmark and the
[AgentDyn](https://github.com/SaFo-Lab/AgentDyn) suites (`shopping`, `github`, `dailylife`).
It exposes the **same command-line interface** as the sibling prompt-injection baselines
(CaMeL, DRIFT), so results are directly comparable.

The IFC core (`fides_defense/lattice.py`, `fides_defense/core.py`) is a faithful port of the
tutorial — the lattices, labeled context, dynamic taint-tracking loop, and policies are not
re-designed. Only a thin adapter (`fides_defense/pipeline.py`) binds that loop to AgentDojo's
runtime, tools, and scoring harness.

### How the tutorial maps onto the benchmark

| Tutorial concept | Benchmark realization |
| --- | --- |
| `LabeledPlanningLoop` (taint-tracking + policy) | `FidesToolsLoop`, an AgentDojo pipeline element that replaces the stock tools loop |
| Integrity label by **source** (e.g. sender `@contoso.com` ⇒ trusted) | per-suite registry of **untrusted-source tools** in `fides_defense/policies.py` (emails, chat, web pages, reviews, files, search results ⇒ `UNTRUSTED`), with a name-pattern fallback for unlisted / AgentDyn tools |
| `policy_no_untrusted_*` raising `PolicyViolation` | a `Policy` evaluated **before** each tool call; a violation deterministically blocks the call (the model gets a refusal tool result, the sink never runs) |
| "Query / ToolCall / Response" actions | model turn delegated to AgentDojo's native-tool-calling LLM element; tool calls run via `runtime.run_function`; loop ends when the model stops calling tools |

Labels are assigned by **data source**, deterministically and independently of whether an
attack is present (the untrusted-source registry marks attacker-reachable channels per suite,
derived from AgentDojo's injection vectors). This is the "generic + extensible per-suite"
design: the per-suite registries are authoritative, and the generic heuristic is the fallback.

### Setup

`agentdojo` is installed **separately** from the lindsey98 / AgentDyn fork (a drop-in with the
same import name, bundling the original 4 suites **plus** `shopping`, `github`, `dailylife`,
and the extra attacks). `requirements.txt` does not pin it, so the last step never re-fetches
it over the network.

```bash
conda create -n fides python=3.11 && conda activate fides
git clone https://github.com/lindsey98/agentdojo.git
pip install -e ./agentdojo
pip install -r requirements.txt
cp .env.example .env          # fill in the key(s) for the provider you run (see the file)
set -a && source .env && set +a
```

Verify all 7 suites are registered:

```bash
python -c "from agentdojo.task_suite.load_suites import get_suites; print(sorted(get_suites('v1.2')))"
# -> ['banking', 'dailylife', 'github', 'shopping', 'slack', 'travel', 'workspace']
```

### Models

`MODEL` is positional; the provider is selected from its name (same routing as CaMeL / DRIFT):

| `MODEL` example | Provider / env |
| --- | --- |
| `gpt-4o-2024-08-06` | OpenAI (`OPENAI_API_KEY`, optional `OPENAI_BASE_URL`) |
| `openai:<any-name>` | OpenAI client forced for **any** model name — use this for a custom OpenAI-compatible endpoint whose served model is not named `gpt-*` (`OPENAI_API_KEY`, `OPENAI_BASE_URL`) |
| `anthropic:claude-sonnet-4-5-20250929` | Anthropic (`ANTHROPIC_API_KEY`) |
| `gemini-2.5-pro` | Google Vertex (`GCP_PROJECT`, `GCP_LOCATION` + ADC), or `OPENAI_BASE_URL` relay |
| `meta-llama/llama-3.3-70b-instruct` | OpenRouter (`OPENROUTER_API_KEY`) |
| `local:Qwen3-30B-A3B-Instruct-2507` or a bare served name | local OpenAI-compatible server (`LOCAL_API_BASE`, `LOCAL_API_KEY`) |

FIDES uses the model's **native** tool-calling API. For a local server without native tool
support, set `FIDES_LOCAL_PROMPTED=1` to use AgentDojo's prompted tool-call path instead.

> **Note (Anthropic):** AgentDojo's native `AnthropicLLM` targets the anthropic 0.x/1.x
> Messages API and breaks on `anthropic>=2.x` (where `MessageParam` is no longer
> instantiable). `requirements.txt` therefore caps `anthropic<2`. If you must use anthropic
> 2.x, route Claude through an OpenAI-compatible gateway with `openai:<name>` instead.

### How to run

```bash
# FIDES defense, under attack -> reports utility AND security (attack success rate):
python main.py gpt-4o-2024-08-06 --run-attack --attack important_instructions \
    --suite banking slack travel workspace --defense fides

# Undefended original model (the baseline to compare against):
python main.py gpt-4o-2024-08-06 --run-attack --attack important_instructions \
    --suite banking slack travel workspace --defense none

# Benign utility only (no attack):
python main.py gpt-4o-2024-08-06 --suite banking slack travel workspace --defense fides

# AgentDyn suites (require the AgentDyn fork installed, as above):
python main.py gpt-4o-2024-08-06 --run-attack --attack important_instructions \
    --suite shopping github dailylife --defense fides
```

Key flags (full list: `python main.py --help`):

| Flag | Meaning |
| --- | --- |
| `--defense {fides,none}` | `fides` = IFC taint-tracking + policy; `none` = undefended original model |
| `--run-attack` | run under attack and also report security; omit for benign utility only |
| `--attack NAME` | any registered AgentDojo attack (default `important_instructions`; also `tool_knowledge`, `ignore_previous`, `direct`, `dos`, `chat_inject_*`, …) |
| `--suite ...` | space-separated suites (default: the original four) |
| `--policy {no_untrusted_sink,no_untrusted_url,none}` | FIDES policy (see below) |
| `--user-task`, `--injection-task` | restrict to specific task ids/numbers |
| `--force-rerun` | re-run tasks even if a cached result exists |

### Policies

- **`no_untrusted_sink`** (default): once untrusted data has entered the context, **no sink
  tool** may be called — the deterministic integrity policy. An injected instruction can only
  cause harm through a sink, and under attack the context is untrusted, so every
  policy-violating sink call is stopped.
- **`no_untrusted_url`**: a faithful port of the tutorial's `policy_no_untrusted_url` — block a
  sink only when the context is untrusted **and** the arguments contain a URL (stops
  exfiltration via link preview / click), letting other sink calls through.
- **`none`**: taint-track only, never block (useful to measure the undefended planner).

Sink and untrusted-source classifications per suite live in `fides_defense/policies.py` and are
straightforward to extend for new suites or tools.

### Results

Per-suite and overall utility / security are printed as the run proceeds. AgentDojo's own
per-task JSON traces are written under `logs/<model>/<suite>/...`, and a compact machine-readable
summary per `(model, defense, policy, attack, suite)` under `logs/_fides_summary/`. Aggregate it
into a table with:

```bash
python print_results.py --logdir ./logs
```

## Citation

If you use wish to cite this work, please cite it as follows

```BibTeX
@misc{securing_ai_agents_with_ifc,
  title         = {Securing {AI} Agents with Information-Flow Control},
  author        = {Costa, Manuel and 
                   K{\"o}pf, Boris and 
                   Kolluri, Aashish and 
                   Paverd, Andrew and 
                   Russinovich, Mark and 
                   Salem, Ahmed and 
                   Tople, Shruti and 
                   Wutschitz, Lukas and 
                   Zanella-B{\'e}guelin, Santiago},
  year          = {2025},
  eprint        = {2505.23643},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CR},
  doi           = {10.48550/arXiv.2505.23643}
}
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.