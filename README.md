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

The IFC core (`fides_defense/lattice.py`, `fides_defense/core.py`) ports the tutorial's
lattices and labeled context. Two planners are provided (`--planner`):

- **`full`** (default) — the paper's planner: **variable hiding** (the privileged planner
  never sees raw untrusted content; tool results are stored as labeled variables and passed
  by reference), a **quarantined LLM** (reads untrusted data in isolation and returns typed
  values), and a **per-argument policy** (a sink is blocked only when untrusted data actually
  flows *into it*). This recovers utility: reading an untrusted email no longer blocks an
  unrelated trusted action. It drives an OpenAI-compatible client directly and lives in
  `fides_defense/planner_full.py` / `varstore.py` / `quarantined.py` / `schemas.py`.
- **`labeled_basic`** — the tutorial's `LabeledBasicPlanner`: context-level dynamic
  taint-tracking + policy, a faithful port of the notebook (`fides_defense/pipeline.py`).
  Works with every provider (native tool-calling via AgentDojo's LLM layer).

> **Scope note.** The tutorial itself only ships the base `LabeledBasicPlanner` (and a
> *non*-IFC variable-passing planner, shown as the attackable example). The `full` planner
> combines variable passing + labels + a quarantined LLM per the **paper**, so it goes beyond
> the notebook's code. Use `labeled_basic` if you want the strictly notebook-faithful version;
> its utility is a lower bound (no hiding / quarantine).

### How the tutorial / paper map onto the benchmark

| FIDES concept | Benchmark realization |
| --- | --- |
| **Static, source-based labels — at the field level** | Labels come from a tool's trusted wrapper, per the paper: a *field* attacked by at least one injection task is untrusted (e.g. an email `body`), while trusted metadata (the sender address) stays visible. `policies.py` holds a per-suite registry of **untrusted-source tools** (a tool surfaces untrusted data only if it carries **third-party-authored** content — inbound mail, chat from others, web pages, reviews, URL downloads, shared docs, calendar invites; the user's own cart/orders/account info/own files/sent mail/contacts/own calendar are **trusted**) and a per-suite set of **untrusted fields**; external data with no signal defaults to untrusted. Grounded in each suite's injection vectors (`threat_model.VECTOR_CLASS`). |
| **Variable hiding — node-level** (`hide.py`) | `full` planner: a tool result is recursively walked and only the nodes labeled **stricter than the context** (its untrusted fields) are replaced by fresh hidden variables; trusted parts stay visible. A bare-string result (web page, file) is hidden whole. The planner passes variables by reference via the tutorial's `anyOf[value, variable_name]` schema (`schemas.py`) and never reads untrusted content. |
| **Quarantined LLM** | `q_extract` (`quarantined.py`): a separate, tool-less model call reads one untrusted variable in isolation, ignores instructions in the data, and returns typed fields (labeled untrusted) |
| **Policy** raising `PolicyViolation` | `full`: per-argument check before a sink runs (`check_sink_call`) — blocks only when untrusted data flows into the sink. `labeled_basic`: context-level check before each tool call. A blocked sink never executes (and is excluded from the scored call trace). |
| Query / ToolCall / Response | `full` drives an OpenAI-compatible client directly; `labeled_basic` delegates the model turn to AgentDojo's native-tool-calling LLM element. Tools run via `runtime.run_function`, mutating the environment for scoring. |

Labels are assigned by **data source**, deterministically and independently of whether an
attack is present. This is the "generic + extensible per-suite" design: the per-suite
registries are authoritative, and the generic heuristic is the fallback.

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

FIDES uses the model's **native** tool-calling API. The default **`--planner full`** drives an
OpenAI-compatible client directly, so it supports `gpt-*`, `openai:<name>` (custom endpoint),
`org/model` (OpenRouter) and `local:`/bare names — but **not** native `anthropic:`/`gemini-*`
(use `--planner labeled_basic` for those, or route them through an OpenAI-compatible gateway
with `openai:<name>` + `OPENAI_BASE_URL`). `--planner labeled_basic` works with every provider;
for a local server without native tool support, set `FIDES_LOCAL_PROMPTED=1`.

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
| `--defense {fides,none}` | `fides` = the FIDES defense; `none` = undefended original model |
| `--planner {full,labeled_basic}` | `full` (default) = variable hiding + quarantined LLM + per-arg policy (OpenAI-compatible endpoints); `labeled_basic` = the tutorial's context-level planner (any provider) |
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

Per-suite and overall utility / security are printed as the run proceeds. Results are also
written **one JSON per task**, DRIFT-style, under a top-level directory that encodes the
defense — `<model>+fides` when the FIDES defense is on, the bare `<model>` for the
undefended model:

```
logs/<model>+fides/<suite>/user_task_<i>/<attack>/injection_task_<j>.json   # under attack
logs/<model>+fides/<suite>/user_task_<i>/none/none.json                     # benign
logs/<model>/<suite>/...                                                    # --defense none
```

Each JSON records `suite_name`, `pipeline_name`, `user_task_id`, `injection_task_id`,
`attack_type`, `injections`, `defense`, `policy`, `fides_events` (the per-tool label +
block/execute decisions), `conversations` (the message trace), `utility`, `security`,
`injection_happened`, and `duration`. Finished tasks are skipped on re-run unless
`--force-rerun` is passed.

`print_results.py` aggregates these JSONs and is paired with `threat_model.py` (the same
trust-separation threat model the sibling baselines use). **Under attack it counts only the
tasks the threat model admits** (`--filter inscope`, the default) — injectable tasks whose
payload is separable from the task's trusted content (`threat_model.qualifying_tasks`) — so
the reported ASR is over the in-scope set, not every AgentDojo pair. `--filter none` counts
every task; utility (no attack) is always over the full set.

```bash
# by model name (mapped to its run dir); --defense picks fides vs the undefended run
python print_results.py --model local:Qwen3.6-35B-A3B --defense fides \
    --suites banking slack travel workspace
# by exact run-dir name, and a different attack
python print_results.py --name Qwen3.6-35B-A3B+fides --attack tool_knowledge
# compare fides vs none (and any other run dirs) side by side
python print_results.py --discover Qwen3.6-35B-A3B
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