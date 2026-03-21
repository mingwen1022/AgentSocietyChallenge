# Enhanced Goodreads Recommendation Multi-Agent System

![License](https://img.shields.io/badge/license-MIT-green)
[![Competition](https://img.shields.io/badge/WWW'25-AgentSociety_Challenge-blue)](https://github.com/tsinghua-fib-lab/AgentSocietyChallenge)
[![Track](https://img.shields.io/badge/Track_2-Recommendation-orange)](https://tsinghua-fib-lab.github.io/AgentSocietyChallenge/pages/recommendation-track.html)
[![arXiv](https://img.shields.io/badge/arXiv-2502.18754-b31b1b.svg)](https://arxiv.org/abs/2502.18754)

**Team:** Shu Han Ho · Alexander Thaik · Ming Wen · Yu-Yun Chen

---

## Overview

We built a multi-agent LLM recommendation system for the **WWW'25 AgentSociety Challenge (Track 2 — Recommendation)**, targeting the Goodreads dataset. Given a user and 20 candidate books, our system returns a ranked recommendation list.

We extend the [AgentSociety Challenge framework](https://github.com/tsinghua-fib-lab/AgentSocietyChallenge) with four core innovations — **Dynamic Profile Generation**, **Permutation Evaluation Pipeline**, **Pairwise Reranking**, and **Long-Term Memory** — achieving significant improvements over the baseline across all Hit Rate metrics.

---

## Results

| Configuration | HR@1 | HR@3 | HR@5 |
|---|:---:|:---:|:---:|
| Baseline | 0.00 | 0.24 | 0.42 |
| **Ours (best)** | **0.26** | **0.54** | **0.60** |

**Best configuration:** `PlanningVoyagerCustom` + `ReasoningIO` + `MemoryDILU` + Pairwise Reranking (MaxRev=70, FieldLen=50)

---

## System Architecture

The pipeline follows four stages:

1. **Planning** — LLM generates dataset-aware sub-task steps (Voyager/DEPS/IO planner variants)
2. **Profile Generation** — `InfoOrchestrator` + `SchemaFitterIO` build structured user and item JSON profiles from raw review data
3. **Memory Retrieval** — `MemoryDILU` fetches similar successful trajectories from an offline-trained vector store as few-shot examples
4. **Reasoning + Reranking** — LLM ranks candidates; `PairwiseRanker` applies a King-of-the-Hill tournament to optimize HR@1

---

## Four Core Innovations

### 1. Dynamic Profile Generation

Instead of feeding raw text dumps to the LLM, the `InfoOrchestrator` module dynamically determines what attributes matter for each user (genre preference, reading style, theme, etc.) and builds a structured JSON schema via `SchemaFitterIO`. Candidate profiles are generated using the same user-specific schema — ensuring the LLM focuses on signal relevant to that individual user rather than generic metadata.

**Key files:** `rec_agent_experiment/info_orchestrator_module.py`, `rec_agent_experiment/schemafitter_module.py`

### 2. Permutation Evaluation Pipeline

A systematic testing framework that cycles through 15+ workflow combinations (6 planning modules × 6 reasoning modules × 4 memory modules) on the same task set for fair comparison. Outputs hit rates, timing, and value-efficiency scores per configuration.

**Key files:** `Ai_AGENT_SH_branch/example/enhanced_agent/workflow_mixins.py`, `rec_agent_experiment/test_recommendation_accuracy.py`

### 3. Pairwise Reranking

After initial pointwise ranking, the top-K=5 candidates are refined using a **King of the Hill** linear scan — each challenger is compared head-to-head against the current king with a "strict judge" Chain-of-Thought prompt. This reduces position bias and hallucinations at O(K) cost (4 LLM calls per task), directly optimizing HR@1.

**Key files:** `pairwise_module_callingexample/pairwise_modules.py`

### 4. Long-Term Memory

An offline training script (`train_longterm_memory.py`) runs the full pipeline and stores successful trajectories (where the ground-truth item lands in top-5) into a `MemoryDILU` vector database (Chroma). At inference, the planner retrieves similar past trajectories as few-shot examples, grounding the LLM and reducing hallucinations.

**Key files:** `rec_agent_experiment/train_longterm_memory.py`, `rec_agent_experiment/memory_modules_custom.py`

---

## Directory Structure

```
.
├── websocietysimulator/          # Core simulation library (base framework)
│   └── agents/modules/           # Planning, reasoning, memory base classes + custom variants
├── rec_agent_experiment/         # Our custom modules and training scripts
│   ├── info_orchestrator_module.py
│   ├── schemafitter_module.py
│   ├── memory_modules_custom.py
│   ├── train_longterm_memory.py
│   └── memory_train/             # Stored successful trajectories
├── Ai_AGENT_SH_branch/           # Enhanced modular agent framework
│   └── example/enhanced_agent/   # WorkflowMixin, EnhancedRecommendationAgent, base_agent.py
├── pairwise_module_callingexample/  # Pairwise reranking implementation
├── example/                      # Baseline agents for comparison
├── evaluation_result/            # JSON results from all experiment runs
├── GTsimulation/                 # Game-theory based agent variants
├── data_images/                  # Visualizations and workflow diagrams
├── docs/                         # Competition documentation site
├── tutorials/                    # Setup and usage guides
├── data_process.py               # Dataset preparation script
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
git clone https://github.com/mingwen1022/AgentSocietyChallenge.git
cd AgentSocietyChallenge

# Recommended: Poetry
poetry install && poetry shell

# Alternative: pip
pip install -r requirements.txt && pip install .
```

### 2. Prepare data

Download the [Goodreads dataset](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home) and process it:

```bash
python data_process.py --input <path_to_raw_dataset> --output <path_to_processed_dataset>
```

Organize into:
```
<dataset_dir>/
├── item.json
├── review.json
└── user.json
```

### 3. Configure API key

Create a `.env` file and add your OpenAI or DeepSeek API key.

### 4. Run the recommendation agent

```python
from websocietysimulator import Simulator
from websocietysimulator.llm import DeepseekLLM

simulator = Simulator(data_dir="path/to/dataset", device="auto", cache=True)
simulator.set_task_and_groundtruth(task_dir="path/to/tasks", groundtruth_dir="path/to/groundtruth")
simulator.set_agent(YourCustomAgent)
simulator.set_llm(DeepseekLLM(api_key="YOUR_API_KEY"))

agent_outputs = simulator.run_simulation(number_of_tasks=50, enable_threading=True, max_workers=10)
evaluation_results = simulator.evaluate()
print(evaluation_results)
```

---

## Credits

This project is built on top of the **WWW'25 AgentSociety Challenge** framework by Tsinghua FIB Lab:

> **AgentSociety Challenge: Designing LLM Agents for User Behavior Simulation and Recommendation**
> [https://github.com/tsinghua-fib-lab/AgentSocietyChallenge](https://github.com/tsinghua-fib-lab/AgentSocietyChallenge)
> arXiv: [2502.18754](https://arxiv.org/abs/2502.18754)

---

## License

MIT License. See [LICENSE](./LICENSE) for details.
