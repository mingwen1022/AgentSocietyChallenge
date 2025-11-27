# rec_agent_experiment/RecAgent_baseline_test_train.py
# Offline trainer for long-term memory (MemoryDILU)

import sys
import os
import json
import logging
import re
from typing import List, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv
from websocietysimulator.agent import RecommendationAgent
from websocietysimulator.llm import LLMBase
from websocietysimulator.agent.modules.reasoning_modules import *
from websocietysimulator.agent.modules.info_orchestrator_module import InfoOrchestrator
from websocietysimulator.agent.modules.schemafitter_module import SchemaFitterIO
from websocietysimulator.tools.interaction_tool import InteractionTool
from gemini import GeminiLLM

from websocietysimulator.agent.modules.planning_module_custom import *
from websocietysimulator.agent.modules.memory_modules_custom import *

# ====== Environment and logging ======

load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY in .env file")

logging.basicConfig(level=logging.INFO)


# ====== Data loading: tasks and groundtruth in two aligned lists ======


def load_tasks_and_groundtruth(
    dataset: str = "goodreads",
    base_dir: str = "./example/track2_test",
) -> Tuple[List[dict], List[dict]]:
    """
    Load tasks and groundtruth in the same style as Simulator.set_task_and_groundtruth:
    - tasks: list of task dicts
    - groundtruth_data: list of groundtruth dicts
    The i-th task corresponds to the i-th groundtruth dict.
    """
    tasks_dir = os.path.join(base_dir, dataset, "tasks")
    gt_dir = os.path.join(base_dir, dataset, "groundtruth")

    tasks: List[dict] = []
    groundtruth_data: List[dict] = []

    # sort task files by index: task_0.json, task_1.json, ...
    task_files = sorted(
        [
            f
            for f in os.listdir(tasks_dir)
            if f.startswith("task_") and f.endswith(".json")
        ],
        key=lambda x: int(x.split("_")[1].split(".")[0]),
    )

    for task_file in task_files:
        # derive corresponding groundtruth filename: groundtruth_{index}.json
        task_index = task_file.split("_")[1].split(".")[0]
        groundtruth_file = f"groundtruth_{task_index}.json"
        groundtruth_path = os.path.join(gt_dir, groundtruth_file)

        if not os.path.exists(groundtruth_path):
            logging.warning(
                f"Groundtruth file {groundtruth_file} not found for task {task_file}"
            )
            continue

        # load task file (raw dict; no SimulationTask/RecommendationTask wrapping)
        task_path = os.path.join(tasks_dir, task_file)
        with open(task_path, "r", encoding="utf-8") as f:
            task_data = json.load(f)

        # load groundtruth file (full dict, e.g. {"ground truth": "827260"})
        with open(groundtruth_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)

        tasks.append(task_data)
        groundtruth_data.append(gt_data)

    logging.info(f"Loaded {len(tasks)} task-groundtruth pairs (local train loader)")
    return tasks, groundtruth_data


# ====== Training Agent: planning + memory; reasoning/tooluse mocked ======


class Track2TrainAgent(RecommendationAgent):
    """
    Agent used only for offline training of long-term memory.
    It:
      - uses PlanningIOCustom to produce a plan,
      - simulates reasoning & tooluse,
      - evaluates against groundtruth,
      - writes only successful trajectories into MemoryDILU.
    """

    def __init__(self, llm: LLMBase, dataset: str = "goodreads"):
        super().__init__(llm=llm)
        self.dataset = dataset
        self.planning = PlanningVoyagerCustom(llm=self.llm)
        # Training phase: create a fresh MemoryDILU instance.
        self.memory_dilu = MemoryDILU(llm, reset=False)
        self.memory_gen = MemoryGenerative(llm, reset=False)
        self.memory_tp = MemoryTP(llm, reset=False)
        self.memory_voyager = MemoryVoyager(llm, reset=False)

        # Reasoning module: rank candidate_list
        self.reasoning = ReasoningSelfRefine(
            profile_type_prompt="You are an intelligent recommendation system.",
            memory=None,
            llm=self.llm,
        )

        # Local interaction tool + InfoOrchestrator for profile generation
        # Note: data_dir points to processed data (adjust if needed)
        self.interaction_tool = InteractionTool(data_dir="./data")
        self.info_orchestrator = InfoOrchestrator(
            memory=self.memory_dilu,
            llm=self.llm,
            schema_fitter=None,
            interaction_tool=None,
            use_fixed_item_params=True,
            max_candidates_to_profile=None,
        )
        self._schema_fitter_llm = self.llm

    def workflow(self, task: dict, groundtruth_item_id: str):
        """
        Run once on a single (task, groundtruth_item_id) pair:
        - call planning
        - build profiles via InfoOrchestrator
        - run reasoning to rank candidate_list
        - evaluate whether groundtruth is in the top-3
        - write trajectory into memory ONLY if groundtruth is in top-3
        """
        print("\n===== TRAIN: Planning + Memory (single task) =====\n")

        # 1) Build task description
        task_description = json.dumps(task, indent=2)

        # 2) Build a compact key from user & category (for logging only here)
        user_id = task.get("user_id", "")
        category = task.get("candidate_category", "")
        task_info = f"user={user_id}, category={category}"

        # 3) No few-shot during training (for simplicity and stability)
        few_shot = ""

        # 4) Call planner
        plan = self.planning(
            task_type="Recommendation Task",
            task_description=task_description,
            feedback="",
            few_shot=few_shot,
        )
        print("[Plan]")
        for step in plan:
            print(f"  - Step: {step['description']}")
            print(f"    Reasoning Instruction: {step['reasoning instruction']}")
        print("----")

        # 5) Initialize InfoOrchestrator's schema_fitter and interaction_tool
        if self.info_orchestrator and self.interaction_tool:
            if self.info_orchestrator.schema_fitter is None:
                schema_fitter = SchemaFitterIO(
                    self._schema_fitter_llm, self.interaction_tool
                )
                self.info_orchestrator.schema_fitter = schema_fitter
            self.info_orchestrator.interaction_tool = self.interaction_tool

        # 6) Build user/item profiles via InfoOrchestrator
        profiles = self.info_orchestrator(
            planner_steps=plan,
            user_id=task.get("user_id"),
            candidate_list=task.get("candidate_list"),
        )
        user_profile = profiles.get("user_profile")
        item_profiles = profiles.get("item_profiles", [])

        # 7) Reasoning: rank candidate_list
        reasoning_context = {
            "task": task,
            "user_profile": user_profile,
            "item_profiles": item_profiles,
            "candidate_list": task["candidate_list"],
        }
        reasoning_prompt = (
            "You are a recommendation system. "
            "Given the JSON context below, rank all items in candidate_list from most to least preferred "
            "for the user. Return ONLY a Python list of item_id strings, in descending preference order, "
            "and each id MUST come from candidate_list. Do not include any explanations.\n\n"
            f"CONTEXT:\n{json.dumps(reasoning_context, ensure_ascii=False)}"
        )

        reasoning_output = self.reasoning(reasoning_prompt)
        print("===== RAW REASONING OUTPUT =====")
        print(reasoning_output)

        # Parse ranked candidate list from reasoning_output
        ranked_list = []
        try:
            match = re.search(r"\[.*?\]", reasoning_output, re.DOTALL)
            if match:
                list_str = match.group(0)
                parsed = eval(list_str)
                if isinstance(parsed, list):
                    ranked_list = parsed
        except Exception:
            ranked_list = []

        # Sanitize: keep only candidate_list ids, preserve order, deduplicate
        seen = set()
        ranked_filtered = []
        for cid in ranked_list:
            if (
                isinstance(cid, str)
                and cid in task["candidate_list"]
                and cid not in seen
            ):
                seen.add(cid)
                ranked_filtered.append(cid)
        # append any missing candidates at the end in original order
        for cid in task["candidate_list"]:
            if cid not in seen:
                ranked_filtered.append(cid)

        # 8) Simple top-3 evaluation against groundtruth using ranked_filtered
        rank_pos = None
        if groundtruth_item_id in ranked_filtered:
            rank_pos = ranked_filtered.index(groundtruth_item_id) + 1  # 1-based
        is_top5 = rank_pos is not None and rank_pos <= 5

        print("[Evaluation]")
        print("  Ground truth item:", groundtruth_item_id)
        print("  Ranked list (first 10):", ranked_filtered[:10])
        print("  Ground truth rank position:", rank_pos)
        print("  Correct@3:", is_top5)

        # 9) Build trajectory
        trajectory = (
            f"Task:\n"
            f"    {task_description}\n\n"
            f"Task Info:\n"
            f"    {task_info}\n\n"
            f"Ground truth:\n"
            f"    {groundtruth_item_id}\n\n"
            f"RankPos:\n"
            f"    {rank_pos}\n\n"
            f"IsTop5:\n"
            f"    {is_top5}\n\n"
            f"Plan:\n"
            f"    {plan}\n\n"
            f"UserProfile:\n"
            f"    {json.dumps(user_profile, ensure_ascii=False)}\n\n"
            f"RankedList:\n"
            f"    {ranked_filtered}\n\n"
        )

        # 10) Only store successful trajectories into memory (groundtruth in top-3)
        if is_top5:
            print("\n[Trajectory to Memory]:", trajectory)
            self.memory_dilu("review: " + trajectory)
            self.memory_gen("review: " + trajectory)
            self.memory_tp("review: " + trajectory)
            self.memory_voyager("review: " + trajectory)
            print("[Memory Added]\n")
        else:
            print("\n[Trajectory NOT added to Memory] (incorrect prediction)\n")

        # Return all info for logging
        return {
            "few_shot": few_shot,
            "task": task,
            "plan": plan,
            "ranked_list": ranked_filtered,
            "rank_pos": rank_pos,
            "is_correct": is_top5,
            "memory_trajectory": trajectory,
        }


# ====== Main entry: iterate over all tasks in track2_test and train memory ======

if __name__ == "__main__":
    print(
        "\n===== Track2 Train: multi-task loop over track2_test (build long-term memory) =====\n"
    )

    # 1) Prepare LLM (use Google Gemini)
    llm_google = GeminiLLM(api_key=GOOGLE_API_KEY, model="gemini-2.5-flash")

    # 2) Initialize training Agent (one Agent runs multiple tasks, memory persists across them)
    agent = Track2TrainAgent(llm_google, dataset="goodreads")

    # 3) Load tasks and groundtruth in two aligned lists (same style as Simulator)
    tasks, groundtruth_data = load_tasks_and_groundtruth(
        dataset="goodreads", base_dir="./example/track2_test"
    )

    total = len(tasks)
    correct_cnt = 0
    run_logs = []
    successful = []

    for idx, (task, gt_dict) in enumerate(zip(tasks, groundtruth_data)):
        gt_item = gt_dict.get("ground truth")

        print(f"\n========== Training on Task {idx} ==========")
        print("Task user_id:", task.get("user_id"))
        print("Candidate count:", len(task.get("candidate_list", [])))
        print("Ground truth:", gt_item)

        # workflow returns a dict with all fields
        run_info = agent.workflow(task, gt_item)

        if run_info["is_correct"]:
            correct_cnt += 1
            successful.append(run_info["memory_trajectory"])

        run_logs.append(
            {
                "index": idx,
                "few_shot": run_info["few_shot"],
                "task": run_info["task"],
                "plan": run_info["plan"],
                "ranked_list": run_info["ranked_list"],
                "rank_pos": run_info["rank_pos"],
                "is_correct": run_info["is_correct"],
                "memory_trajectory": run_info["memory_trajectory"],
            }
        )

    summary = {"total": total, "correct_at_5": correct_cnt}
    output = {"summary": summary, "runs": run_logs}

    out_path_memorytxt = (
        "./rec_agent_experiment/memory_train/successful_memory_trajectories.txt"
    )
    with open(out_path_memorytxt, "w", encoding="utf-8") as f:
        for traj in successful:
            f.write(traj + "\n\n" + "=" * 80 + "\n\n")

    out_path_runs = "./rec_agent_experiment/memory_train/memory_train_runs.json"
    with open(out_path_runs, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n===== Training Summary =====")
    print(f"Total tasks: {total}")
    print(f"Correct@5: {correct_cnt}/{total}")
    print(f"Training run log saved to: {out_path_runs}")
    print("Long-term memory stored under ./db/dilu (to be reused in Simulator).")
