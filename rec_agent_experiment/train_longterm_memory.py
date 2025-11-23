# rec_agent_experiment/RecAgent_baseline_test_train.py
# Offline trainer for long-term memory (MemoryDILU)

import sys
import os
import json
import logging
from typing import List, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv
from websocietysimulator.agent import RecommendationAgent
from websocietysimulator.llm import LLMBase
from gemini import GeminiLLM

from planning_module_custom import PlanningIOCustom
from memory_modules_custom import *

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
        self.planning = PlanningIOCustom(llm=self.llm)
        # Training phase: create a fresh MemoryDILU instance.
        self.memory_dilu = MemoryDILU(llm, reset=True)
        self.memory_gen = MemoryGenerative(llm, reset=True)
        self.memory_tp = MemoryTP(llm, reset=True)
        self.memory_voyager = MemoryVoyager(llm, reset=True)

    def workflow(self, task: dict, groundtruth_item_id: str):
        """
        Run once on a single (task, groundtruth_item_id) pair:
        - call planning
        - simulate reasoning & tooluse
        - evaluate whether groundtruth is in the top-3
        - write trajectory into memory ONLY if correct
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

        # 5) Simulate reasoning & tooluse (only ensure structure is valid here)
        tooluse_result = "example_tooluse: simulated tool use (fetch user/item/review)"
        # Here we directly use candidate_list as the "predicted ranking result"
        reasoning_result = list(task["candidate_list"])

        # 6) Simple top-3 evaluation against groundtruth
        top3 = reasoning_result[:3]
        is_correct = groundtruth_item_id in top3

        print("[Evaluation]")
        print("  Ground truth item:", groundtruth_item_id)
        print("  Pred top-3:", top3)
        print("  Correct@3:", is_correct)

        # 7) Build trajectory
        trajectory = (
            f"Task:\n"
            f"    {task_description}\n\n"
            f"Task Info:\n"
            f"    {task_info}\n\n"
            f"Ground truth:\n"
            f"    {groundtruth_item_id}\n\n"
            f"Is Correct:\n"
            f"    {is_correct}\n\n"
            f"Plan:\n"
            f"    {plan}\n\n"
            f"Reasoning:\n"
            f"    {reasoning_result}\n\n"
            f"ToolUse:\n"
            f"    {tooluse_result}\n"
        )

        # 8) Only store successful trajectories into memory
        if is_correct:
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
            "tooluse": tooluse_result,
            "reasoning_result": reasoning_result,
            "is_correct": is_correct,
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

        run_logs.append(
            {
                "index": idx,
                "few_shot": run_info["few_shot"],
                "task": run_info["task"],
                "plan": run_info["plan"],
                "tooluse": run_info["tooluse"],
                "reasoning_result": run_info["reasoning_result"],
                "is_correct": run_info["is_correct"],
                "memory_trajectory": run_info["memory_trajectory"],
            }
        )

    summary = {"total": total, "correct_at_3": correct_cnt}
    output = {"summary": summary, "runs": run_logs}

    out_path = "memory_train_runs.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n===== Training Summary =====")
    print(f"Total tasks: {total}")
    print(f"Correct@3: {correct_cnt}/{total}")
    print(f"Training run log saved to: {out_path}")
    print("Long-term memory stored under ./db/dilu (to be reused in Simulator).")
