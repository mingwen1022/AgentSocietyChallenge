# rec_agent_experiment/rec_agent_experiment_simulator_test.py

import sys
import os
import json
import logging

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv
from websocietysimulator import Simulator
from websocietysimulator.agent import RecommendationAgent
from websocietysimulator.llm import LLMBase
from gemini import GeminiLLM

from planning_module_custom import PlanningIOCustom
from memory_modules_custom import MemoryDILU

logging.basicConfig(level=logging.INFO)


class MyRecommendationAgent(RecommendationAgent):
    """
    Recommendation agent for track2_test using:
    - PlanningIOCustom as planner
    - MemoryDILU as long-term memory
    - Simulated reasoning & tool use
    """

    def __init__(self, llm: LLMBase):
        super().__init__(llm=llm)
        self.planning = PlanningIOCustom(llm=self.llm)
        self.memory = MemoryDILU(llm=self.llm)
        # self.reasoning = ...
        # self.tooluse = ...

    def workflow(self):
        """
        Main workflow for a single recommendation task.
        Simulator will call this once per task.

        Returns:
            list[str]: Ranked list of candidate item IDs.
        """
        # Current task is already converted to dict by RecommendationAgent.insert_task
        task = self.task

        # 1) Build task description
        task_description = json.dumps(task, indent=2)

        # 2) Build a compact key to retrieve relevant few-shot memory
        user_id = task.get("user_id", "")
        category = task.get("candidate_category", "")
        task_info = f"user={user_id}, category={category}"

        # 3) Retrieve few-shot examples from memory (if any)
        few_shot = ""
        if self.memory is not None:
            # Query memory using a concise task key
            few_shot = self.memory("Task: " + task_info) or ""

        # 4) Call planner to generate sub-tasks
        plan = self.planning(
            task_type="Recommendation Task",
            task_description=task_description,
            feedback="",
            few_shot=few_shot,
        )

        # ADD REASONING & TOOL USE HERE (simulated for now)
        # ---------- Reasoning & tool use ----------
        # INPUT: plan (list of sub-tasks)

        # 5) Simulate reasoning & tool use
        tooluse_result = "example_tooluse: simulated tool use (fetch user/item/review)"
        # For testing, simply use candidate_list as the predicted ranking
        reasoning_result = list(task["candidate_list"])

        # ---------- END OF Reasoning & tool use----------
        ##OUTPUT: reasoning_result (list of candidate item IDs), tooluse_result (string)

        return reasoning_result


if __name__ == "__main__":
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("Missing GOOGLE_API_KEY in environment variables")

    # Use a small test split: example/track2_test/goodreads
    task_set = "goodreads"
    task_dir = f"./example/track2/{task_set}/tasks"
    groundtruth_dir = f"./example/track2/{task_set}/groundtruth"

    # Initialize Simulator
    # data_dir should point to where item.json / user.json / review.json are stored
    simulator = Simulator(data_dir="./data", device="auto", cache=False)

    # Load scenarios from track2_test
    simulator.set_task_and_groundtruth(
        task_dir=task_dir,
        groundtruth_dir=groundtruth_dir,
    )

    # Set custom agent
    simulator.set_agent(MyRecommendationAgent)

    # Set LLM client (using Gemini here; you can swap to another LLM if needed)
    llm_google = GeminiLLM(api_key=GOOGLE_API_KEY, model="gemini-2.5-flash")
    simulator.set_llm(llm_google)

    # Run simulation on all tasks in track2_test
    agent_outputs = simulator.run_simulation(
        number_of_tasks=2,  # None means "run all loaded tasks"
        enable_threading=False,  # single-thread for easier debugging
        max_workers=4,
    )

    # Evaluate with groundtruth from track2_test
    evaluation_results = simulator.evaluate()
    out_path = f"./evaluation_results_track2_test_{task_set}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=4, ensure_ascii=False)

    print(f"The evaluation_results is: {evaluation_results}")
    print(f"Saved evaluation results to: {out_path}")
