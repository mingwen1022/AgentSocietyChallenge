# rec_agent_experiment/rec_agent_experiment_simulator_test.py
import random
import sys
import os
import json
import logging
import re
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv
from websocietysimulator import Simulator
from websocietysimulator.agent import RecommendationAgent
from websocietysimulator.llm import LLMBase
from websocietysimulator.agent.modules.reasoning_modules import *
from websocietysimulator.agent.modules.info_orchestrator_module import InfoOrchestrator
from websocietysimulator.agent.modules.schemafitter_module import SchemaFitterIO
from gemini import GeminiLLM

from websocietysimulator.agent.modules.planning_module_custom import *
from websocietysimulator.agent.modules.memory_modules_custom import *

logging.basicConfig(level=logging.INFO)


class MyRecommendationAgent(RecommendationAgent):
    """
    Recommendation agent for track2_test using:
    - PlanningIOCustom as planner
    - InfoOrchestrator + SchemaFitterIO for user/item profiles
    - MemoryDILU as long-term memory (trajectory storage)
    - Reasoning to rank candidate_list
    """

    def __init__(self, llm: LLMBase):
        super().__init__(llm=llm)
        self.planning = PlanningVoyagerCustom(llm=self.llm)
        self.memory = MemoryDILU(llm=self.llm)
        self.reasoning = ReasoningSelfRefine(
            profile_type_prompt="You are an intelligent recommendation system.",
            memory=None,
            llm=self.llm,
        )
        # InfoOrchestrator; schema_fitter and interaction_tool will be set later
        self.info_orchestrator = InfoOrchestrator(
            memory=None,
            llm=self.llm,
            schema_fitter=None,
            interaction_tool=None,
            use_fixed_item_params=True,
            max_candidates_to_profile=None,
        )
        self._schema_fitter_llm = self.llm

    def set_interaction_tool(self, interaction_tool):
        """
        Override to also wire interaction_tool into InfoOrchestrator.
        """
        super().set_interaction_tool(interaction_tool)
        if self.info_orchestrator and interaction_tool:
            if self.info_orchestrator.schema_fitter is None:
                schema_fitter = SchemaFitterIO(
                    self._schema_fitter_llm, interaction_tool
                )
                self.info_orchestrator.schema_fitter = schema_fitter
            self.info_orchestrator.interaction_tool = interaction_tool

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

        # 3) Retrieve few-shot examples from memory (if any)
        task_query = json.dumps(task, ensure_ascii=False)
        few_shot = self.memory(task_query) or ""

        # 4) Call planner to generate sub-tasks
        plan = self.planning(
            task_type="Recommendation Task",
            task_description=task_description,
            feedback="",
            few_shot="",
        )

        # 5) Build user/item profiles via InfoOrchestrator
        profiles = self.info_orchestrator(
            planner_steps=plan,
            user_id=task.get("user_id"),
            candidate_list=task.get("candidate_list"),
        )
        user_profile = profiles.get("user_profile")
        item_profiles = profiles.get("item_profiles", [])

        # 6) Reasoning: rank candidate_list
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
        logging.info("RAW REASONING OUTPUT:\n%s", reasoning_output)

        # Parse ranked candidate list from reasoning output
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

        logging.info("FINAL RANKED LIST (len=%d)", len(ranked_filtered))

        # # 7) Store trajectory in memory (no groundtruth here; simulator will evaluate separately)
        # trajectory = (
        #     f"Task:\n"
        #     f"    {task_description}\n\n"
        #     f"Plan:\n"
        #     f"    {plan}\n\n"
        #     f"UserProfile:\n"
        #     f"    {json.dumps(user_profile, ensure_ascii=False)}\n\n"
        #     f"RankedList:\n"
        #     f"    {ranked_filtered}\n\n"
        # )
        # self.memory("review: " + trajectory)

        return ranked_filtered


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
    # 随机选 k 个任务（保持和 groundtruth 对齐）
    all_indices = list(range(len(simulator.tasks)))
    selected = random.sample(all_indices, k=20)
    simulator.tasks = [simulator.tasks[i] for i in selected]
    simulator.groundtruth_data = [simulator.groundtruth_data[i] for i in selected]

    # Set custom agent
    simulator.set_agent(MyRecommendationAgent)

    # Set LLM client (using Gemini here; you can swap to another LLM if needed)
    llm_google = GeminiLLM(api_key=GOOGLE_API_KEY, model="gemini-2.5-flash")
    simulator.set_llm(llm_google)

    # Run simulation on all tasks in track2_test
    start = time.time()
    agent_outputs = simulator.run_simulation(
        number_of_tasks=10,  # None means "run all loaded tasks"
        enable_threading=True,  # single-thread for easier debugging
        max_workers=4,
    )

    # Evaluate with groundtruth from track2_test
    evaluation_results = simulator.evaluate()
    end = time.time()
    out_path = f"./evaluation_result/evaluation_results_{task_set}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=4, ensure_ascii=False)

    print(f"The evaluation_results is: {evaluation_results}")
    print(f"Saved evaluation results to: {out_path}")
    print(f"Simulation + evaluation for 10 tasks took {end - start:.1f} seconds")
