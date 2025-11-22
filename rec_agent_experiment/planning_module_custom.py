# planning_recommendation_voyager.py
"""
A custom Planning module for recommendation tasks,
built on top of PlanningVoyager but with additional
field descriptions and task goals.
"""

import re
import ast
from websocietysimulator.agent.modules.planning_modules import PlanningBase

FIELD_GUIDE = """
You are solving a RECOMMENDATION planning task.

Input task fields (Yelp):
-type: is's fixed as "recommendation"
- user_id: appears in user and review
- candidate_list: a list of item_id; each appears in item and review
- candidate_category: high-level category of candidate items
- loc: (latitude, longitude)

Relevant datasets (Yelp):

user dataset — one row per user:
- user_id: unique user identifier; links to review.user_id
- name: user nickname
- review_count: total number of reviews this user has written
- yelping_since: when the user joined Yelp
- useful / funny / cool: votes received for their reviews
- elite: elite years string or None
- friends: list of friend user_ids or None
- fans: number of fans
- average_stars: user's average rating across reviews
- compliment_hot / more / profile / cute / list / note / plain / cool /
  funny / writer / photos: compliment counters indicating engagement level
- source: fixed as 'yelp'

item dataset — one row per business:
- item_id: unique POI identifier; links to review.item_id
- name: business name
- address: street address
- city / state / postal_code: business location metadata
- latitude / longitude: coordinates for distance matching
- stars: average rating for business
- review_count: number of reviews for business
- is_open: 1 = open, 0 = closed
- attributes: key-value dict of business properties
    (e.g., Alcohol, HasTV, Ambience, NoiseLevel, WiFi, etc.)
- categories: comma-separated high-level tags (e.g., "Restaurants, Barbeque")
- hours: daily opening times dictionary
- source: fixed as 'yelp'
- type: object type, usually "business"

review dataset — one row per review:
- review_id: unique review identifier
- user_id: reviewer (matches user.user_id)
- item_id: reviewed business (matches item.item_id)
- stars: star rating (1-5)
- useful / funny / cool: votes on the review
- text: free-text content of user's review
- date: timestamp of review
- source: fixed as 'yelp'
- type: object type, usually "business review"

Your goal:
1. Generate subgoals that retrieve necessary information from 
   user_yelp, item_yelp, and review_yelp.
2. Ensure subgoals enable a reasoning module to later rank 
   the candidate_list from most relevant to least relevant for the user.
3. Keep subgoals high-level, concise, and logically ordered.
4. For each subgoal, output both "description" and "reasoning instruction".
"""

OUTPUT_STYLE_GUIDE = """
You must output a list of subgoals in the following style:
sub-task 1: {"description": "...", "reasoning instruction": "..."}
sub-task 2: {"description": "...", "reasoning instruction": "..."}
"""


def build_few_shot_block(few_shot):
    if few_shot in [None, "", []]:
        return ""
    return f"""
Here are successful examples to guide your planning:
{few_shot}

"""


class PlanningVoyagerCustom(PlanningBase):

    def create_prompt(self, task_type, task_description, feedback, few_shot):
        """
        Enhanced Voyager-style planner:
        - keeps the subgoal-generation style
        - adds lightweight field descriptions
        - adds recommendation task goals
        """
        # relevant datasets need to add instruction for each features

        # build few-shot block if provided
        FEW_SHOT_BLOCK = build_few_shot_block(few_shot)

        if feedback == "":
            prompt = f"""
You are a helpful assistant that generates subgoals to complete any {task_type} task.

{FIELD_GUIDE}

{FEW_SHOT_BLOCK}

Task: {task_description}

{OUTPUT_STYLE_GUIDE}
"""
        else:
            prompt = f"""
You are a helpful assistant that generates subgoals to complete any {task_type} task.

{FIELD_GUIDE}

{FEW_SHOT_BLOCK}

end
--------------------
reflexion:{feedback}

Task: {task_description}

{OUTPUT_STYLE_GUIDE}
"""

        return prompt


class PlanningIOCustom(PlanningBase):
    def create_prompt(self, task_type, task_description, feedback, few_shot):
        few_shot_block = build_few_shot_block(few_shot)

        if feedback == "":
            prompt = f"""
You are a planner who divides a {task_type} task into several subtasks. You also need to give the reasoning instructions for each subtask and the instructions for calling the tool. Your output format should follow the example below.

{FIELD_GUIDE}
{few_shot_block}

Task: {task_description}
{OUTPUT_STYLE_GUIDE}
"""
        else:
            prompt = f"""
You are a planner who divides a {task_type} task into several subtasks. You also need to give the reasoning instructions for each subtask and the instructions for calling the tool. Your output format should follow the example below.

{FIELD_GUIDE}
{few_shot_block}
end
--------------------
Reflexion:{feedback}
Task:{task_description}
{OUTPUT_STYLE_GUIDE}
"""
        return prompt


class PlanningDEPSCustom(PlanningBase):
    def create_prompt(self, task_type, task_description, feedback, few_shot):
        few_shot_block = build_few_shot_block(few_shot)

        if feedback == "":
            prompt = f"""
You are a helper AI agent in reasoning. You need to generate the sequences of sub-goals (actions) for a {task_type} task in multi-hop questions. You also need to give the reasoning instructions for each subtask and the instructions for calling the tool. Your output format should follow the example below.

{FIELD_GUIDE}
{few_shot_block}
Task: {task_description}
{OUTPUT_STYLE_GUIDE}
"""
        else:
            prompt = f"""
You are a helper AI agent in reasoning. You need to generate the sequences of sub-goals (actions) for a {task_type} task in multi-hop questions. You also need to give the reasoning instructions for each subtask and the instructions for calling the tool. Your output format should follow the example below.

{FIELD_GUIDE}
{few_shot_block}
end
--------------------
Reflexion:{feedback}
Task:{task_description}
{OUTPUT_STYLE_GUIDE}
"""
        return prompt


class PlanningTDCustom(PlanningBase):
    def create_prompt(self, task_type, task_description, feedback, few_shot):
        few_shot_block = build_few_shot_block(few_shot)

        if feedback == "":
            prompt = f"""
You are a planner who divides a {task_type} task into several subtasks with explicit temporal dependencies.
Consider the order of actions and their dependencies to ensure logical sequencing.
Your output format must follow the example below, specifying the order and dependencies.

{FIELD_GUIDE}
{few_shot_block}
Task: {task_description}
{OUTPUT_STYLE_GUIDE}
"""
        else:
            prompt = f"""
You are a planner who divides a {task_type} task into several subtasks with explicit temporal dependencies.
Consider the order of actions and their dependencies to ensure logical sequencing.
Your output format should follow the example below, specifying the order and dependencies.

{FIELD_GUIDE}
{few_shot_block}
end
--------------------
Reflexion:{feedback}
Task:{task_description}
{OUTPUT_STYLE_GUIDE}
"""
        return prompt


class PlanningVoyagerCustom(PlanningBase):
    def create_prompt(self, task_type, task_description, feedback, few_shot):
        few_shot_block = build_few_shot_block(few_shot)

        if feedback == "":
            prompt = f"""
You are a helpful assistant that generates subgoals to complete any {task_type} task specified by me.
I'll give you a final task, you need to decompose the task into a list of subgoals.
You must follow the following criteria:
1) Return a list of subgoals that can be completed in order to complete the specified task.
2) Give the reasoning instructions for each subgoal and the instructions for calling the tool. 
You also need to give the reasoning instructions for each subtask and the instructions for calling the tool. Your output format should follow the example below.

{FIELD_GUIDE}
{few_shot_block}
Task: {task_description}
{OUTPUT_STYLE_GUIDE}
"""
        else:
            prompt = f"""
You are a helpful assistant that generates subgoals to complete any {task_type} task specified by me.
I'll give you a final task, you need to decompose the task into a list of subgoals.
You must follow the following criteria:
1) Return a list of subgoals that can be completed in order to complete the specified task.
2) Give the reasoning instructions for each subgoal and the instructions for calling the tool. 
You also need to give the reasoning instructions for each subtask and the instructions for calling the tool. Your output format should follow the example below.

{FIELD_GUIDE}
{few_shot_block}
end
--------------------
reflexion:{feedback}
task:{task_description}
{OUTPUT_STYLE_GUIDE}
"""
        return prompt


class PlanningOPENAGICustom(PlanningBase):
    def create_prompt(self, task_type, task_description, feedback, few_shot):
        few_shot_block = build_few_shot_block(few_shot)

        if feedback == "":
            prompt = f"""
You are a planner who is an expert at coming up with a todo list for a given {task_type} objective.
For each task, you also need to give the reasoning instructions for each subtask and the instructions for calling the tool.
Ensure the list is as short as possible, and tasks in it are relevant, effective and described in a single sentence.
Develop a concise to-do list to achieve the objective.  
Your output format should follow the example below.

{FIELD_GUIDE}
{few_shot_block}
Task: {task_description}
{OUTPUT_STYLE_GUIDE}
"""
        else:
            prompt = f"""
You are a planner who is an expert at coming up with a todo list for a given {task_type} objective.
For each task, you also need to give the reasoning instructions for each subtask and the instructions for calling the tool.
Ensure the list is as short as possible, and tasks in it are relevant, effective and described in a single sentence.
Develop a concise to-do list to achieve the objective.
Your output format should follow the example below.

{FIELD_GUIDE}
{few_shot_block}
end
--------------------
Reflexion:{feedback}
Task:{task_description}
{OUTPUT_STYLE_GUIDE}
"""
        return prompt


class PlanningHUGGINGGPTCustom(PlanningBase):
    def create_prompt(self, task_type, task_description, feedback, few_shot):
        few_shot_block = build_few_shot_block(few_shot)

        if feedback == "":
            prompt = f"""
You are a planner who divides a {task_type} task into several subtasks. Think step by step about all the tasks needed to resolve the user's request. Parse out as few tasks as possible while ensuring that the user request can be resolved. Pay attention to the dependencies and order among tasks. you also need to give the reasoning instructions for each subtask and the instructions for calling the tool. Your output format should follow the example below.

{FIELD_GUIDE}
{few_shot_block}
Task: {task_description}
{OUTPUT_STYLE_GUIDE}
"""
        else:
            prompt = f"""
You are a planner who divides a {task_type} task into several subtasks. Think step by step about all the tasks needed to resolve the user's request. Parse out as few tasks as possible while ensuring that the user request can be resolved. Pay attention to the dependencies and order among tasks. you also need to give the reasoning instructions for each subtask and the instructions for calling the tool. Your output format should follow the example below.

{FIELD_GUIDE}
{few_shot_block}
end
--------------------
Reflexion:{feedback}
Task:{task_description}
{OUTPUT_STYLE_GUIDE}
"""
        return prompt
