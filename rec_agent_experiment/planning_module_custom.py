# planning_recommendation_voyager.py
"""
A custom Planning module for recommendation tasks,
built on top of PlanningVoyager but with additional
field descriptions and task goals.
"""

import re
import ast
from websocietysimulator.agent.modules.planning_modules import PlanningBase


class PlanningRecommendationVoyager(PlanningBase):

    def create_prompt(self, task_type, task_description, feedback, few_shot):
        """
        Enhanced Voyager-style planner:
        - keeps the subgoal-generation style
        - adds lightweight field descriptions
        - adds recommendation task goals
        """
        # relevant datasets need to add instruction for each features
        FIELD_GUIDE = """
You are solving a RECOMMENDATION planning task.

Input task fields (Yelp):
- user_id: appears in user and review
- candidate_list: a list of item_id; each appears in item and review
- candidate_category: high-level category of candidate items
- loc: (latitude, longitude)

Relevant datasets:
- user: user profile ('user_id', 'name', 'review_count', 'yelping_since', 'useful', 'funny','cool', 'elite', 'friends', 'fans', 'average_stars', 'compliment_hot','compliment_more', 'compliment_profile', 'compliment_cute','compliment_list', 'compliment_note', 'compliment_plain','compliment_cool', 'compliment_funny', 'compliment_writer','compliment_photos', 'source')
- item: business metadata ('item_id', 'name', 'address', 'city', 'state', 'postal_code','latitude', 'longitude', 'stars', 'review_count', 'is_open','attributes', 'categories', 'hours', 'source', 'type')
- review: ('review_id', 'user_id', 'item_id', 'stars', 'useful', 'funny', 'cool','text', 'date', 'source', 'type')

Your goal:
1. Generate subgoals that retrieve necessary information from 
   user_yelp, item_yelp, and review_yelp.
2. Ensure subgoals enable a reasoning module to later rank 
   the candidate_list from most relevant to least relevant for the user.
3. Keep subgoals high-level, concise, and logically ordered.
4. For each subgoal, output both "description" and "reasoning instruction".
"""
        # build few-shot block if provided
        FEW_SHOT_BLOCK = ""
        if few_shot not in [None, "", []]:
            FEW_SHOT_BLOCK = f"""
Here are successful examples to guide your planning:
{few_shot}

"""

        if feedback == "":
            prompt = f"""
You are a helpful assistant that generates subgoals to complete any {task_type} task.

{FIELD_GUIDE}

{FEW_SHOT_BLOCK}

You must output a list of subgoals in the following style:
sub-task 1: {{"description": "...", "reasoning instruction": "..."}}
sub-task 2: {{"description": "...", "reasoning instruction": "..."}}

Task: {task_description}
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
"""

        return prompt
