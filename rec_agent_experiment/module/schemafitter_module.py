import re
import ast
import json
import logging
from typing import Dict, Any, List, Optional
from websocietysimulator.agent.modules.profile_module import ProfileBuilder
from websocietysimulator.tools.interaction_tool import InteractionTool

logger = logging.getLogger("websocietysimulator")

class SchemaFitterBase:
    def __init__(self, llm):
        """Base class for schema-fitting summarizers."""
        self.llm = llm
        self.output = None

    def create_prompt(self, schema, items, profile_type="user"):
        """Subclasses will define how the system prompt is constructed."""
        raise NotImplementedError

    def __call__(self, schema, items):
        """Execute the summarization / schema-fitting call."""
        prompt = self.create_prompt(schema, items)
        messages = [{"role": "user", "content": prompt}]
        raw_output = self.llm(messages=messages, temperature=0.1)

        # Extract JSON-like dicts from output
        dict_strings = re.findall(r"\{[^{}]*\}", raw_output)
        dicts = []
        for ds in dict_strings:
            try:
                dicts.append(ast.literal_eval(ds))
            except Exception:
                pass

        self.output = dicts
        return dicts
    
class SchemaFitterIO(SchemaFitterBase):
    """
    Schema fitter that orchestrates profile building:
    1. Calls ProfileBuilder to fetch and prepare data
    2. Fits the data to the provided schema using LLM
    """
    def __init__(self, llm, interaction_tool: Optional[InteractionTool] = None):
        super().__init__(llm)
        self.profile_builder = ProfileBuilder(interaction_tool) if interaction_tool else None

    def build_profile(
        self,
        schema: Dict[str, Any],
        entity_id: str,
        profile_type: str = "user",
        max_reviews: int | None = 50,
    ) -> Dict[str, Any]:
        """Build a profile by calling profile module, then fitting to schema."""
        if not self.profile_builder:
            raise ValueError("interaction_tool required for build_profile")
        
        # Fetch data from profile module
        if profile_type == "user":
            items = self.profile_builder.build_user_profile_data(entity_id, max_reviews)
        else:
            items = self.profile_builder.build_item_profile_data(entity_id, max_reviews)
        
        # Fit data to schema using LLM
        profiles = self(schema=schema, items=items, profile_type=profile_type)
        result = profiles[0] if profiles else {}
        return result

    def create_prompt(self, schema, items, profile_type="user"):
        """Create prompt for schema fitting."""
        if profile_type == "user":
            entity_name = "user"
            entity_id_field = "user_id"
            entity_data_field = "user"
        else:
            entity_name = "book/item"
            entity_id_field = "item_id"
            entity_data_field = "item"

        schema_str = json.dumps(schema, indent=2) if isinstance(schema, dict) else str(schema)
        items_str = json.dumps(items, indent=2, default=str) if isinstance(items, (list, dict)) else str(items)
        
        prompt = f"""Extract profile information from Goodreads book data. Analyze "{entity_data_field}" metadata and ALL reviews to fill the schema fields.

SCHEMA:
{schema_str}

DATA:
{items_str}

INSTRUCTIONS:
- Read ALL reviews carefully (most valuable information)
- Extract meaningful values for each schema field
- For books: focus on genre, reading_level, theme, author_style
- For users: focus on genre_preference, reading_style, theme_preference, author_preference
- DO NOT return null unless information truly cannot be found
- Include "{entity_id_field}" in output
- Output ONLY a valid JSON array, no markdown, no explanations

Example: [{{"{entity_id_field}": "id123", "genre": "fantasy", "reading_level": "young_adult"}}]

Output JSON array:"""
        return prompt

    def _extract_json_array(self, raw_output):
        """Extract JSON array from code block or direct match."""
        code_block = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', raw_output, re.DOTALL)
        if code_block:
            return code_block.group(1)
        array_match = re.search(r'\[[^\[\]]*(?:\{[^}]*\}[^\[\]]*)*\]', raw_output, re.DOTALL)
        return array_match.group(0) if array_match else None

    def __call__(self, schema, items, profile_type="user"):
        """Execute the summarization / schema-fitting call."""
        # Normalize inputs
        if isinstance(schema, list):
            schema = {param: "string" for param in schema}
        if isinstance(items, dict):
            items = [items]
        
        # Get LLM response
        prompt = self.create_prompt(schema, items, profile_type)
        raw_output = self.llm(messages=[{"role": "user", "content": prompt}], temperature=0.1)
        
        # Extract JSON array
        json_array_str = self._extract_json_array(raw_output)
        if json_array_str:
            try:
                dicts = json.loads(json_array_str)
                if isinstance(dicts, list) and dicts:
                    self.output = dicts
                    return dicts
            except json.JSONDecodeError as e:
                logger.error(f"  [SchemaFitter] JSON decode error: {e}")
        
        logger.error(f"  [SchemaFitter] FAILED to extract valid JSON array from LLM response!")
        self.output = []
        return []
