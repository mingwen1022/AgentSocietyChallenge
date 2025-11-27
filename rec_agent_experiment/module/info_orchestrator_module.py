"""InfoOrchestrator: Orchestrates profile generation for user and item profiles."""

import json
import logging
from typing import Any, Dict, List, Optional

from websocietysimulator.agent.modules.parameter_retriever_user_item import (
    ItemParameterRetriever,
    UserParameterRetriever,
)

logger = logging.getLogger("websocietysimulator")


class InfoOrchestrator:
    USER_PROFILE = "user"
    ITEM_PROFILE = "item"
    
    PROFILE_CONFIG = {
        USER_PROFILE: {
            "detection_keywords": [
                "user", "user's", "user profile", "user behavior", 
                "review history", "preference", "sentiment", "rating"
            ],
            "memory_query_keywords": ["user", "review", "preference", "sentiment"],
            "example_parameters": [
                "genre_preference", "reading_style", "theme_preference", 
                "author_preference", "review_sentiment"
            ]
        },
        ITEM_PROFILE: {
            "detection_keywords": [
                "candidate", "candidates", "metadata", "location", "categories", 
                "book", "books", "item", "items", "features", "attributes", 
                "characteristics", "properties", "recommendation", "options"
            ],
            "memory_query_keywords": ["item", "business", "item", "category", "book"],
            "example_parameters": [
                "genre", "reading_level", "theme", "author_style", "rating"
            ]
        }
    }

    def __init__(self, memory=None, llm=None, schema_fitter=None, interaction_tool=None, 
                 use_fixed_item_params=True, max_candidates_to_profile=None):
        """
        Args:
            memory: Memory module for storing profiles
            llm: Language model for parameter extraction
            schema_fitter: Schema fitter for building profiles
            interaction_tool: Tool for interacting with data
            use_fixed_item_params: If True, use same parameter schema for all items (faster).
                                   If False, extract custom parameters per item (slower but more accurate).
            max_candidates_to_profile: Maximum number of candidates to profile. None = all candidates.
        """
        self.memory = memory
        self.llm = llm
        self.schema_fitter = schema_fitter
        self.interaction_tool = interaction_tool
        self.use_fixed_item_params = use_fixed_item_params
        self.max_candidates_to_profile = max_candidates_to_profile
        self.user_retriever = UserParameterRetriever(llm, interaction_tool)
        self.item_retriever = ItemParameterRetriever(llm, interaction_tool)

    def __call__(
        self, 
        planner_steps: List[Dict[str, Any]], 
        user_id: Optional[str] = None,
        item_id: Optional[str] = None,
        candidate_list: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        # Generate user profile
        user_profile = self._generate_profile(planner_steps, user_id, self.USER_PROFILE, candidate_list) if user_id else {}
        
        # Extract user parameters (excluding user_id)
        user_params = [k for k in user_profile.keys() if k != 'user_id'] if user_profile else None
        
        # Log user profile keys
        if user_profile:
            logger.info(f"✓ User profile keys: {user_params}")
        
        # Generate item profiles for ALL candidates using user-aligned parameters
        item_profiles = []
        if candidate_list and self._requires_profile(planner_steps, self.ITEM_PROFILE):
            item_profiles = self._generate_all_item_profiles(planner_steps, candidate_list, user_params)
        
        if self.memory:
            if user_profile:
                self._store_profile_in_memory(user_profile, self.USER_PROFILE, planner_steps)
            for item_profile in item_profiles:
                self._store_profile_in_memory(item_profile, self.ITEM_PROFILE, planner_steps)
        
        consolidated = {}
        if user_profile:
            consolidated["user_profile"] = user_profile
        if item_profiles:
            consolidated["item_profiles"] = item_profiles
        
        # Log raw JSON structure being returned to reasoning module
        # logger.info("=" * 80)
        # logger.info("PROFILE DATA STRUCTURE RETURNED TO REASONING MODULE:")
        # logger.info("=" * 80)
        # logger.info(json.dumps(consolidated, indent=2, default=str))
        # logger.info("=" * 80)
        
        return consolidated

    def _generate_profile(
        self, 
        planner_steps: List[Dict[str, Any]], 
        entity_id: Optional[str],
        profile_type: str,
        candidate_list: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        if not self._requires_profile(planner_steps, profile_type) or not self.schema_fitter or not entity_id:
            return {}
        
        if self.interaction_tool:
            self.user_retriever.interaction_tool = self.interaction_tool
            self.item_retriever.interaction_tool = self.interaction_tool
        if self.llm:
            self.user_retriever.llm = self.llm
            self.item_retriever.llm = self.llm
        
        if profile_type == self.USER_PROFILE:
            params = self.user_retriever.retrieve_parameters(entity_id, planner_steps)
        elif profile_type == self.ITEM_PROFILE:
            params = self.item_retriever.retrieve_parameters(entity_id, planner_steps, candidate_list)
        else:
            params = UserParameterRetriever.DEFAULT_PARAMETERS if profile_type == self.USER_PROFILE else ItemParameterRetriever.DEFAULT_PARAMETERS
        
        return self._call_schema_fitter(params, entity_id, profile_type)

    def _requires_profile(self, steps: List[Dict[str, Any]], profile_type: str) -> bool:
        keywords = self.PROFILE_CONFIG[profile_type]["detection_keywords"]
        step_text = json.dumps(steps, default=str).lower()
        matches = [kw for kw in keywords if kw in step_text]
        return len(matches) > 0

    def _generate_all_item_profiles(
        self, 
        planner_steps: List[Dict[str, Any]], 
        candidate_list: List[str],
        user_params: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Generate profiles for all candidate items using user-aligned parameters."""
        if not self.schema_fitter or not candidate_list:
            return []
        
        # Determine item parameters based on user preferences
        if user_params:
            item_params = self._convert_user_to_item_params(user_params)
        else:
            item_params = ItemParameterRetriever.DEFAULT_PARAMETERS
        
        # Limit number of candidates if specified
        candidates_to_profile = candidate_list
        if self.max_candidates_to_profile and len(candidate_list) > self.max_candidates_to_profile:
            candidates_to_profile = candidate_list[:self.max_candidates_to_profile]
        
        # Generate profiles in batches for efficiency
        batch_size = 5
        item_profiles = []
        num_batches = (len(candidates_to_profile) + batch_size - 1) // batch_size
        
        for batch_start in range(0, len(candidates_to_profile), batch_size):
            batch_end = min(batch_start + batch_size, len(candidates_to_profile))
            batch_ids = candidates_to_profile[batch_start:batch_end]
            batch_num = batch_start // batch_size + 1
            
            logger.info(f"✓ Batch {batch_num}/{num_batches} (items {batch_start+1}-{batch_end}) using keys: {item_params}")
            
            # Generate profiles for this batch
            for item_id in batch_ids:
                try:
                    profile = self._call_schema_fitter(item_params, item_id, self.ITEM_PROFILE)
                    if profile:
                        item_profiles.append(profile)
                except Exception as e:
                    logger.warning(f"  ✗ Failed item {item_id}: {e}")
        
        return item_profiles
    
    def _convert_user_to_item_params(self, user_params: List[str]) -> List[str]:
        """Use the same parameters for items as the user profile for direct comparison."""
        # Return user params as-is (same keys for both user and items)
        # This allows direct apples-to-apples comparison
        
        # Ensure we have at least 2-4 parameters
        if len(user_params) < 2:
            user_params = list(user_params) + ['genre', 'theme'][:2 - len(user_params)]
        
        return user_params[:4]  # Cap at 4 parameters
    
    def _call_schema_fitter(self, params: List[str], entity_id: Optional[str], profile_type: str) -> Dict[str, Any]:
        if not self.schema_fitter or not entity_id:
            return {}
        
        schema = {param: "string" for param in params}
        schema["user_id" if profile_type == self.USER_PROFILE else "item_id"] = "string"
        
        try:
            result = self.schema_fitter.build_profile(schema=schema, entity_id=entity_id, profile_type=profile_type, max_reviews=50)
            return result if isinstance(result, dict) else {}
        except Exception as e:
            logger.error(f"Error calling schema_fitter: {e}", exc_info=True)
            return {}
    
    def _store_profile_in_memory(self, profile: Dict[str, Any], profile_type: str, planner_steps: List[Dict[str, Any]]):
        if not self.memory or not profile:
            return
        
        param_keys = [k for k in profile.keys() if k not in ['user_id', 'item_id']]
        step_summaries = [s.get("description", "") for s in planner_steps[:3]]
        
        trajectory = f"""Successful {profile_type} profile generation.
Parameters: {', '.join(param_keys)}
Steps: {'; '.join(step_summaries)}
Structure: {json.dumps({k: 'generated' for k in param_keys}, indent=2)}"""
        
        try:
            if hasattr(self.memory, 'addMemory'):
                self.memory.addMemory(trajectory)
            elif hasattr(self.memory, '__call__'):
                self.memory(f"profile:{trajectory}")
        except Exception as e:
            logger.warning(f"  [InfoOrchestrator] Failed to store in memory: {e}")
