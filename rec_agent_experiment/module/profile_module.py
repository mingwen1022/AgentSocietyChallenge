import logging
from typing import Dict, Any, List
from itertools import islice
from websocietysimulator.tools.interaction_tool import InteractionTool

logger = logging.getLogger("websocietysimulator")


def _take_first_n(lst, n):
    return list(islice(lst, n)) if n is not None else lst


class ProfileBuilder:
    """
    Profile builder that fetches and prepares data for user/book/item profiles.
    This module is called by SchemaFitterIO to get the raw data.
    """
    def __init__(self, interaction_tool: InteractionTool):
        self.tool = interaction_tool

    def build_user_profile_data(
        self,
        user_id: str,
        max_reviews: int | None = 50,
    ) -> Dict[str, Any]:
        """Build profile data for ONE user. Returns raw data dict for schema_fitter."""
        user_record = self.tool.get_user(user_id)
        user_reviews = self.tool.get_reviews(user_id=user_id)
        user_reviews = _take_first_n(user_reviews, max_reviews)
        
        return {
            "user_id": user_id,
            "user": user_record,
            "reviews": user_reviews,
        }

    def build_item_profile_data(
        self,
        item_id: str,
        max_reviews: int | None = 50,
    ) -> Dict[str, Any]:
        """Build profile data for ONE book/item. Returns raw data dict for schema_fitter."""
        item_record = self.tool.get_item(item_id=item_id)
        item_reviews = self.tool.get_reviews(item_id=item_id)
        item_reviews = _take_first_n(item_reviews, max_reviews)
        
        title = item_record.get('title') or item_record.get('name', 'N/A') if item_record else 'N/A'
        
        return {
            "item_id": item_id,
            "item": item_record,
            "reviews": item_reviews,
        }

    def build_user_profiles_data(
        self,
        user_ids: List[str],
        max_reviews: int | None = 50,
    ) -> List[Dict[str, Any]]:
        """Build profile data for MANY users. Returns list of data dicts for batch processing."""
        batch_items = []
        for uid in user_ids:
            user_record = self.tool.get_user(uid)
            user_reviews = self.tool.get_reviews(user_id=uid)
            user_reviews = _take_first_n(user_reviews, max_reviews)
            batch_items.append({
                "user_id": uid,
                "user": user_record,
                "reviews": user_reviews,
            })
        return batch_items

    def build_item_profiles_data(
        self,
        item_ids: List[str],
        max_reviews: int | None = 50,
    ) -> List[Dict[str, Any]]:
        """Build profile data for MANY books/items. Returns list of data dicts for batch processing."""
        batch_items = []
        for item_id in item_ids:
            item_record = self.tool.get_item(item_id=item_id)
            item_reviews = self.tool.get_reviews(item_id=item_id)
            item_reviews = _take_first_n(item_reviews, max_reviews)
            batch_items.append({
                "item_id": item_id,
                "item": item_record,
                "reviews": item_reviews,
            })
        return batch_items

