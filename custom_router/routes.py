from dataclasses import dataclass
from typing import List
from custom_router.samples import research_sample, chitchat

@dataclass
class Route:
    name: str
    samples: List[str]

research_route = Route(
    name="research",
    samples=research_sample
)

chitchat_route = Route(
    name="chitchat",
    samples=chitchat
)