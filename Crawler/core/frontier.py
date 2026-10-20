"""BFS URL Frontier.

REVIEW MAP
----------
SECTION 1: FrontierItem = URL + depth
SECTION 2: append/popleft => FIFO => BFS
SECTION 3: queued/visited duplicate control
"""
from collections import deque
from dataclasses import dataclass


# ============================================================
# SECTION 1 - FRONTIER ITEM
# ============================================================
@dataclass(frozen=True)
class FrontierItem:
    url: str
    depth: int


class URLFrontier:
    def __init__(self):
        self.queue = deque()
        self.queued: set[str] = set()
        self.visited: set[str] = set()

    # ========================================================
    # SECTION 2 - FIFO OPERATIONS (BFS)
    # ========================================================
    def add(self, url: str, depth: int) -> bool:
        if url in self.visited or url in self.queued:
            return False
        self.queue.append(FrontierItem(url, depth))
        self.queued.add(url)
        return True

    def pop(self) -> FrontierItem | None:
        if not self.queue:
            return None
        item = self.queue.popleft()
        self.queued.discard(item.url)
        return item

    # ========================================================
    # SECTION 3 - DUPLICATE URL STATE
    # ========================================================
    def mark_visited(self, url: str) -> None:
        self.visited.add(url)
        self.queued.discard(url)

    def restore_visited(self, urls) -> None:
        self.visited.update(urls)

    def empty(self) -> bool:
        return not self.queue

    def __len__(self) -> int:
        return len(self.queue)
