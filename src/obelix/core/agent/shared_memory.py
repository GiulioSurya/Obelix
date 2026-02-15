"""Directed dependency graph with shared memory between agents."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import networkx as nx

from obelix.infrastructure.logging import get_logger

logger = get_logger(__name__)


class PropagationPolicy(str, Enum):
    FINAL_RESPONSE_ONLY = "final_response_only"
    LAST_TOOL_RESULT = "last_tool_result"


@dataclass
class MemoryItem:
    source_id: str
    content: str
    timestamp: datetime
    policy: PropagationPolicy


@dataclass
class NodeData:
    last_final: str | None = None
    last_tool_result: str | None = None
    timestamp: datetime | None = None
    metadata: dict = field(default_factory=dict)


class SharedMemoryGraph:
    """Directed dependency graph with shared memory between agents.

    Thread-safety:
    - publish() uses asyncio.Lock (concurrent writes)
    - pull_for() is read-only (no lock needed)
    - add_agent()/add_edge() are called during setup (no concurrency)
    """

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()
        self._lock: asyncio.Lock = asyncio.Lock()

    def add_agent(self, node_id: str) -> None:
        if not self._graph.has_node(node_id):
            self._graph.add_node(node_id, data=NodeData())
            logger.debug(f"SharedMemoryGraph: node '{node_id}' added")

    def add_edge(
        self,
        src: str,
        dst: str,
        policy: PropagationPolicy = PropagationPolicy.FINAL_RESPONSE_ONLY,
    ) -> None:
        self.add_agent(src)
        self.add_agent(dst)
        if nx.has_path(self._graph, dst, src):
            raise ValueError(f"Adding edge '{src}' -> '{dst}' would create a cycle")
        self._graph.add_edge(src, dst, policy=policy)
        logger.debug(f"SharedMemoryGraph: edge '{src}' -> '{dst}' added (policy={policy.value})")

    def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    async def publish(
        self,
        node_id: str,
        content: str,
        kind: str = "final",
        metadata: dict | None = None,
    ) -> None:
        async with self._lock:
            self.add_agent(node_id)
            node_data: NodeData = self._graph.nodes[node_id]["data"]
            if kind == "tool_result":
                node_data.last_tool_result = content
            else:
                node_data.last_final = content
            node_data.timestamp = datetime.now()
            if metadata:
                node_data.metadata.update(metadata)
            logger.info(f"SharedMemoryGraph: '{node_id}' published {kind} ({len(content)} chars)")

    def pull_for(self, node_id: str) -> list[MemoryItem]:
        if not self._graph.has_node(node_id):
            return []
        items: list[MemoryItem] = []
        for pred in self._graph.predecessors(node_id):
            node_data: NodeData = self._graph.nodes[pred]["data"]
            edge_data = self._graph.edges[pred, node_id]
            policy = edge_data.get("policy", PropagationPolicy.FINAL_RESPONSE_ONLY)

            if policy == PropagationPolicy.LAST_TOOL_RESULT:
                content = node_data.last_tool_result
            else:
                content = node_data.last_final

            if content is None:
                continue

            items.append(MemoryItem(
                source_id=pred,
                content=content,
                timestamp=node_data.timestamp,
                policy=policy,
            ))
        return items

    def pull_for_indirect(self, node_id: str) -> list[MemoryItem]:
        if not self._graph.has_node(node_id):
            return []
        items: list[MemoryItem] = []
        reversed_graph = self._graph.reverse()
        for _, ancestor in nx.bfs_edges(reversed_graph, node_id):
            node_data: NodeData = self._graph.nodes[ancestor]["data"]
            if node_data.last_final is None:
                continue
            items.append(MemoryItem(
                source_id=ancestor,
                content=node_data.last_final,
                timestamp=node_data.timestamp,
                policy=PropagationPolicy.FINAL_RESPONSE_ONLY,
            ))
        return items

    def get_edges_for_nodes(self, node_ids: list[str]) -> list[tuple[str, str]]:
        node_set = set(node_ids)
        return [(u, v) for u, v in self._graph.edges() if u in node_set and v in node_set]

    def get_topological_order(self, node_ids: list[str] | None = None) -> list[str]:
        if node_ids:
            subgraph = self._graph.subgraph(node_ids)
            return list(nx.topological_sort(subgraph))
        return list(nx.topological_sort(self._graph))

    def clear_published_data(self) -> None:
        for node_id in self._graph.nodes():
            self._graph.nodes[node_id]["data"] = NodeData()
        logger.info("SharedMemoryGraph: all published data cleared")