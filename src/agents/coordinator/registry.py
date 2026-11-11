"""
Lead 2 Multi-Agent Coordinator Agent Registry & Delegation Protocol
===================================================================
Implements the central registration and capability-based task delegation mechanism:
- CoordinatorAgent (runtime checkable protocol)
- AgentRegistry (discovery, registration, duplicate protection, and delegation)
- Typed exceptions for missing, ambiguous, or failed agent invocations
"""

from typing import Any, Dict, List, Optional, Protocol, Sequence, Set, Tuple, runtime_checkable

from .schemas import CoordinatorStage, TaskContext


class CoordinatorRegistryError(Exception):
    """Base exception for Coordinator Registry operations."""
    pass


class DuplicateAgentRegistrationError(CoordinatorRegistryError):
    """Raised when registering an agent whose ID already exists without override=True."""
    pass


class AgentNotFoundError(CoordinatorRegistryError):
    """Raised when no agent is registered or capable of handling a requested stage or task."""
    pass


class AmbiguousAgentCapabilityError(CoordinatorRegistryError):
    """Raised when multiple registered agents report capability to handle the same task."""
    pass


class IncompatibleAgentCapabilityError(CoordinatorRegistryError):
    """Raised when an agent does not satisfy the CoordinatorAgent protocol or stage contract."""
    pass


class TaskDelegationError(CoordinatorRegistryError):
    """Raised when task execution through an agent adapter fails."""
    pass


@runtime_checkable
class CoordinatorAgent(Protocol):
    """
    Standard protocol satisfied by any agent adapter registering with the Coordinator.
    Provides identification, supported stages, capability verification, input validation,
    and task execution.
    """
    @property
    def agent_id(self) -> str:
        """Unique identifier of the agent adapter (e.g. 'inventory_engine_adapter')."""
        ...

    @property
    def agent_name(self) -> str:
        """Human-readable name of the agent."""
        ...

    @property
    def handled_stages(self) -> Tuple[CoordinatorStage, ...]:
        """Tuple of Coordinator stages this agent can process."""
        ...

    def can_handle(self, task: TaskContext) -> bool:
        """Evaluates whether this agent is capable of processing the specific task."""
        ...

    def validate_task_input(self, task: TaskContext) -> bool:
        """Validates that task.input_payload satisfies the required contracts before execution."""
        ...

    def execute_task(self, task: TaskContext) -> Dict[str, Any]:
        """
        Executes the delegated task and returns a dictionary payload to be recorded
        in task.output_payload.
        """
        ...


class AgentRegistry:
    """
    Thread-safe registry for agent adapters participating in coordinator workflows.
    Provides capability discovery, duplicate registration protection, and task delegation.
    """

    def __init__(self) -> None:
        self._agents: Dict[str, CoordinatorAgent] = {}
        self._stage_map: Dict[CoordinatorStage, List[str]] = {
            stage: [] for stage in CoordinatorStage
        }

    def register(self, agent: CoordinatorAgent, override: bool = False) -> None:
        """
        Registers an agent adapter.
        
        Args:
            agent: Instance conforming to CoordinatorAgent protocol.
            override: If True, replaces an existing registration with the same agent_id.

        Raises:
            IncompatibleAgentCapabilityError: If agent does not satisfy protocol.
            DuplicateAgentRegistrationError: If agent_id already registered and override is False.
        """
        if not isinstance(agent, CoordinatorAgent):
            raise IncompatibleAgentCapabilityError(
                f"Object of type '{type(agent).__name__}' does not satisfy CoordinatorAgent protocol."
            )

        aid = agent.agent_id.strip() if agent.agent_id else ""
        if not aid:
            raise IncompatibleAgentCapabilityError("Agent must have a non-empty 'agent_id'.")

        if aid in self._agents and not override:
            raise DuplicateAgentRegistrationError(
                f"Agent with identifier '{aid}' is already registered. Set override=True to replace."
            )

        # If overriding, clean up existing stage map references
        if aid in self._agents:
            self.unregister(aid)

        stages = agent.handled_stages
        if not stages:
            raise IncompatibleAgentCapabilityError(
                f"Agent '{aid}' must declare at least one handled CoordinatorStage."
            )

        self._agents[aid] = agent
        for stage in stages:
            if stage not in self._stage_map:
                self._stage_map[stage] = []
            if aid not in self._stage_map[stage]:
                self._stage_map[stage].append(aid)

    def unregister(self, agent_id: str) -> None:
        """Unregisters an agent by its identifier."""
        clean_id = agent_id.strip() if agent_id else ""
        if clean_id in self._agents:
            del self._agents[clean_id]
            for stage_list in self._stage_map.values():
                if clean_id in stage_list:
                    stage_list.remove(clean_id)

    def has_agent(self, agent_id: str) -> bool:
        """Returns True if agent_id is registered."""
        return agent_id.strip() in self._agents

    def get_agent(self, agent_id: str) -> CoordinatorAgent:
        """
        Retrieves a registered agent by identifier.
        
        Raises:
            AgentNotFoundError: If agent_id is not registered.
        """
        clean_id = agent_id.strip() if agent_id else ""
        if clean_id not in self._agents:
            raise AgentNotFoundError(f"No agent registered with identifier '{clean_id}'.")
        return self._agents[clean_id]

    def get_agents_for_stage(self, stage: CoordinatorStage) -> Tuple[CoordinatorAgent, ...]:
        """Returns all agents registered to handle a given stage."""
        agent_ids = self._stage_map.get(stage, [])
        return tuple(self._agents[aid] for aid in agent_ids if aid in self._agents)

    def find_agent_for_task(self, task: TaskContext) -> CoordinatorAgent:
        """
        Finds the unique agent capable of executing the specified task.

        Raises:
            AgentNotFoundError: If no registered agent handles this task.
            AmbiguousAgentCapabilityError: If multiple agents claim capability for this task.
        """
        candidate_agents = self.get_agents_for_stage(task.stage)
        if not candidate_agents:
            raise AgentNotFoundError(
                f"No agents registered for stage '{task.stage.value}' to execute task '{task.task_id}'."
            )

        matching = [agent for agent in candidate_agents if agent.can_handle(task)]
        if len(matching) == 0:
            raise AgentNotFoundError(
                f"None of the {len(candidate_agents)} agents registered for stage '{task.stage.value}' "
                f"can handle task '{task.task_id}' (node_key='{task.node_key}')."
            )

        if len(matching) > 1:
            matching_ids = [a.agent_id for a in matching]
            raise AmbiguousAgentCapabilityError(
                f"Ambiguous agent assignment for task '{task.task_id}' (stage '{task.stage.value}'): "
                f"multiple agents claim capability: {matching_ids}."
            )

        return matching[0]

    def delegate_task(self, task: TaskContext) -> Dict[str, Any]:
        """
        Delegates a task to its capable agent:
        1. Resolves agent via find_agent_for_task.
        2. Validates task input via agent.validate_task_input.
        3. Executes task via agent.execute_task.
        4. Returns the output dictionary payload.

        Raises:
            TaskDelegationError: If input validation fails or agent execution raises an error.
        """
        agent = self.find_agent_for_task(task)

        # Validate task input
        try:
            is_valid = agent.validate_task_input(task)
            if not is_valid:
                raise TaskDelegationError(
                    f"Agent '{agent.agent_id}' rejected input payload for task '{task.task_id}'."
                )
        except Exception as exc:
            if isinstance(exc, TaskDelegationError):
                raise
            raise TaskDelegationError(
                f"Validation error in agent '{agent.agent_id}' for task '{task.task_id}': {exc}"
            ) from exc

        # Execute task
        try:
            output = agent.execute_task(task)
            if not isinstance(output, dict):
                raise TaskDelegationError(
                    f"Agent '{agent.agent_id}' returned non-dict output ({type(output).__name__}) "
                    f"for task '{task.task_id}'."
                )
            return output
        except Exception as exc:
            if isinstance(exc, TaskDelegationError):
                raise
            raise TaskDelegationError(
                f"Execution failure in agent '{agent.agent_id}' for task '{task.task_id}': {exc}"
            ) from exc

    def list_agents(self) -> List[Dict[str, Any]]:
        """Returns summary metadata for all registered agents."""
        return [
            {
                "agent_id": agent.agent_id,
                "agent_name": agent.agent_name,
                "handled_stages": [s.value for s in agent.handled_stages],
            }
            for agent in self._agents.values()
        ]

