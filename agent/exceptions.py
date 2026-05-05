class AgentWorkflowError(Exception):
    """Base exception for agent workflow errors."""

    status_code = 500
    error_code = "agent_workflow_error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class InvalidInputError(AgentWorkflowError):
    status_code = 422
    error_code = "invalid_input"


class WorkflowExecutionError(AgentWorkflowError):
    status_code = 500
    error_code = "workflow_execution_error"


class MissingRAGIndexError(AgentWorkflowError):
    status_code = 503
    error_code = "missing_rag_index"
