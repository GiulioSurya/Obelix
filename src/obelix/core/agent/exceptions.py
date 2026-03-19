class TaskRejectedError(Exception):
    """Raised when an agent deliberately refuses to handle a request.

    Maps to A2A TaskState.rejected. The reason is included in
    TaskStatus.message as a TextPart.
    """

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)
