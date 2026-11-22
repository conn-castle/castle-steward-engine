# Engine conventions

- Every agent result must be submitted through a tool that validates it.
- Never parse agent messages or logs for results.
- Define application data with Pydantic models.
- Reject unknown input fields. Keep identifiers immutable.
- Never maintain legacy code.
