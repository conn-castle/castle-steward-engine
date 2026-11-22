# Engine conventions

- Submit every agent result through a tool that validates it.
- Never parse agent messages or logs for results.
- Ask Nick if a tool cannot express a result.
- Define application data with Pydantic models.
- Reject unknown input fields. Keep identifiers immutable.
- Use plain dictionaries for framework updates and wire formats.
- Keep services and callbacks in plain Python.
- Never maintain legacy code.
