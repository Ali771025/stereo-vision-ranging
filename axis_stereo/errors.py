"""Dependency-light error labels; never print private camera URLs."""
def redacted_error(exc):
    response = getattr(exc, 'response', None)
    status = getattr(response, 'status_code', None)
    if status is not None:
        return f'HTTP {status}'
    return type(exc).__name__
