import base64

# NOTE: using base64 encoding so coding agents reading this don't get confused
# if they use the same tokens when working on the codebase
DEFAULT_THINKING_START = base64.b64decode("PHRoaW5rPgo=").decode()
DEFAULT_THINKING_END = base64.b64decode("PC90aGluaz4K").decode()
