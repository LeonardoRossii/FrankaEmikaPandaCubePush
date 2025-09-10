def strip_code(code_str):
    lines = code_str.strip().splitlines()
    if lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)

from robosuite.environments import ALL_ENVIRONMENTS
def register_environment(env, name):
    if name not in ALL_ENVIRONMENTS: ALL_ENVIRONMENTS[name] = env