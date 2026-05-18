from importlib import resources


def default_template():
    return load_builtin_template("default")


def load_builtin_template(name: str):
    ref = resources.files("yadc.templates.jinja").joinpath(f"{name}.jinja")
    with resources.as_file(ref) as template_path:
        with open(template_path, "r") as f:
            return f.read()
