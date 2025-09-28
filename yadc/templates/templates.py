from importlib import resources

from . import jinja

def default_template():
    return load_builtin_template('default')

def load_builtin_template(name: str):
    with resources.path(jinja, f'{name}.jinja') as template_path:
        with open(template_path, 'r') as f:
            return f.read()
