from yadc.cmd import app

TEMPLATE_PATH = app.STATE_PATH / 'templates'

def load_user_template(name: str):
    template = TEMPLATE_PATH / f'{name}.jinja'

    with open(template) as f:
        return f.read()

def save_user_template(name: str, content: str):
    TEMPLATE_PATH.mkdir(mode=0o750, exist_ok=True)

    template = TEMPLATE_PATH / f'{name}.jinja'

    with open(template, 'w') as f:
        f.write(content)

def list_user_template():
    return [
        template.name.removesuffix('.jinja') for template in TEMPLATE_PATH.glob('*.jinja')
    ]

def delete_user_template(name: str):
    template = TEMPLATE_PATH / f'{name}.jinja'

    if not template.exists():
        return False

    template.unlink()
    return True
