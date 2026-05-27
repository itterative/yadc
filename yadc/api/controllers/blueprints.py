from injector import singleton
from quart import Blueprint


@singleton
class ApiBlueprint(Blueprint):
    pass


@singleton
class AppBlueprint(Blueprint):
    pass
