from flask import Blueprint
from injector import singleton


@singleton
class ApiBlueprint(Blueprint):
    pass


@singleton
class AppBlueprint(Blueprint):
    pass
