"""``GET /api/info`` — server metadata (platform, version, …)."""

import sys

from . import controller
from .blueprints import ApiBlueprint


@controller
def api_info(app: ApiBlueprint):
    @app.get("/info")
    async def get_info():
        return {"platform": sys.platform}
