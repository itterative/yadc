import abc

import requests

from yadc.core import logging
from yadc.core import Captioner

from .session import Session

_logger = logging.get_logger(__name__)

class BaseAPICaptioner(Captioner, abc.ABC):
    def __init__(self, **kwargs):
        """
        Initializes the BaseAPICaptioner 

        Args:
            api_url (str): Base URL for the API endpoint.
            api_token (str): API key for authentication.

            **kwargs: Optional keyword arguments:
                - `prompt_template_name` (str): Filename of the Jinja2 template to use (default: 'default.jinja').
                - `prompt_template` (str): Direct template string to override file-based templates.
                - `session` (requests.Session, options): Override the session for the API calls

        Raises:
            ValueError: If `api_url` is not provided.
        """

        super().__init__(**kwargs)

        warnings: bool = kwargs.get('_warnings', True)

        self._api_url: str = kwargs.get('api_url', '')
        self._api_token: str = kwargs.get('api_token', '')

        if not self._api_url:
            raise ValueError("no api_url")

        session_headers = {}

        if self._api_token:
            session_headers['Authorization'] = f'Bearer {self._api_token}'
        elif warnings:
            _logger.warning('Warning: no api_token is set, requests will fail if api uses authentication')

        session: requests.Session|None = kwargs.get('session', None)
        assert session is None or isinstance(session, requests.Session)

        self._session = Session(self._api_url, headers=session_headers, session=session)

    @abc.abstractmethod
    def log_usage(self):
        raise NotImplemented
