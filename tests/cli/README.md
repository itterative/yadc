## cli tests

To run these tests you need to setup the environment for each API client.

By default, these tests are marked to be skipped if the environment is not set up. This is done intentionally, because they will incur costs due to making real API calls.

Environments:
* integration-tests-local-koboldcpp
* integration-tests-gemini
* integration-tests-openrouter
* integration-tests-openai

Required settings:
* api_url
* api_token
* api_model_name
