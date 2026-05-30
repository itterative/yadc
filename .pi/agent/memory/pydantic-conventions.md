---
name: pydantic-conventions
description: Pydantic model conventions — model_validate vs ** unpacking, model_config placement. Read when working with Pydantic models.
category: convention
priority: 1
---

# Pydantic Conventions

## model_validate over ** unpacking

When constructing Pydantic models from dicts (e.g. parsed JSON, TOML), always use `Model.model_validate(data)` instead of `Model(**data)`. This is important because:

- `model_validate` properly handles discriminated unions (e.g. `_OpenAIReasoningDetail` union of text/summary/encrypted types)
- It ensures full validation including nested models and union type resolution
- It is the recommended Pydantic v2 API for deserializing from dicts

Using `**` unpacking bypasses Pydantic's union resolution — it only passes keyword arguments, which can fail or produce incorrect results for union-typed fields.

### When ** is acceptable
- `super().__init__(**kwargs)` for calling parent constructors
- `SomeCaptioner(**kwargs)` for passing through runtime configuration kwargs (not dicts from parsed data)
- `DatasetImage(path=..., caption=..., **extras)` where extras are known keyword arguments, not a raw data dict
- `SafetySettings(data=...)` or `KoboldAdminSettingsReponse(data=...)` where specific keyword args are being set explicitly

## model_config placement

When a Pydantic model uses `model_config`, declare it as a **class variable** annotated with `ClassVar[ConfigDict]` (e.g. `model_config: ClassVar[ConfigDict] = ConfigDict(...)`) and place it **at the end of the class body**, after all field definitions. This keeps fields visible at the top and configuration separate at the bottom.
