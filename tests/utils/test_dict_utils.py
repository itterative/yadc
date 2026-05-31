"""Tests for toml_merge — tomlkit-aware deep merge preserving comments."""

import tomlkit

from yadc.utils.dict_utils import deep_merge, toml_merge, toml_to_plain

# ---------------------------------------------------------------------------
# toml_merge: basic merge behaviour
# ---------------------------------------------------------------------------


class TestTomlMergeBasic:
    def test_scalar_override(self):
        doc = tomlkit.loads("[api]\nurl = 'http://old'\nmodel_name = 'gemma3'")
        result = toml_merge(doc, {"api": {"url": "http://new"}})
        assert result["api"]["url"] == "http://new"
        assert result["api"]["model_name"] == "gemma3"

    def test_nested_dict_merge(self):
        doc = tomlkit.loads("[api]\nurl = 'http://old'\nmodel_name = 'gemma3'\n[settings]\nmax_tokens = 512")
        result = toml_merge(doc, {"settings": {"max_tokens": 1024, "store_conversation": True}})
        assert result["settings"]["max_tokens"] == 1024
        assert result["settings"]["store_conversation"] is True
        # Unchanged keys preserved
        assert result["api"]["url"] == "http://old"

    def test_list_replaced_atomically(self):
        doc = tomlkit.loads("[[dataset]]\npath = '/old'\n[dataset.extras]\nstyle = 'watercolor'")
        result = toml_merge(doc, {"dataset": [{"path": "/new"}]})
        assert len(result["dataset"]) == 1
        assert result["dataset"][0]["path"] == "/new"

    def test_new_top_level_key(self):
        doc = tomlkit.loads("[api]\nurl = 'http://localhost'")
        result = toml_merge(doc, {"env": "production"})
        assert result["env"] == "production"
        assert result["api"]["url"] == "http://localhost"

    def test_new_nested_key(self):
        doc = tomlkit.loads("[api]\nurl = 'http://localhost'")
        result = toml_merge(doc, {"api": {"model_name": "gemma3"}})
        assert result["api"]["model_name"] == "gemma3"
        assert result["api"]["url"] == "http://localhost"

    def test_empty_override(self):
        doc = tomlkit.loads("[api]\nurl = 'http://localhost'")
        result = toml_merge(doc, {})
        assert result["api"]["url"] == "http://localhost"

    def test_does_not_mutate_base(self):
        doc = tomlkit.loads("[api]\nurl = 'http://old'")
        toml_merge(doc, {"api": {"url": "http://new"}})
        assert doc["api"]["url"] == "http://old"


# ---------------------------------------------------------------------------
# toml_merge: comment and formatting preservation
# ---------------------------------------------------------------------------


class TestTomlMergeComments:
    def test_preserves_top_level_comment(self):
        doc = tomlkit.loads("# Top comment\n[api]\nurl = 'http://localhost'")
        result = toml_merge(doc, {"api": {"url": "http://new"}})
        output = tomlkit.dumps(result)
        assert "# Top comment" in output

    def test_preserves_inline_comment_on_changed_key(self):
        doc = tomlkit.loads("[api]\nurl = 'http://localhost'  # inline")
        result = toml_merge(doc, {"api": {"url": "http://new"}})
        output = tomlkit.dumps(result)
        assert "# inline" in output

    def test_preserves_inline_comment_on_unchanged_key(self):
        doc = tomlkit.loads("[api]\nurl = 'http://localhost'  # keep\nmodel_name = 'gemma3'")
        result = toml_merge(doc, {"api": {"model_name": "llama"}})
        output = tomlkit.dumps(result)
        assert "# keep" in output

    def test_preserves_section_comment(self):
        doc = tomlkit.loads("# Section comment\n[settings]\nmax_tokens = 512")
        result = toml_merge(doc, {"settings": {"max_tokens": 1024}})
        output = tomlkit.dumps(result)
        assert "# Section comment" in output

    def test_preserves_blank_lines(self):
        src = "[api]\nurl = 'http://localhost'\n\n\n[settings]\nmax_tokens = 512"
        doc = tomlkit.loads(src)
        result = toml_merge(doc, {"settings": {"max_tokens": 1024}})
        output = tomlkit.dumps(result)
        # Blank lines between sections should survive
        assert "\n\n" in output


# ---------------------------------------------------------------------------
# toml_merge: type handling (Pydantic compatibility)
# ---------------------------------------------------------------------------


class TestTomlMergeTypes:
    def test_boolean_values_are_python_bools(self):
        """tomlkit wraps assigned values, but bools should be assignable."""
        doc = tomlkit.loads("[settings]\nstore_conversation = false")
        result = toml_merge(doc, {"settings": {"store_conversation": False}})
        val = result["settings"]["store_conversation"]
        assert val is False

    def test_boolean_override_true(self):
        doc = tomlkit.loads("[settings]\nstore_conversation = false")
        result = toml_merge(doc, {"settings": {"store_conversation": True}})
        val = result["settings"]["store_conversation"]
        assert val is True

    def test_integer_override(self):
        doc = tomlkit.loads("[settings]\nmax_tokens = 512")
        result = toml_merge(doc, {"settings": {"max_tokens": 1024}})
        assert result["settings"]["max_tokens"] == 1024

    def test_string_override(self):
        doc = tomlkit.loads("[api]\nurl = 'http://old'")
        result = toml_merge(doc, {"api": {"url": "http://new"}})
        assert result["api"]["url"] == "http://new"

    def test_pydantic_validation_on_merged_result(self):
        """Regression test: merged tomlkit doc must pass Pydantic validation."""
        from yadc.core.config import parse_config

        doc = tomlkit.loads("[api]\nurl = 'http://localhost:11434'\nmodel_name = 'gemma3'\n[settings]\nmax_tokens = 512\nimage_quality = 'auto'")
        override = {
            "api": {"url": "", "model_name": ""},
            "prompt": {"name": ""},
            "settings": {"store_conversation": False, "max_tokens": 10240},
            "overwrite_captions": False,
            "reasoning": {"enable": False, "thinking_effort": "low", "exclude_from_output": True},
            "env": "",
            "dataset": [{"path": "/test", "extras": {"artist": "test"}}],
        }
        merged = toml_merge(doc, override)
        config = parse_config(toml_to_plain(merged), strict=False)
        assert config.settings.max_tokens == 10240
        assert config.settings.store_conversation is False
        assert config.reasoning.enable is False
        assert config.reasoning.exclude_from_output is True


# ---------------------------------------------------------------------------
# deep_merge (original, non-tomlkit) still works
# ---------------------------------------------------------------------------


class TestDeepMergeUnchanged:
    def test_original_deep_merge_works(self):
        base = {"api": {"url": "http://old", "model_name": "gemma3"}}
        result = deep_merge(base, {"api": {"url": "http://new"}})
        assert result == {"api": {"url": "http://new", "model_name": "gemma3"}}


# ---------------------------------------------------------------------------
# toml_to_plain: converts tomlkit wrappers to plain Python types
# ---------------------------------------------------------------------------


class TestTomlToPlain:
    def test_converts_nested_tables(self):
        doc = tomlkit.loads("[api]\nurl = 'http://localhost'")
        plain = toml_to_plain(doc)
        assert type(plain) is dict
        assert type(plain["api"]) is dict
        assert type(plain["api"]["url"]) is str

    def test_converts_booleans(self):

        doc = tomlkit.loads("[settings]\nstore_conversation = true")
        plain = toml_to_plain(doc)
        val = plain["settings"]["store_conversation"]
        assert type(val) is bool
        assert val is True

    def test_converts_integers(self):

        doc = tomlkit.loads("[settings]\nmax_tokens = 1024")
        plain = toml_to_plain(doc)
        val = plain["settings"]["max_tokens"]
        assert type(val) is int
        assert val == 1024

    def test_converts_arrays_of_tables(self):

        doc = tomlkit.loads('[[dataset]]\npath = "/data"\n[dataset.extras]\nstyle = "oil"')
        plain = toml_to_plain(doc)
        assert type(plain["dataset"]) is list
        assert type(plain["dataset"][0]) is dict
        assert plain["dataset"][0]["path"] == "/data"
        assert type(plain["dataset"][0]["extras"]) is dict

    def test_pydantic_accepts_converted_result(self):
        from yadc.core.config import parse_config

        doc = tomlkit.loads(
            "[api]\nurl = 'http://localhost:11434'\nmodel_name = 'gemma3'\n"
            "[settings]\nmax_tokens = 512\nstore_conversation = true\nimage_quality = 'auto'\n"
            "[reasoning]\nenable = true\nthinking_effort = 'high'\nexclude_from_output = false"
        )
        plain = toml_to_plain(doc)
        config = parse_config(plain, strict=False)
        assert config.settings.store_conversation is True
        assert config.reasoning.enable is True

    def test_pydantic_accepts_merged_and_converted_result(self):
        """End-to-end: merge + convert + validate."""
        from yadc.core.config import parse_config

        doc = tomlkit.loads("[api]\nurl = 'http://localhost:11434'\nmodel_name = 'gemma3'\n[settings]\nmax_tokens = 512\nimage_quality = 'auto'")
        override = {
            "api": {"url": "", "model_name": ""},
            "prompt": {"name": ""},
            "settings": {"store_conversation": False, "max_tokens": 10240},
            "overwrite_captions": False,
            "reasoning": {"enable": False, "thinking_effort": "low", "exclude_from_output": True},
            "env": "",
            "dataset": [{"path": "/test", "extras": {"artist": "test"}}],
        }
        merged = toml_merge(doc, override)
        plain = toml_to_plain(merged)
        config = parse_config(plain, strict=False)
        assert config.settings.max_tokens == 10240
        assert config.settings.store_conversation is False
        assert config.reasoning.enable is False
