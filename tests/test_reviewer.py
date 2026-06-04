"""Tests for agents/reviewer.py — parse_verdict robustness."""
import json

from agents.reviewer import parse_verdict


# ------------------------------------------------------------------ helpers
def _verdict(status="pass", summarizer="", extractor="", drafter=""):
    return json.dumps({
        "status": status,
        "issues": {"summarizer": summarizer,
                   "extractor": extractor,
                   "drafter": drafter},
    })


# ------------------------------------------------------------------ happy path
class TestParseVerdictPass:
    def test_clean_pass(self):
        v = parse_verdict(_verdict("pass"))
        assert v.status == "pass"
        assert not v.needs_revision

    def test_revise_all(self):
        raw = _verdict("revise", summarizer="Too short",
                       extractor="Missing owner", drafter="Wrong tone")
        v = parse_verdict(raw)
        assert v.status == "revise"
        assert v.needs_revision
        assert set(v.workers_to_revise()) == {"summarizer", "extractor", "drafter"}

    def test_revise_one_worker(self):
        raw = _verdict("revise", summarizer="Hallucination found")
        v = parse_verdict(raw)
        assert v.needs_revision
        assert v.workers_to_revise() == ["summarizer"]

    def test_revise_empty_issues_no_revision(self):
        """status=revise but all issue strings empty → no actual revision."""
        raw = _verdict("revise", summarizer="", extractor="", drafter="")
        v = parse_verdict(raw)
        assert v.status == "revise"
        assert not v.needs_revision
        assert v.workers_to_revise() == []

    def test_status_case_insensitive(self):
        raw = json.dumps({"status": "PASS", "issues": {}})
        v = parse_verdict(raw)
        assert v.status == "pass"


# ------------------------------------------------------------------ malformed input
class TestParseVerdictMalformed:
    def test_empty_string_defaults_pass(self):
        v = parse_verdict("")
        assert v.status == "pass"
        assert not v.needs_revision

    def test_none_defaults_pass(self):
        v = parse_verdict(None)  # type: ignore[arg-type]
        assert v.status == "pass"

    def test_invalid_json_defaults_pass(self):
        v = parse_verdict("{not valid json")
        assert v.status == "pass"

    def test_json_with_no_status_defaults_pass(self):
        v = parse_verdict(json.dumps({"issues": {"summarizer": "bad"}}))
        assert v.status == "pass"

    def test_prose_wrapping_json_still_parsed(self):
        """Model sometimes wraps JSON in prose — regex should still find it."""
        raw = ('Here is my verdict:\n'
               + _verdict("revise", extractor="Missing deadline field")
               + '\nEnd of verdict.')
        v = parse_verdict(raw)
        assert v.status == "revise"
        assert "extractor" in v.workers_to_revise()

    def test_non_dict_issues_defaults_empty(self):
        raw = json.dumps({"status": "revise", "issues": ["bad", "list"]})
        v = parse_verdict(raw)
        assert isinstance(v.issues, dict)

    def test_extra_fields_ignored(self):
        raw = json.dumps({"status": "pass", "issues": {}, "extra": "ignored"})
        v = parse_verdict(raw)
        assert v.status == "pass"
