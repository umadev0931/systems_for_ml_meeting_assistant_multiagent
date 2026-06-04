"""Tests for bench/vllm_metrics.py — Prometheus text parser."""
import pytest
from bench.vllm_metrics import parse_prometheus, VllmMetricsScraper, MetricSnapshot


SAMPLE_PROMETHEUS = """\
# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc 0.42
# HELP vllm:num_requests_running Running requests
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running 3
vllm:num_requests_waiting 0
# HELP vllm:prompt_tokens_total Prompt tokens
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total 12345.0
vllm:generation_tokens_total 678.0
vllm:prefix_cache_queries_total 100.0
vllm:prefix_cache_hits_total 57.0
"""


class TestParsePrometheus:
    def test_parses_gauge(self):
        raw = parse_prometheus(SAMPLE_PROMETHEUS)
        assert raw["vllm:gpu_cache_usage_perc"] == pytest.approx(0.42)

    def test_parses_counter(self):
        raw = parse_prometheus(SAMPLE_PROMETHEUS)
        assert raw["vllm:prompt_tokens_total"] == pytest.approx(12345.0)

    def test_running_requests(self):
        raw = parse_prometheus(SAMPLE_PROMETHEUS)
        assert raw["vllm:num_requests_running"] == pytest.approx(3.0)

    def test_comments_and_blanks_ignored(self):
        raw = parse_prometheus("# HELP foo bar\n\nvllm:foo{label=\"x\"} 1.5\n")
        assert raw["vllm:foo"] == pytest.approx(1.5)

    def test_empty_string(self):
        assert parse_prometheus("") == {}

    def test_labels_are_summed(self):
        """Multiple lines with different labels for the same metric are summed."""
        text = "vllm:req{model=\"a\"} 10\nvllm:req{model=\"b\"} 5\n"
        raw = parse_prometheus(text)
        assert raw["vllm:req"] == pytest.approx(15.0)

    def test_invalid_lines_skipped(self):
        text = "not_a_metric\nvllm:good 1.0\n"
        raw = parse_prometheus(text)
        assert "vllm:good" in raw
        assert len(raw) == 1


class TestVllmMetricsScraperCounterDelta:
    def _snap(self, counters: dict) -> MetricSnapshot:
        s = MetricSnapshot(available=True)
        s.counters = counters
        return s

    def test_delta_positive(self):
        before = self._snap({"prefix_cache_hits_total": 10.0,
                             "prefix_cache_queries_total": 20.0})
        after  = self._snap({"prefix_cache_hits_total": 17.0,
                             "prefix_cache_queries_total": 30.0})
        delta = VllmMetricsScraper.counter_delta(before, after)
        assert delta["prefix_cache_hits_total"] == pytest.approx(7.0)
        assert delta["prefix_cache_queries_total"] == pytest.approx(10.0)

    def test_delta_hit_rate_computed(self):
        before = self._snap({"prefix_cache_hits_total": 0.0,
                             "prefix_cache_queries_total": 0.0})
        after  = self._snap({"prefix_cache_hits_total": 60.0,
                             "prefix_cache_queries_total": 100.0})
        delta = VllmMetricsScraper.counter_delta(before, after)
        assert delta["server_prefix_cache_hit_rate"] == pytest.approx(0.60)

    def test_delta_zero_queries_no_division_error(self):
        before = self._snap({"prefix_cache_hits_total": 0.0,
                             "prefix_cache_queries_total": 0.0})
        after  = self._snap({"prefix_cache_hits_total": 0.0,
                             "prefix_cache_queries_total": 0.0})
        delta = VllmMetricsScraper.counter_delta(before, after)
        assert delta.get("server_prefix_cache_hit_rate") in (None, 0.0)
