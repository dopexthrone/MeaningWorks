"""Tests for mother/web_tools.py — web search, fetch, and browser control.

Tests are fully mocked. No real HTTP requests or browser launches.
"""

import json
import pytest
from unittest.mock import patch, MagicMock, PropertyMock


# ---------------------------------------------------------------------------
# URL Safety
# ---------------------------------------------------------------------------

class TestURLSafety:
    """Test URL safety validation."""

    def test_safe_https_url(self):
        from mother.web_tools import _is_url_safe
        assert _is_url_safe("https://example.com") is None

    def test_safe_http_url(self):
        from mother.web_tools import _is_url_safe
        assert _is_url_safe("http://example.com") is None

    def test_blocks_localhost(self):
        from mother.web_tools import _is_url_safe
        assert _is_url_safe("http://localhost:8080") is not None

    def test_blocks_127(self):
        from mother.web_tools import _is_url_safe
        assert _is_url_safe("http://127.0.0.1/admin") is not None

    def test_blocks_aws_metadata(self):
        from mother.web_tools import _is_url_safe
        assert _is_url_safe("http://169.254.169.254/latest/meta-data/") is not None

    def test_blocks_file_scheme(self):
        from mother.web_tools import _is_url_safe
        assert _is_url_safe("file:///etc/passwd") is not None

    def test_blocks_javascript_scheme(self):
        from mother.web_tools import _is_url_safe
        assert _is_url_safe("javascript:alert(1)") is not None

    def test_blocks_private_10(self):
        from mother.web_tools import _is_url_safe
        assert _is_url_safe("http://10.0.0.1/") is not None

    def test_blocks_private_192(self):
        from mother.web_tools import _is_url_safe
        assert _is_url_safe("http://192.168.1.1/") is not None

    def test_blocks_private_172(self):
        from mother.web_tools import _is_url_safe
        assert _is_url_safe("http://172.16.0.1/") is not None

    def test_allows_172_outside_range(self):
        from mother.web_tools import _is_url_safe
        assert _is_url_safe("http://172.32.0.1/") is None

    def test_requires_scheme(self):
        from mother.web_tools import _is_url_safe
        assert _is_url_safe("example.com") is not None

    def test_blocks_gcp_metadata(self):
        from mother.web_tools import _is_url_safe
        assert _is_url_safe("http://metadata.google.internal/") is not None


# ---------------------------------------------------------------------------
# HTML to Text
# ---------------------------------------------------------------------------

class TestHTMLToText:
    """Test HTML → readable text conversion."""

    def test_strips_tags(self):
        from mother.web_tools import _html_to_text
        assert "Hello" in _html_to_text("<p>Hello</p>")
        assert "<p>" not in _html_to_text("<p>Hello</p>")

    def test_strips_script(self):
        from mother.web_tools import _html_to_text
        result = _html_to_text("<script>alert('xss')</script><p>Safe</p>")
        assert "alert" not in result
        assert "Safe" in result

    def test_strips_style(self):
        from mother.web_tools import _html_to_text
        result = _html_to_text("<style>.x{color:red}</style><p>Content</p>")
        assert "color" not in result
        assert "Content" in result

    def test_converts_headings(self):
        from mother.web_tools import _html_to_text
        result = _html_to_text("<h1>Title</h1>")
        assert "Title" in result

    def test_converts_list_items(self):
        from mother.web_tools import _html_to_text
        result = _html_to_text("<ul><li>One</li><li>Two</li></ul>")
        assert "One" in result
        assert "Two" in result

    def test_preserves_links(self):
        from mother.web_tools import _html_to_text
        result = _html_to_text('<a href="https://example.com">Click</a>')
        assert "Click" in result
        assert "example.com" in result

    def test_truncates_long_content(self):
        from mother.web_tools import _html_to_text
        long = "<p>" + "x" * 300000 + "</p>"
        result = _html_to_text(long, max_length=1000)
        assert len(result) <= 1100  # max + truncation message

    def test_decodes_entities(self):
        from mother.web_tools import _html_to_text
        result = _html_to_text("<p>A &amp; B &lt; C</p>")
        assert "A & B < C" in result

    def test_empty_input(self):
        from mother.web_tools import _html_to_text
        assert _html_to_text("") == ""
        assert _html_to_text(None) == ""


# ---------------------------------------------------------------------------
# Web Fetch (mocked HTTP)
# ---------------------------------------------------------------------------

class TestWebFetch:
    """Test web fetch with mocked HTTP client."""

    def _mock_response(self, text="<p>Hello</p>", content_type="text/html",
                       status_code=200, url="https://example.com"):
        resp = MagicMock()
        resp.text = text
        resp.status_code = status_code
        resp.headers = {"content-type": content_type}
        resp.raise_for_status = MagicMock()
        if status_code >= 400:
            from httpx import HTTPStatusError
            resp.raise_for_status.side_effect = HTTPStatusError(
                f"HTTP {status_code}", request=MagicMock(), response=resp)
        return resp

    def test_fetch_html_page(self):
        from mother.web_tools import execute_web_fetch
        mock_resp = self._mock_response(text="<title>Test</title><p>Content here</p>")
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = MagicMock(return_value=mock_client.return_value)
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.get.return_value = mock_resp

            result = execute_web_fetch({"url": "https://example.com"})
            assert "Content here" in result
            assert "Test" in result

    def test_fetch_json_response(self):
        from mother.web_tools import execute_web_fetch
        data = {"key": "value", "count": 42}
        mock_resp = self._mock_response(
            text=json.dumps(data),
            content_type="application/json",
        )
        mock_resp.json = MagicMock(return_value=data)
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = MagicMock(return_value=mock_client.return_value)
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.get.return_value = mock_resp

            result = execute_web_fetch({"url": "https://api.example.com/data"})
            assert "key" in result
            assert "value" in result

    def test_fetch_requires_url(self):
        from mother.web_tools import execute_web_fetch
        result = execute_web_fetch({})
        assert "Error" in result

    def test_fetch_blocks_localhost(self):
        from mother.web_tools import execute_web_fetch
        result = execute_web_fetch({"url": "http://localhost:8080"})
        assert "Error" in result
        assert "Blocked" in result

    def test_fetch_auto_https(self):
        from mother.web_tools import execute_web_fetch
        with patch("httpx.Client") as mock_client:
            mock_resp = self._mock_response()
            mock_client.return_value.__enter__ = MagicMock(return_value=mock_client.return_value)
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.get.return_value = mock_resp

            execute_web_fetch({"url": "example.com"})
            call_args = mock_client.return_value.get.call_args
            assert call_args[0][0].startswith("https://")

    def test_fetch_handles_timeout(self):
        from mother.web_tools import execute_web_fetch
        import httpx
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = MagicMock(return_value=mock_client.return_value)
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.get.side_effect = httpx.TimeoutException("timed out")

            result = execute_web_fetch({"url": "https://slow.example.com"})
            assert "Timeout" in result


# ---------------------------------------------------------------------------
# Web Search (mocked HTTP)
# ---------------------------------------------------------------------------

class TestWebSearch:
    """Test web search with mocked HTTP."""

    def _make_ddg_html(self, results):
        """Build fake DuckDuckGo HTML."""
        blocks = []
        for r in results:
            blocks.append(
                f'<a class="result__a" href="{r["url"]}">{r["title"]}</a>'
                f'<a class="result__snippet">{r["snippet"]}</a>'
            )
        return "<html><body>" + "\n".join(blocks) + "</body></html>"

    def test_search_returns_results(self):
        from mother.web_tools import execute_web_search
        fake_html = self._make_ddg_html([
            {"url": "https://example.com", "title": "Example", "snippet": "A test"},
            {"url": "https://other.com", "title": "Other", "snippet": "Another"},
        ])
        mock_resp = MagicMock()
        mock_resp.text = fake_html
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = MagicMock(return_value=mock_client.return_value)
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.get.return_value = mock_resp

            result = execute_web_search({"query": "test query"})
            assert "Example" in result
            assert "example.com" in result

    def test_search_requires_query(self):
        from mother.web_tools import execute_web_search
        result = execute_web_search({})
        assert "Error" in result

    def test_search_handles_no_results(self):
        from mother.web_tools import execute_web_search
        mock_resp = MagicMock()
        mock_resp.text = "<html><body>No results</body></html>"
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = MagicMock(return_value=mock_client.return_value)
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.get.return_value = mock_resp

            result = execute_web_search({"query": "xyznonexistent"})
            assert "No results" in result

    def test_search_caps_results(self):
        from mother.web_tools import execute_web_search, MAX_SEARCH_RESULTS
        results = [
            {"url": f"https://site{i}.com", "title": f"Site {i}", "snippet": f"Result {i}"}
            for i in range(20)
        ]
        fake_html = self._make_ddg_html(results)
        mock_resp = MagicMock()
        mock_resp.text = fake_html
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = MagicMock(return_value=mock_client.return_value)
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.get.return_value = mock_resp

            result = execute_web_search({"query": "test", "num_results": 3})
            # Should cap at requested num_results
            lines = [l for l in result.split("\n") if l.strip().startswith("http")]
            assert len(lines) <= 3


# ---------------------------------------------------------------------------
# Code Engine Registration
# ---------------------------------------------------------------------------

class TestCodeEngineRegistration:
    """Test that web tools are registered in code engine."""

    def test_tools_list_has_web_fetch(self):
        from mother.code_engine import TOOLS
        names = [t.name for t in TOOLS]
        assert "web_fetch" in names

    def test_tools_list_has_web_search(self):
        from mother.code_engine import TOOLS
        names = [t.name for t in TOOLS]
        assert "web_search" in names

    def test_tools_list_has_browser_action(self):
        from mother.code_engine import TOOLS
        names = [t.name for t in TOOLS]
        assert "browser_action" in names

    def test_tool_count_is_10(self):
        from mother.code_engine import TOOLS
        assert len(TOOLS) == 10  # 7 original + 3 new

    def test_executor_dispatch_web_fetch(self):
        from mother.code_engine import _TOOL_EXECUTORS
        assert "web_fetch" in _TOOL_EXECUTORS

    def test_executor_dispatch_web_search(self):
        from mother.code_engine import _TOOL_EXECUTORS
        assert "web_search" in _TOOL_EXECUTORS

    def test_executor_dispatch_browser_action(self):
        from mother.code_engine import _TOOL_EXECUTORS
        assert "browser_action" in _TOOL_EXECUTORS

    def test_all_tools_have_executors(self):
        from mother.code_engine import TOOLS, _TOOL_EXECUTORS
        for tool in TOOLS:
            assert tool.name in _TOOL_EXECUTORS, f"Missing executor for {tool.name}"

    def test_execute_tool_routes_web_fetch(self):
        from mother.code_engine import execute_tool, CodeEngineConfig
        # Should error gracefully for missing URL (not crash)
        config = CodeEngineConfig()
        result = execute_tool("web_fetch", {}, config)
        assert "Error" in result

    def test_execute_tool_routes_web_search(self):
        from mother.code_engine import execute_tool, CodeEngineConfig
        config = CodeEngineConfig()
        result = execute_tool("web_search", {}, config)
        assert "Error" in result


# ---------------------------------------------------------------------------
# Browser Control (mocked playwright)
# ---------------------------------------------------------------------------

class TestBrowserAction:
    """Test browser_action with mocked playwright."""

    def test_unknown_action_returns_error(self):
        from mother.web_tools import execute_browser_action
        result = execute_browser_action({"action": "fly"})
        assert "Error" in result
        assert "Unknown action" in result

    def test_action_required(self):
        from mother.web_tools import execute_browser_action
        result = execute_browser_action({})
        assert "Error" in result

    def test_close_action(self):
        from mother.web_tools import execute_browser_action, _browser_state
        # Should work even when no browser is open
        result = execute_browser_action({"action": "close"})
        assert "closed" in result.lower()

    def test_navigate_requires_url(self):
        from mother.web_tools import execute_browser_action
        # Mock the page creation to avoid real playwright
        mock_page = MagicMock()
        with patch("mother.web_tools._get_page", return_value=(mock_page, None)):
            result = execute_browser_action({"action": "navigate"})
            assert "Error" in result

    def test_navigate_blocks_localhost(self):
        from mother.web_tools import execute_browser_action
        mock_page = MagicMock()
        with patch("mother.web_tools._get_page", return_value=(mock_page, None)):
            result = execute_browser_action({"action": "navigate", "url": "http://localhost"})
            assert "Error" in result
            assert "Blocked" in result

    def test_navigate_success(self):
        from mother.web_tools import execute_browser_action
        mock_page = MagicMock()
        mock_page.title.return_value = "Example"
        with patch("mother.web_tools._get_page", return_value=(mock_page, None)):
            result = execute_browser_action({"action": "navigate", "url": "https://example.com"})
            assert "Navigated" in result
            assert "Example" in result
            mock_page.goto.assert_called_once()

    def test_click_by_text(self):
        from mother.web_tools import execute_browser_action
        mock_page = MagicMock()
        mock_element = MagicMock()
        mock_page.get_by_text.return_value.first = mock_element
        with patch("mother.web_tools._get_page", return_value=(mock_page, None)):
            result = execute_browser_action({"action": "click", "text": "Submit"})
            assert "Clicked" in result
            mock_element.click.assert_called_once()

    def test_click_by_selector(self):
        from mother.web_tools import execute_browser_action
        mock_page = MagicMock()
        with patch("mother.web_tools._get_page", return_value=(mock_page, None)):
            result = execute_browser_action({"action": "click", "selector": "#btn"})
            assert "Clicked" in result
            mock_page.click.assert_called_once()

    def test_type_into_field(self):
        from mother.web_tools import execute_browser_action
        mock_page = MagicMock()
        with patch("mother.web_tools._get_page", return_value=(mock_page, None)):
            result = execute_browser_action({"action": "type", "selector": "#email", "text": "test@example.com"})
            assert "Typed" in result
            mock_page.fill.assert_called_once()

    def test_press_key(self):
        from mother.web_tools import execute_browser_action
        mock_page = MagicMock()
        with patch("mother.web_tools._get_page", return_value=(mock_page, None)):
            result = execute_browser_action({"action": "press", "key": "Enter"})
            assert "Pressed" in result

    def test_extract_content(self):
        from mother.web_tools import execute_browser_action
        mock_page = MagicMock()
        mock_element = MagicMock()
        mock_element.inner_text.return_value = "Extracted text content"
        mock_page.query_selector.return_value = mock_element
        with patch("mother.web_tools._get_page", return_value=(mock_page, None)):
            result = execute_browser_action({"action": "extract", "selector": "main"})
            assert "Extracted text content" in result

    def test_scroll_down(self):
        from mother.web_tools import execute_browser_action
        mock_page = MagicMock()
        with patch("mother.web_tools._get_page", return_value=(mock_page, None)):
            result = execute_browser_action({"action": "scroll", "direction": "down", "amount": 300})
            assert "Scrolled" in result
            mock_page.evaluate.assert_called_once()

    def test_get_url(self):
        from mother.web_tools import execute_browser_action
        mock_page = MagicMock()
        mock_page.url = "https://example.com/page"
        with patch("mother.web_tools._get_page", return_value=(mock_page, None)):
            result = execute_browser_action({"action": "get_url"})
            assert "example.com/page" in result

    def test_evaluate_blocks_dangerous_js(self):
        from mother.web_tools import execute_browser_action
        mock_page = MagicMock()
        with patch("mother.web_tools._get_page", return_value=(mock_page, None)):
            result = execute_browser_action({"action": "evaluate", "expression": "document.cookie"})
            assert "Error" in result
            assert "blocked" in result.lower()

    def test_evaluate_safe_expression(self):
        from mother.web_tools import execute_browser_action
        mock_page = MagicMock()
        mock_page.evaluate.return_value = {"width": 1280}
        with patch("mother.web_tools._get_page", return_value=(mock_page, None)):
            result = execute_browser_action({"action": "evaluate", "expression": "window.innerWidth"})
            assert "1280" in result

    def test_screenshot(self):
        from mother.web_tools import execute_browser_action
        mock_page = MagicMock()
        with patch("mother.web_tools._get_page", return_value=(mock_page, None)):
            result = execute_browser_action({"action": "screenshot"})
            assert "Screenshot" in result
            mock_page.screenshot.assert_called_once()

    def test_wait_for_selector(self):
        from mother.web_tools import execute_browser_action
        mock_page = MagicMock()
        with patch("mother.web_tools._get_page", return_value=(mock_page, None)):
            result = execute_browser_action({"action": "wait", "selector": "#loaded"})
            assert "found" in result.lower()

    def test_page_error_returns_error(self):
        from mother.web_tools import execute_browser_action
        with patch("mother.web_tools._get_page", return_value=(None, "Playwright not installed")):
            result = execute_browser_action({"action": "navigate", "url": "https://example.com"})
            assert "Error" in result
            assert "Playwright" in result


# ---------------------------------------------------------------------------
# DDG Result Parsing
# ---------------------------------------------------------------------------

class TestDDGParsing:
    """Test DuckDuckGo HTML result parsing."""

    def test_parses_standard_results(self):
        from mother.web_tools import _parse_ddg_results
        html = '''
        <a class="result__a" href="https://example.com">Example Site</a>
        <a class="result__snippet">This is a snippet</a>
        <a class="result__a" href="https://other.com">Other Site</a>
        <a class="result__snippet">Another snippet</a>
        '''
        results = _parse_ddg_results(html)
        assert len(results) == 2
        assert results[0]["title"] == "Example Site"
        assert results[0]["url"] == "https://example.com"
        assert results[0]["snippet"] == "This is a snippet"

    def test_fallback_generic_links(self):
        from mother.web_tools import _parse_ddg_results
        html = '''
        <a href="https://example.com">Example Site Title</a>
        <a href="https://other.com">Other Site Title</a>
        <a href="https://duckduckgo.com">DDG (should skip)</a>
        '''
        results = _parse_ddg_results(html)
        assert len(results) >= 2
        urls = [r["url"] for r in results]
        assert "https://example.com" in urls
        assert "https://other.com" in urls
        # DDG links should be filtered
        assert "https://duckduckgo.com" not in urls

    def test_empty_html(self):
        from mother.web_tools import _parse_ddg_results
        results = _parse_ddg_results("")
        assert results == []

    def test_decodes_html_entities(self):
        from mother.web_tools import _parse_ddg_results
        html = '<a class="result__a" href="https://ex.com">A &amp; B</a><a class="result__snippet">C &lt; D</a>'
        results = _parse_ddg_results(html)
        assert results[0]["title"] == "A & B"
        assert results[0]["snippet"] == "C < D"
