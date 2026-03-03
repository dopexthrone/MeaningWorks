"""
Web tools for Mother's code engine — web search, fetch, and browser control.

LEAF module. Uses httpx (already in deps) for HTTP. Playwright for browser
control (lazy-installed at first use). No imports from core/ or mother/.

These tools extend the code engine's 7-tool surface to enable web interaction:
- web_search: search the web via DuckDuckGo HTML (no API key)
- web_fetch: fetch a URL, convert HTML to readable text
- browser_action: full browser control (navigate, click, type, screenshot, extract)
"""

import html
import json
import logging
import os
import re
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus, urljoin, urlparse

logger = logging.getLogger("mother.web_tools")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Max response sizes
MAX_FETCH_BYTES = 200_000   # 200KB max fetched content
MAX_SEARCH_RESULTS = 10     # max search results to return
FETCH_TIMEOUT = 30          # seconds
BROWSER_TIMEOUT = 30        # seconds per browser action

# URL safety: block local/private network access
_BLOCKED_HOSTS = frozenset({
    "localhost", "127.0.0.1", "0.0.0.0", "::1",
    "169.254.169.254",  # AWS metadata
    "metadata.google.internal",  # GCP metadata
})

_BLOCKED_SCHEMES = frozenset({"file", "ftp", "data", "javascript"})


def _is_url_safe(url: str) -> Optional[str]:
    """Return error if URL is unsafe, else None."""
    try:
        parsed = urlparse(url)
    except Exception as e:
        return f"Invalid URL: {e}"

    if not parsed.scheme:
        return "URL must include scheme (https://...)"

    if parsed.scheme.lower() in _BLOCKED_SCHEMES:
        return f"Blocked scheme: {parsed.scheme}"

    host = (parsed.hostname or "").lower()
    if host in _BLOCKED_HOSTS:
        return f"Blocked host: {host}"

    # Block private IP ranges
    if host.startswith("10.") or host.startswith("192.168."):
        return f"Blocked private IP: {host}"
    if host.startswith("172."):
        parts = host.split(".")
        if len(parts) >= 2:
            try:
                second = int(parts[1])
                if 16 <= second <= 31:
                    return f"Blocked private IP: {host}"
            except ValueError:
                pass

    return None


# ---------------------------------------------------------------------------
# HTML → readable text conversion (no external deps)
# ---------------------------------------------------------------------------

def _html_to_text(raw_html: str, max_length: int = MAX_FETCH_BYTES) -> str:
    """Convert HTML to readable text. Lightweight, no BeautifulSoup needed."""
    if not raw_html:
        return ""

    text = raw_html

    # Remove script/style blocks
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<noscript[^>]*>.*?</noscript>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # Convert common elements to text markers
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p>", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</div>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</li>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<li[^>]*>", "  - ", text, flags=re.IGNORECASE)
    text = re.sub(r"<h([1-6])[^>]*>(.*?)</h\1>", r"\n## \2\n", text, flags=re.DOTALL | re.IGNORECASE)

    # Extract link text with URL
    def _link_repl(m):
        href = m.group(1)
        link_text = re.sub(r"<[^>]+>", "", m.group(2))
        if href and not href.startswith("#") and not href.startswith("javascript:"):
            return f"{link_text} ({href})"
        return link_text
    text = re.sub(r'<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>', _link_repl, text, flags=re.DOTALL | re.IGNORECASE)

    # Strip remaining HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Decode HTML entities
    text = html.unescape(text)

    # Clean up whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(line for line in lines if line)

    if len(text) > max_length:
        text = text[:max_length] + "\n... (truncated)"

    return text


# ---------------------------------------------------------------------------
# Web Fetch
# ---------------------------------------------------------------------------

def execute_web_fetch(args: Dict, config: Any = None) -> str:
    """Fetch a URL and return the content as readable text."""
    url = args.get("url", "").strip()
    if not url:
        return "Error: url is required"

    # Auto-upgrade http to https
    if url.startswith("http://"):
        url = "https://" + url[7:]
    elif not url.startswith("https://"):
        url = "https://" + url

    err = _is_url_safe(url)
    if err:
        return f"Error: {err}"

    extract_selector = args.get("selector", "")
    include_links = args.get("include_links", False)

    try:
        import httpx
    except ImportError:
        return "Error: httpx not installed"

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/131.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

        with httpx.Client(
            follow_redirects=True,
            timeout=FETCH_TIMEOUT,
            max_redirects=5,
        ) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()

        content_type = response.headers.get("content-type", "")

        # JSON response
        if "json" in content_type:
            try:
                data = response.json()
                text = json.dumps(data, indent=2, ensure_ascii=False)
                if len(text) > MAX_FETCH_BYTES:
                    text = text[:MAX_FETCH_BYTES] + "\n... (truncated)"
                return f"[JSON from {url}]\n{text}"
            except Exception:
                pass

        # Plain text
        if "text/plain" in content_type:
            text = response.text[:MAX_FETCH_BYTES]
            return f"[Text from {url}]\n{text}"

        # HTML → readable text
        raw = response.text
        if len(raw) > MAX_FETCH_BYTES * 3:
            raw = raw[:MAX_FETCH_BYTES * 3]

        text = _html_to_text(raw)

        # Extract title
        title_match = re.search(r"<title[^>]*>(.*?)</title>", response.text[:5000], re.DOTALL | re.IGNORECASE)
        title = html.unescape(title_match.group(1).strip()) if title_match else ""

        header = f"[{title}]({url})" if title else f"[{url}]"
        return f"{header}\n\n{text}"

    except httpx.HTTPStatusError as e:
        return f"Error: HTTP {e.response.status_code} from {url}"
    except httpx.TimeoutException:
        return f"Error: Timeout fetching {url} (>{FETCH_TIMEOUT}s)"
    except httpx.ConnectError as e:
        return f"Error: Connection failed to {url}: {e}"
    except Exception as e:
        return f"Error fetching {url}: {e}"


# ---------------------------------------------------------------------------
# Web Search (DuckDuckGo HTML — no API key needed)
# ---------------------------------------------------------------------------

def _parse_ddg_results(raw_html: str) -> List[Dict[str, str]]:
    """Parse DuckDuckGo HTML search results into structured data."""
    results = []

    # DuckDuckGo HTML results pattern
    # Each result is in a div with class="result" or similar
    result_blocks = re.findall(
        r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?'
        r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
        raw_html, re.DOTALL | re.IGNORECASE,
    )

    if result_blocks:
        for href, title, snippet in result_blocks[:MAX_SEARCH_RESULTS]:
            title = re.sub(r"<[^>]+>", "", title).strip()
            snippet = re.sub(r"<[^>]+>", "", snippet).strip()
            title = html.unescape(title)
            snippet = html.unescape(snippet)
            if href and title:
                results.append({"title": title, "url": href, "snippet": snippet})
        return results

    # Fallback: DuckDuckGo lite results
    lite_blocks = re.findall(
        r'<a[^>]*href="([^"]*)"[^>]*class="result-link"[^>]*>(.*?)</a>.*?'
        r'<td[^>]*class="result-snippet"[^>]*>(.*?)</td>',
        raw_html, re.DOTALL | re.IGNORECASE,
    )

    if lite_blocks:
        for href, title, snippet in lite_blocks[:MAX_SEARCH_RESULTS]:
            title = re.sub(r"<[^>]+>", "", title).strip()
            snippet = re.sub(r"<[^>]+>", "", snippet).strip()
            title = html.unescape(title)
            snippet = html.unescape(snippet)
            if href and title:
                results.append({"title": title, "url": href, "snippet": snippet})
        return results

    # Fallback: generic link extraction from any DDG page
    links = re.findall(
        r'<a[^>]*href="(https?://[^"]*)"[^>]*>(.*?)</a>',
        raw_html, re.DOTALL | re.IGNORECASE,
    )
    seen_hosts = set()
    for href, text in links:
        text = re.sub(r"<[^>]+>", "", text).strip()
        text = html.unescape(text)
        parsed = urlparse(href)
        host = parsed.hostname or ""
        # Skip DuckDuckGo's own links
        if "duckduckgo" in host or not text or len(text) < 5:
            continue
        if host in seen_hosts:
            continue
        seen_hosts.add(host)
        results.append({"title": text[:200], "url": href, "snippet": ""})
        if len(results) >= MAX_SEARCH_RESULTS:
            break

    return results


def execute_web_search(args: Dict, config: Any = None) -> str:
    """Search the web and return results."""
    query = args.get("query", "").strip()
    if not query:
        return "Error: query is required"

    num_results = min(args.get("num_results", 8) or 8, MAX_SEARCH_RESULTS)

    try:
        import httpx
    except ImportError:
        return "Error: httpx not installed"

    # Try DuckDuckGo HTML
    search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/131.0.0.0 Safari/537.36",
            "Accept": "text/html",
            "Accept-Language": "en-US,en;q=0.5",
        }

        with httpx.Client(
            follow_redirects=True,
            timeout=15,
            max_redirects=3,
        ) as client:
            response = client.get(search_url, headers=headers)
            response.raise_for_status()

        results = _parse_ddg_results(response.text)[:num_results]

        if not results:
            return f"No results found for: {query}"

        lines = [f"Search results for: {query}\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['title']}")
            lines.append(f"   {r['url']}")
            if r.get("snippet"):
                lines.append(f"   {r['snippet']}")
            lines.append("")

        return "\n".join(lines)

    except httpx.TimeoutException:
        return f"Error: Search timed out for: {query}"
    except Exception as e:
        return f"Error searching for '{query}': {e}"


# ---------------------------------------------------------------------------
# Browser Control (Playwright-based)
# ---------------------------------------------------------------------------

# Lazy global browser state — persists across tool calls within a code engine run
_browser_state: Dict[str, Any] = {
    "browser": None,
    "context": None,
    "page": None,
    "installed": None,  # None=unknown, True/False=checked
}


def _ensure_playwright() -> Optional[str]:
    """Ensure playwright is installed. Returns error string or None."""
    if _browser_state["installed"] is True:
        return None

    try:
        import playwright  # noqa: F401
        _browser_state["installed"] = True
        return None
    except ImportError:
        pass

    # Try to install playwright
    logger.info("Installing playwright...")
    try:
        result = subprocess.run(
            [".venv/bin/pip", "install", "playwright"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            _browser_state["installed"] = False
            return f"Failed to install playwright: {result.stderr[:200]}"
    except Exception as e:
        _browser_state["installed"] = False
        return f"Failed to install playwright: {e}"

    # Install chromium browser
    try:
        result = subprocess.run(
            [".venv/bin/playwright", "install", "chromium"],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            _browser_state["installed"] = False
            return f"Failed to install chromium: {result.stderr[:200]}"
    except Exception as e:
        _browser_state["installed"] = False
        return f"Failed to install chromium: {e}"

    _browser_state["installed"] = True
    return None


def _get_page():
    """Get or create browser page. Returns (page, error_string)."""
    if _browser_state["page"] is not None:
        return _browser_state["page"], None

    err = _ensure_playwright()
    if err:
        return None, err

    try:
        from playwright.sync_api import sync_playwright
        pw = sync_playwright().start()
        browser = pw.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--window-size=1280,720",
            ],
        )
        context = browser.new_context(
            viewport={"width": 1280, "height": 720},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
        )
        page = context.new_page()

        _browser_state["browser"] = browser
        _browser_state["context"] = context
        _browser_state["page"] = page
        _browser_state["_pw"] = pw

        return page, None

    except Exception as e:
        return None, f"Failed to launch browser: {e}"


def _close_browser() -> None:
    """Close browser and clean up state."""
    try:
        if _browser_state.get("browser"):
            _browser_state["browser"].close()
        if _browser_state.get("_pw"):
            _browser_state["_pw"].stop()
    except Exception:
        pass
    _browser_state["browser"] = None
    _browser_state["context"] = None
    _browser_state["page"] = None
    _browser_state.pop("_pw", None)


def execute_browser_action(args: Dict, config: Any = None) -> str:
    """Execute a browser action. Supports navigate, click, type, screenshot, extract, scroll, close."""
    action = args.get("action", "").strip().lower()
    if not action:
        return "Error: action is required. Options: navigate, click, type, screenshot, extract, scroll, close, get_url"

    # Close action — no page needed
    if action == "close":
        _close_browser()
        return "Browser closed."

    # Validate action before acquiring page (avoids unnecessary playwright setup)
    _VALID_ACTIONS = frozenset({
        "navigate", "click", "type", "press", "screenshot",
        "extract", "scroll", "get_url", "wait", "evaluate",
    })
    if action not in _VALID_ACTIONS:
        return f"Error: Unknown action '{action}'. Options: navigate, click, type, press, screenshot, extract, scroll, close, get_url, wait, evaluate"

    # All other actions need a page
    page, err = _get_page()
    if err:
        return f"Error: {err}"

    try:
        if action == "navigate":
            url = args.get("url", "").strip()
            if not url:
                return "Error: url is required for navigate"
            if not url.startswith("http"):
                url = "https://" + url

            url_err = _is_url_safe(url)
            if url_err:
                return f"Error: {url_err}"

            page.goto(url, wait_until="domcontentloaded", timeout=BROWSER_TIMEOUT * 1000)
            title = page.title()
            return f"Navigated to: {url}\nTitle: {title}"

        elif action == "click":
            selector = args.get("selector", "").strip()
            text = args.get("text", "").strip()
            if not selector and not text:
                return "Error: selector or text is required for click"

            if text and not selector:
                # Click by visible text
                page.get_by_text(text, exact=False).first.click(timeout=BROWSER_TIMEOUT * 1000)
                return f"Clicked element with text: {text}"
            else:
                page.click(selector, timeout=BROWSER_TIMEOUT * 1000)
                return f"Clicked: {selector}"

        elif action == "type":
            selector = args.get("selector", "").strip()
            text = args.get("text", "").strip()
            if not selector or text is None:
                return "Error: selector and text are required for type"

            clear = args.get("clear", True)
            if clear:
                page.fill(selector, text, timeout=BROWSER_TIMEOUT * 1000)
            else:
                page.type(selector, text, timeout=BROWSER_TIMEOUT * 1000)
            return f"Typed into {selector}: {text[:50]}{'...' if len(text) > 50 else ''}"

        elif action == "press":
            key = args.get("key", "").strip()
            if not key:
                return "Error: key is required for press (e.g., 'Enter', 'Tab', 'Escape')"
            selector = args.get("selector", "")
            if selector:
                page.press(selector, key, timeout=BROWSER_TIMEOUT * 1000)
            else:
                page.keyboard.press(key)
            return f"Pressed key: {key}"

        elif action == "screenshot":
            # Take screenshot and save to temp file
            tmp_dir = tempfile.gettempdir()
            filename = f"mother_browser_{int(time.time())}.png"
            path = os.path.join(tmp_dir, filename)
            page.screenshot(path=path, full_page=args.get("full_page", False))
            return f"Screenshot saved to: {path}"

        elif action == "extract":
            selector = args.get("selector", "body").strip()
            try:
                element = page.query_selector(selector)
                if not element:
                    return f"No element found for selector: {selector}"
                text = element.inner_text()
                if len(text) > MAX_FETCH_BYTES:
                    text = text[:MAX_FETCH_BYTES] + "\n... (truncated)"
                return text
            except Exception as e:
                # Fallback: get full page text
                text = page.inner_text("body")
                if len(text) > MAX_FETCH_BYTES:
                    text = text[:MAX_FETCH_BYTES] + "\n... (truncated)"
                return text

        elif action == "scroll":
            direction = args.get("direction", "down").lower()
            amount = args.get("amount", 500)
            if direction == "down":
                page.evaluate(f"window.scrollBy(0, {amount})")
            elif direction == "up":
                page.evaluate(f"window.scrollBy(0, -{amount})")
            elif direction == "top":
                page.evaluate("window.scrollTo(0, 0)")
            elif direction == "bottom":
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            return f"Scrolled {direction} by {amount}px"

        elif action == "get_url":
            return f"Current URL: {page.url}"

        elif action == "wait":
            selector = args.get("selector", "")
            wait_ms = min(args.get("timeout", 5000) or 5000, BROWSER_TIMEOUT * 1000)
            if selector:
                page.wait_for_selector(selector, timeout=wait_ms)
                return f"Element found: {selector}"
            else:
                page.wait_for_timeout(wait_ms)
                return f"Waited {wait_ms}ms"

        elif action == "evaluate":
            expression = args.get("expression", "").strip()
            if not expression:
                return "Error: expression is required for evaluate"
            # Safety: block obviously dangerous JS
            dangerous = ["fetch(", "XMLHttpRequest", "document.cookie", "localStorage", "sessionStorage"]
            for d in dangerous:
                if d.lower() in expression.lower():
                    return f"Error: expression contains blocked pattern: {d}"
            result = page.evaluate(expression)
            return json.dumps(result, indent=2, default=str)[:MAX_FETCH_BYTES]

        else:
            return f"Error: Unknown action '{action}'. Options: navigate, click, type, press, screenshot, extract, scroll, close, get_url, wait, evaluate"

    except Exception as e:
        error_msg = str(e)
        if len(error_msg) > 500:
            error_msg = error_msg[:500] + "..."
        return f"Error: Browser action '{action}' failed: {error_msg}"
