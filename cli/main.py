"""
Motherlabs CLI - Clean, calm, elegant interface.

Usage:
    motherlabs compile "your description here"
    motherlabs compile --file spec.txt
    motherlabs self-compile           # Compile Motherlabs itself
    motherlabs compile-tree "desc"    # Compile as tree of subsystems
    motherlabs corpus list            # List stored compilations
    motherlabs corpus show <hash>     # Show specific compilation
    motherlabs config set provider grok
    motherlabs config get provider

Visual language:
    ◇  insight node (main discovery)
    │  chain continues
    ├─ branch (sub-insight)
    └─ terminal (resolution)
    ✓  done
"""

import sys
import os

# Add project root to path for module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import select
import yaml
import logging
import threading
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

# =============================================================================
# LOGGING SETUP (Phase 5.1: Error Handling & Stability)
# =============================================================================

# CLI logger - separate from engine logger
logger = logging.getLogger("motherlabs.cli")


class JsonLogFormatter(logging.Formatter):
    """
    JSON log formatter for structured logging.

    Phase 19: Produces one JSON object per line for machine consumption.
    """

    def format(self, record: logging.LogRecord) -> str:
        import json as _json
        log_entry = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        return _json.dumps(log_entry)


def setup_logging(
    debug: bool = False,
    log_file: Optional[str] = None,
    json_format: bool = False,
):
    """
    Configure CLI logging.

    Args:
        debug: Enable DEBUG level (otherwise INFO)
        log_file: Optional file path for log output
        json_format: Use structured JSON format (Phase 19)
    """
    level = logging.DEBUG if debug else logging.INFO

    # Create formatter
    if json_format:
        formatter = JsonLogFormatter(datefmt="%Y-%m-%dT%H:%M:%S")
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    # Console handler - only show warnings and above (don't pollute CLI output)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)

    # Configure logger
    logger.setLevel(level)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.debug("CLI logging initialized")


# Colors and styles (ANSI)
class OutputFormat(Enum):
    JSON = "json"
    YAML = "yaml"
    MARKDOWN = "markdown"


class Style:
    RESET = "\033[0m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    ITALIC = "\033[3m"

    # Muted palette - calm, not rainbow
    WHITE = "\033[97m"
    GRAY = "\033[90m"
    CYAN = "\033[36m"      # Accent color (muted)
    GREEN = "\033[32m"     # Success (muted)
    YELLOW = "\033[33m"    # Warning (muted)
    RED = "\033[31m"       # Error (muted)
    BLUE = "\033[34m"      # Info (muted)


# =============================================================================
# OUTPUT FORMATTER (from CLI blueprint - subsystem with JSON, YAML, markdown)
# =============================================================================

class OutputFormatter:
    """Format blueprints in multiple output formats."""

    @staticmethod
    def format(blueprint: Dict[str, Any], fmt: OutputFormat, insights: List[str] = None) -> str:
        if fmt == OutputFormat.JSON:
            return OutputFormatter.to_json(blueprint)
        elif fmt == OutputFormat.YAML:
            return OutputFormatter.to_yaml(blueprint)
        elif fmt == OutputFormat.MARKDOWN:
            return OutputFormatter.to_markdown(blueprint, insights or [])
        else:
            return OutputFormatter.to_json(blueprint)

    @staticmethod
    def to_json(blueprint: Dict[str, Any]) -> str:
        return json.dumps(blueprint, indent=2)

    @staticmethod
    def to_yaml(blueprint: Dict[str, Any]) -> str:
        return yaml.dump(blueprint, default_flow_style=False, sort_keys=False)

    @staticmethod
    def to_markdown(blueprint: Dict[str, Any], insights: List[str]) -> str:
        lines = [
            "# Blueprint",
            "",
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
            "",
            "---",
            "",
        ]

        if insights:
            lines.extend([
                "## Derivation Chain",
                "",
                "Key insights that shaped this architecture:",
                ""
            ])
            for i, insight in enumerate(insights, 1):
                clean = insight.replace("◇ ", "").replace("→ ", "-> ")
                lines.append(f"{i}. {clean}")
            lines.extend(["", "---", ""])

        lines.extend(["## Components", ""])
        for comp in blueprint.get("components", []):
            name = comp.get('name', 'Unnamed')
            ctype = comp.get('type', 'unknown')
            desc = comp.get('description', '')
            derived = comp.get('derived_from', '')

            lines.append(f"### {name}")
            lines.append("")
            lines.append(f"**Type:** {ctype}")
            lines.append("")
            lines.append(desc)
            if derived:
                lines.append("")
                lines.append(f"*Derived from: {derived}*")
            lines.append("")

        if blueprint.get("relationships"):
            lines.extend(["---", "", "## Relationships", ""])
            for rel in blueprint["relationships"]:
                lines.append(f"- **{rel.get('from')}** -> **{rel.get('to')}** ({rel.get('type', 'related')})")
                if rel.get('description'):
                    lines.append(f"  - {rel['description']}")
            lines.append("")

        if blueprint.get("constraints"):
            lines.extend(["---", "", "## Constraints", ""])
            for con in blueprint["constraints"]:
                desc = con.get('description', str(con)) if isinstance(con, dict) else str(con)
                lines.append(f"- {desc}")
            lines.append("")

        if blueprint.get("unresolved"):
            lines.extend(["---", "", "## Unresolved", ""])
            lines.append("These items need clarification:")
            lines.append("")
            for u in blueprint["unresolved"]:
                lines.append(f"- {u}")

        return "\n".join(lines)


# =============================================================================
# PROGRESS INDICATOR (from CLI blueprint)
# =============================================================================

class ProgressIndicator:
    """Shows compilation progress with animated spinner."""

    _FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, cli: 'CLI'):
        self.cli = cli
        self.current_stage = None
        self.stages = ["intent", "personas", "dialogue", "synthesis", "verify"]
        self.stage_idx = 0
        self._spinning = False
        self._spin_message = ""
        self._spinner_thread: Optional[threading.Thread] = None
        self._is_tty = hasattr(sys.stderr, 'isatty') and sys.stderr.isatty()

    def _spin(self):
        """Background spinner loop — writes to stderr with carriage return."""
        idx = 0
        while self._spinning:
            frame = self._FRAMES[idx % len(self._FRAMES)]
            msg = self._spin_message or self.current_stage or ""
            sys.stderr.write(f"\r  {Style.CYAN}{frame}{Style.RESET} {Style.DIM}{msg}{Style.RESET}  ")
            sys.stderr.flush()
            idx += 1
            time.sleep(0.08)
        # Clear spinner line
        sys.stderr.write("\r" + " " * 80 + "\r")
        sys.stderr.flush()

    def _start_spinner(self, message: str = ""):
        if not self._is_tty:
            return
        self._spin_message = message
        self._spinning = True
        self._spinner_thread = threading.Thread(target=self._spin, daemon=True)
        self._spinner_thread.start()

    def _stop_spinner(self):
        if not self._spinning:
            return
        self._spinning = False
        if self._spinner_thread:
            self._spinner_thread.join(timeout=0.5)
            self._spinner_thread = None

    def start(self, stage: str):
        self._stop_spinner()
        self.current_stage = stage
        if stage in self.stages:
            self.stage_idx = self.stages.index(stage)
        self.cli.phase(f"[{self.stage_idx + 1}/{len(self.stages)}] {stage}")
        self._start_spinner(stage)

    def update(self, message: str):
        self._stop_spinner()
        self.cli.insight(message)
        self._start_spinner(message)

    def done(self):
        self._stop_spinner()


# =============================================================================
# CONFIG MANAGER (from CLI blueprint - persists settings)
# =============================================================================

class ConfigManager:
    """Manages CLI configuration (provider, model, etc.)."""

    DEFAULT_CONFIG = {
        "provider": "grok",
        "model": None,  # Use provider default
        "output_format": "json",
        "output_dir": "./output",
    }

    def __init__(self, config_path: Path = None):
        self.config_path = config_path or Path.home() / ".motherlabs" / "config.json"
        self._config = None

    @property
    def config(self) -> Dict[str, Any]:
        if self._config is None:
            self._load()
        return self._config

    def _load(self):
        if self.config_path.exists():
            with open(self.config_path) as f:
                self._config = {**self.DEFAULT_CONFIG, **json.load(f)}
        else:
            self._config = self.DEFAULT_CONFIG.copy()

    def _save(self):
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as f:
            json.dump(self._config, f, indent=2)

    def get(self, key: str) -> Any:
        return self.config.get(key)

    def set(self, key: str, value: Any):
        self.config[key] = value
        self._save()

    def list_all(self) -> Dict[str, Any]:
        return self.config.copy()


def styled(text: str, *styles) -> str:
    """Apply styles to text."""
    prefix = "".join(styles)
    return f"{prefix}{text}{Style.RESET}"


class CLI:
    """
    Motherlabs CLI interface.
    
    Design principles:
    - Minimalistic, clean, calm
    - Each insight chainable to next
    - User can verify each step mentally
    - Optional flagging for disagreement
    """
    
    def __init__(self, interactive_flags: bool = False):
        self.indent = 0
        self.insight_count = 0
        self.in_chain = False
        self.flags: List[int] = []
        self.interactive_flags = interactive_flags
    
    def banner(self):
        """Print minimal banner."""
        print()
        print(styled("  motherlabs", Style.BOLD, Style.WHITE))
        print(styled("  semantic compiler", Style.DIM, Style.GRAY))
        print()
    
    def phase(self, name: str):
        """Start a new phase."""
        if self.in_chain:
            print(styled("  │", Style.DIM))
        print()
        print(styled(f"  {name}", Style.CYAN, Style.BOLD))
        self.in_chain = True
    
    def insight(self, text: str, depth: int = 0):
        """
        Print insight with tree structure.
        
        Visual language:
        ◇ = main node
        │ = chain continues  
        ├─ = branch
        └─ = terminal/resolution
        """
        self.insight_count += 1
        
        # Build prefix based on depth
        if depth == 0:
            # Main insight node
            prefix = "◇"
            style = Style.WHITE
        else:
            # Sub-insight
            prefix = "│ " + "  " * (depth - 1) + "├─"
            style = Style.GRAY
        
        # Check for resolution markers
        if "→" in text and ("resolved" in text.lower() or "conflict" in text.lower()):
            prefix = "└─"
            style = Style.GREEN
        elif "→" in text:
            # Implication
            style = Style.WHITE
        elif "≠" in text:
            # Contrast
            style = Style.YELLOW
        
        line = f"  {prefix} {text}"
        print(styled(line, style))
        
        # Chain connector
        if depth == 0:
            print(styled("  │", Style.DIM))
        
        # Optional: check for flag input (non-blocking)
        if self.interactive_flags:
            self._check_flag()
    
    def branch(self, items: List[str]):
        """Print branch items."""
        for i, item in enumerate(items):
            last = i == len(items) - 1
            prefix = "└─" if last else "├─"
            print(styled(f"    {prefix} {item}", Style.DIM))
    
    def resolution(self, conflict: str, solution: str):
        """Print conflict resolution."""
        print(styled(f"  ├─ conflict: {conflict}", Style.YELLOW))
        print(styled(f"  └─ resolved: {solution}", Style.GREEN))
        print(styled("  │", Style.DIM))
    
    def status(self, message: str):
        """Print status line."""
        print(styled(f"  ⋯ {message}", Style.DIM))
    
    def success(self, files: list, stats: dict = None):
        """Print success with file list and optional stats."""
        print()
        
        # Stats line
        if stats:
            parts = []
            if stats.get("components"):
                parts.append(f"{stats['components']} components")
            if stats.get("conflicts_resolved"):
                parts.append(f"{stats['conflicts_resolved']} conflicts resolved")
            if self.flags:
                parts.append(f"{len(self.flags)} flags")
            else:
                parts.append("0 flags")
            
            print(styled(f"  ✓ {' · '.join(parts)}", Style.GREEN, Style.BOLD))
        else:
            print(styled("  ✓ Done", Style.GREEN, Style.BOLD))
        
        print()
        
        # File tree
        print(styled("  output/", Style.WHITE))
        for i, f in enumerate(files):
            last = i == len(files) - 1
            prefix = "└─" if last else "├─"
            # Just show filename, not full path
            filename = Path(f).name
            print(styled(f"    {prefix} {filename}", Style.GRAY))
        print()
    
    def error(self, message: str):
        """Print error."""
        print()
        print(styled(f"  ✗ {message}", Style.RED))
        print()
    
    def warning(self, message: str):
        """Print warning."""
        print(styled(f"  ⚠ {message}", Style.YELLOW))

    def trust_summary(self, trust: 'TrustIndicators'):
        """Render trust indicators block after compilation.

        Args:
            trust: TrustIndicators dataclass (from core.trust)
        """
        from cli.viewport import _bar

        print()

        # Badge with color
        badge = trust.verification_badge
        if badge == "verified":
            badge_display = styled(f"■ {badge}", Style.GREEN, Style.BOLD)
        elif badge == "partial":
            badge_display = styled(f"■ {badge}", Style.YELLOW, Style.BOLD)
        else:
            badge_display = styled(f"■ {badge}", Style.RED, Style.BOLD)

        print(f"  trust: {badge_display}")

        # Core fidelity scores
        core_dims = ["completeness", "consistency", "coherence", "traceability"]
        for i, dim in enumerate(core_dims):
            score = trust.fidelity_scores.get(dim, 0)
            bar = _bar(score / 100.0)
            last = i == len(core_dims) - 1
            connector = "└─" if last and trust.provenance_depth == 0 and not trust.gap_report else "├─"
            print(styled(f"  {connector} {dim:<14} {score:>3}  {bar}", Style.GRAY))

        # Provenance depth
        max_depth = 3
        print(styled(f"  ├─ provenance depth: {trust.provenance_depth}/{max_depth}", Style.GRAY))

        # Gaps
        n_gaps = len(trust.gap_report)
        gap_label = f"{n_gaps} gap{'s' if n_gaps != 1 else ''}" if n_gaps > 0 else "no gaps"
        print(styled(f"  └─ gaps: {gap_label}", Style.GRAY))

    def rich_error(self, exc: Exception):
        """Render structured error for MotherlabsError, fallback for others.

        Args:
            exc: The exception to render
        """
        if not hasattr(exc, 'to_user_dict'):
            self.error(str(exc))
            return

        info = exc.to_user_dict()
        error_code = info.get("error_code", "")
        user_message = info.get("error", str(exc))
        root_cause = info.get("root_cause", "")
        fix_examples = info.get("fix_examples", [])
        suggestion = info.get("suggestion", "")

        print()
        # Header: error code + title
        if error_code:
            print(styled(f"  ✗ {error_code} — {user_message}", Style.RED, Style.BOLD))
        else:
            print(styled(f"  ✗ {user_message}", Style.RED, Style.BOLD))

        # Root cause
        if root_cause:
            print()
            print(styled(f"  {root_cause}", Style.WHITE))

        # Fix examples
        if fix_examples:
            print()
            print(styled("  Try:", Style.WHITE))
            for i, fix in enumerate(fix_examples):
                last = i == len(fix_examples) - 1
                connector = "└─" if last else "├─"
                print(styled(f"  {connector} {fix}", Style.GRAY))
        elif suggestion:
            print()
            print(styled(f"  {suggestion}", Style.GRAY))

        # Debug hint
        print()
        print(styled("  (Run with DEBUG=1 for full traceback)", Style.DIM))
        print()

    def cost_line(self, engine):
        """Render one-line cost summary from engine token tracking.

        Args:
            engine: MotherlabsEngine instance with _compilation_tokens
        """
        from core.telemetry import estimate_cost

        tokens_list = getattr(engine, '_compilation_tokens', [])
        if not tokens_list:
            return

        total_tokens = sum(tu.total_tokens for tu in tokens_list)
        total_cost = sum(estimate_cost(tu).total_cost for tu in tokens_list)

        if total_cost <= 0 and total_tokens <= 0:
            return

        model = getattr(engine, 'model_name', 'unknown')
        cost_str = f"${total_cost:.2f}" if total_cost >= 0.01 else f"${total_cost:.4f}"
        print(styled(f"  ⋯ {cost_str} · {total_tokens:,} tokens · {model}", Style.DIM))
    
    def prompt(self) -> str:
        """Interactive prompt."""
        print(styled("  What are we building?", Style.WHITE))
        print()
        try:
            user_input = input(styled("  › ", Style.CYAN))
            return user_input.strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return ""
    
    def flag_prompt(self) -> bool:
        """Ask user to resolve flags."""
        if not self.flags:
            return True
        
        print()
        print(styled(f"  {len(self.flags)} points flagged. Resolve? [y/n]", Style.YELLOW))
        try:
            response = input(styled("  › ", Style.CYAN)).strip().lower()
            return response == 'y'
        except:
            return False
    
    def _check_flag(self):
        """Non-blocking check for flag input."""
        # Only works on Unix-like systems
        try:
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                char = sys.stdin.read(1)
                if char.lower() == 'f':
                    self.flags.append(self.insight_count)
                    print(styled("  [flagged]", Style.YELLOW, Style.DIM))
        except:
            pass  # Ignore on Windows or if select fails


def format_trace_md(context_graph: dict) -> str:
    """Format decision trace as readable markdown."""
    lines = [
        "# Decision Trace",
        "",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
        "This document shows how each architectural decision was derived.",
        "",
        "---",
        ""
    ]
    
    # Personas (domain perspectives)
    if context_graph.get("personas"):
        lines.extend(["## Perspectives", ""])
        for p in context_graph["personas"]:
            name = p.get('name', 'Unknown')
            perspective = p.get('perspective', '')
            blind_spots = p.get('blind_spots', '')
            
            lines.append(f"### {name}")
            lines.append("")
            lines.append(f"*{perspective}*")
            if blind_spots:
                lines.append("")
                lines.append(f"Blind spots: {blind_spots}")
            lines.append("")
    
    # Decision trace (dialogue)
    lines.extend(["---", "", "## Dialogue Trace", ""])
    
    for i, msg in enumerate(context_graph.get("decision_trace", []), 1):
        sender = msg.get("sender", "Unknown")
        content = msg.get("content", "")[:500]
        msg_type = msg.get("type", "")
        insight = msg.get("insight", "")
        
        # Format sender with type
        type_badge = f"[{msg_type}]" if msg_type else ""
        lines.append(f"### Turn {i}: {sender} {type_badge}")
        lines.append("")
        
        # Content (indented)
        for para in content.split("\n"):
            if para.strip():
                lines.append(f"> {para}")
        lines.append("")
        
        # Insight highlight
        if insight:
            lines.append(f"**💡 Insight:** {insight}")
            lines.append("")
    
    # Flags (if any)
    if context_graph.get("flags"):
        lines.extend(["---", "", "## ⚠ Flagged Points", ""])
        lines.append("These points were flagged during compilation:")
        lines.append("")
        for flag_idx in context_graph["flags"]:
            insights = context_graph.get("insights", [])
            if flag_idx < len(insights):
                lines.append(f"- Point {flag_idx + 1}: {insights[flag_idx]}")
    
    return "\n".join(lines)


# =============================================================================
# COMMAND IMPLEMENTATIONS (from CLI blueprint)
# =============================================================================

def cmd_compile(args, cli: CLI, config: ConfigManager):
    """CompileCommand: takes input text or file, outputs blueprint."""
    description = None

    if args.file:
        try:
            with open(args.file) as f:
                description = f.read().strip()
            cli.status(f"Reading from {args.file}")
            logger.debug(f"Read input from file: {args.file}")
        except Exception as e:
            logger.error(f"Failed to read file {args.file}: {e}")
            cli.error(f"Cannot read file: {e}")
            sys.exit(1)
    elif args.description:
        # Auto-detect file paths: if the description looks like a file path
        # and that file exists, read it instead of using the filename as input.
        desc = args.description
        if (desc.endswith(('.md', '.txt', '.yaml', '.yml', '.json'))
                or '/' in desc or '\\' in desc):
            try:
                with open(desc) as f:
                    description = f.read().strip()
                cli.status(f"Reading from {desc}")
                logger.debug(f"Auto-detected file path: {desc}")
            except (FileNotFoundError, IsADirectoryError, PermissionError):
                # Not a file — use as literal description
                description = desc
        else:
            description = desc
    else:
        description = cli.prompt()

    if not description:
        logger.warning("No description provided")
        cli.error("No description provided")
        sys.exit(1)

    # Get provider/model from config or args
    provider = args.provider or config.get("provider")
    model = args.model or config.get("model")
    domain = getattr(args, "domain", "software")
    logger.info(f"Starting compile: provider={provider}, model={model}, domain={domain}")
    logger.debug(f"Description: {description[:100]}...")

    # Import engine
    from core.engine import MotherlabsEngine
    from core.exceptions import MotherlabsError

    # Resolve domain adapter
    domain_adapter = None
    if domain != "software":
        from core.adapter_registry import get_adapter
        import adapters  # noqa: F401 — triggers adapter registration
        try:
            domain_adapter = get_adapter(domain)
            cli.status(f"Domain: {domain}")
        except Exception as e:
            cli.error(f"Unknown domain '{domain}': {e}")
            sys.exit(1)

    try:
        progress = ProgressIndicator(cli)
        insights = []

        def on_insight(text):
            insights.append(text)
            progress.update(text)

        def cli_interrogate(request):
            """Blocking stdin interrogation for CLI."""
            from core.interrogation import InterrogationResponse
            progress._stop_spinner()  # Pause spinner during Q&A
            print()
            print(styled("  ◇ Clarification needed", Style.YELLOW, Style.BOLD))
            print(styled(f"    {request.context}", Style.DIM))
            print()
            answers = {}
            for q in request.questions:
                print(styled(f"  ? {q.question}", Style.WHITE))
                if q.options:
                    for i, opt in enumerate(q.options, 1):
                        print(styled(f"      {i}. {opt}", Style.DIM))
                print()
                try:
                    answer = input(styled("    › ", Style.CYAN)).strip()
                    if answer and q.options and answer.isdigit():
                        idx = int(answer) - 1
                        if 0 <= idx < len(q.options):
                            answer = q.options[idx]
                    if answer:
                        answers[q.id] = answer
                except (KeyboardInterrupt, EOFError):
                    return InterrogationResponse(answers={}, skip=True)
                print()
            return InterrogationResponse(answers=answers)

        engine = MotherlabsEngine(
            provider=provider, model=model, on_insight=on_insight,
            domain_adapter=domain_adapter, on_interrogate=cli_interrogate,
        )
        result = engine.compile(description)
        progress.done()

        if result.fracture:
            print()
            print(styled("  ◇ Intent Fracture Detected", Style.YELLOW, Style.BOLD))
            print()
            f = result.fracture
            print(f"    Stage: {styled(f['stage'], Style.CYAN)}")
            print(f"    The intent resolves to competing structures:")
            for i, config in enumerate(f['competing_configs'], 1):
                print(f"      {i}. {styled(config, Style.WHITE)}")
            print()
            print(f"    To resolve: {styled(f['collapsing_constraint'], Style.DIM)}")
            print()
            print(styled("    Re-run with the constraint appended to your description.", Style.DIM))
            print()
            sys.exit(0)

        if result.error:
            logger.error(f"Compilation failed: {result.error}")
            cli.error(result.error)
            sys.exit(1)

        logger.info(f"Compilation successful: {len(result.blueprint.get('components', []))} components")

        # Output format
        fmt_str = args.format or config.get("output_format") or "json"
        fmt = OutputFormat(fmt_str)

        # Write output (includes trust.json)
        output_dir = Path(args.output or config.get("output_dir") or "./output")
        files, stats = write_output(result, output_dir, fmt, insights)
        logger.debug(f"Output written to: {output_dir}")

        cli.success(files, stats)

        # Trust summary
        _show_trust_summary(cli, result)

        # Cost line
        cli.cost_line(engine)

        # Inline viewport (Phase C)
        if getattr(args, "viewport", False) or getattr(args, "emit_code", False):
            from cli.viewport import (
                render_dimensional_summary,
                render_node_map,
                render_interface_contracts,
                render_emission_result,
                render_compilation_overview,
            )
            from core.agent_emission import serialize_emission_result

            dim_meta = result.dimensional_metadata
            imap = result.interface_map

            if getattr(args, "viewport", False):
                print(render_compilation_overview(result.blueprint, dim_meta, imap))
                print()
                print(render_dimensional_summary(dim_meta))
                print()
                print(render_node_map(dim_meta))
                print()
                print(render_interface_contracts(imap))
                print()

            if getattr(args, "emit_code", False):
                cli.phase("Emitting code from blueprint...")
                emission_result = engine.emit_code(
                    result.blueprint,
                    interface_map=None,  # Engine will extract from blueprint
                    dim_meta=None,
                )
                emission_dict = serialize_emission_result(emission_result)
                print(render_emission_result(emission_dict))
                print()

                emission_file = output_dir / "emission.json"
                with open(emission_file, "w") as f:
                    json.dump(emission_dict, f, indent=2)
                cli.status(f"Emission saved to {emission_file}")

                if getattr(args, "write_project", False):
                    from core.project_writer import (
                        write_project as _write_project,
                        ProjectConfig as _ProjectConfig,
                        _infer_project_name,
                    )
                    _generated = {}
                    for batch in emission_result.batch_emissions:
                        for node in batch.emissions:
                            if node.success:
                                _generated[node.component_name] = node.code
                    if _generated:
                        _project_name = _infer_project_name(result.blueprint)
                        _proj_config = _ProjectConfig(project_name=_project_name)
                        _manifest = _write_project(
                            _generated, result.blueprint,
                            str(output_dir), _proj_config,
                        )
                        cli.status(f"Project written to {_manifest.project_dir}")
                        for _f in _manifest.files_written:
                            cli.status(f"  {_f}")
                    else:
                        cli.status("No successful emissions to write")

    except KeyboardInterrupt:
        print()
        logger.info("Compilation interrupted by user")
        cli.status("Interrupted")
        sys.exit(130)
    except MotherlabsError as e:
        logger.error(f"Motherlabs error: {type(e).__name__}: {e}")
        cli.rich_error(e)
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error during compilation: {e}")
        cli.rich_error(e)
        if os.environ.get("DEBUG"):
            import traceback
            traceback.print_exc()
        sys.exit(1)


def cmd_self_compile(args, cli: CLI, config: ConfigManager):
    """Self-compile command: compiles Motherlabs itself."""
    provider = args.provider or config.get("provider")
    model = args.model or config.get("model")
    logger.info(f"Starting self-compile: provider={provider}, model={model}")

    from core.engine import MotherlabsEngine
    from core.exceptions import MotherlabsError

    try:
        insights = []

        def on_insight(text):
            insights.append(text)
            cli.insight(text)

        engine = MotherlabsEngine(provider=provider, model=model, on_insight=on_insight)

        loop_runs = getattr(args, 'loop', 0)
        if loop_runs and loop_runs > 0:
            # Self-compile loop — convergence tracking
            cli.phase(f"Running self-compile loop ({loop_runs} iterations)...")
            report = engine.run_self_compile_loop(runs=loop_runs)

            # Render convergence report
            cli.insight(f"Converged: {report.convergence.is_converged}")
            cli.insight(f"Variance: {report.convergence.variance.variance_score:.2f}")
            cli.insight(f"Runs: {report.convergence.variance.run_count}")
            cli.insight(f"Structural drift: {report.convergence.structural_drift:.2f}")
            cli.insight(f"Code diffs: {len(report.code_diffs)}")
            for diff in report.code_diffs:
                cli.insight(f"  {diff.file_path}: score={diff.overall_score:.2f}")
            cli.insight(f"Patterns extracted: {len(report.patterns)}")
            for pat in report.patterns:
                cli.insight(f"  [{pat.pattern_type}] {pat.name} ({pat.frequency:.0%})")
            cli.insight(f"Overall health: {report.overall_health:.0%}")

            # Write report as JSON
            import json
            from dataclasses import asdict
            output_dir = Path(args.output or config.get("output_dir") or "./output")
            output_dir.mkdir(parents=True, exist_ok=True)
            report_path = output_dir / "self-compile-loop.json"
            with open(report_path, "w") as f:
                json.dump(asdict(report), f, indent=2, default=str)
            cli.insight(f"Report written to {report_path}")
            return

        cli.phase("Self-compiling Motherlabs...")
        result = engine.self_compile()

        if result.error:
            logger.error(f"Self-compile failed: {result.error}")
            cli.error(result.error)
            sys.exit(1)

        logger.info(f"Self-compile successful: {len(result.blueprint.get('components', []))} components")

        # Output format
        fmt_str = args.format or config.get("output_format") or "json"
        fmt = OutputFormat(fmt_str)

        # Write output
        output_dir = Path(args.output or config.get("output_dir") or "./output")
        files, stats = write_output(result, output_dir, fmt, insights)
        logger.debug(f"Output written to: {output_dir}")

        cli.success(files, stats)

    except KeyboardInterrupt:
        print()
        logger.info("Self-compile interrupted by user")
        cli.status("Interrupted")
        sys.exit(130)
    except MotherlabsError as e:
        logger.error(f"Motherlabs error: {type(e).__name__}: {e}")
        cli.rich_error(e)
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error during self-compile: {e}")
        cli.rich_error(e)
        if os.environ.get("DEBUG"):
            import traceback
            traceback.print_exc()
        sys.exit(1)


def cmd_compile_tree(args, cli: CLI, config: ConfigManager):
    """Compile a large task as a tree of subsystems."""
    # Get description
    description = args.description
    if not description and args.file:
        description = Path(args.file).read_text().strip()
    if not description:
        description = cli.prompt("Describe what to build")
    if not description:
        cli.error("No description provided")
        sys.exit(1)

    provider = args.provider or config.get("provider")
    model = args.model or config.get("model")
    max_children = args.max_children

    from core.engine import MotherlabsEngine
    from core.exceptions import MotherlabsError

    try:
        cli.phase(f"Compiling tree (max {max_children} subsystems)...")
        engine = MotherlabsEngine(provider=provider, model=model)
        tree_result = engine.compile_tree(
            description=description,
            max_children=max_children,
        )

        # Render tree result
        root_comps = tree_result.root_blueprint.get("components", [])
        cli.insight(f"Root: {len(root_comps)} components")
        cli.insight(f"Subsystems: {len(tree_result.child_results)}")

        for child in tree_result.child_results:
            status = styled("pass", Style.GREEN) if child.success else styled("fail", Style.RED)
            child_comps = child.blueprint.get("components", []) if child.blueprint else []
            cli.insight(f"  {child.subsystem_name}: {len(child_comps)} components [{status}]")

        cli.insight(f"L2 patterns: {len(tree_result.l2_synthesis.shared_vocabulary)}")
        cli.insight(f"Tree health: {tree_result.tree_health:.0%}")
        cli.insight(f"Total components: {tree_result.total_components}")

        # Write output
        import json
        from dataclasses import asdict
        output_dir = Path(args.output or config.get("output_dir") or "./output")
        output_dir.mkdir(parents=True, exist_ok=True)
        tree_path = output_dir / "compile-tree.json"
        with open(tree_path, "w") as f:
            json.dump(asdict(tree_result), f, indent=2, default=str)
        cli.insight(f"Written to {tree_path}")

    except KeyboardInterrupt:
        print()
        cli.status("Interrupted")
        sys.exit(130)
    except MotherlabsError as e:
        cli.rich_error(e)
        sys.exit(1)
    except Exception as e:
        cli.rich_error(e)
        if os.environ.get("DEBUG"):
            import traceback
            traceback.print_exc()
        sys.exit(1)


def cmd_corpus(args, cli: CLI, config: ConfigManager):
    """CorpusCommand: list/query stored compilations."""
    from persistence.corpus import Corpus

    corpus = Corpus()
    subcommand = args.corpus_command

    if subcommand == "list":
        records = corpus.list_all()
        if not records:
            cli.status("No compilations stored")
            return

        print()
        print(styled("  Stored Compilations:", Style.WHITE, Style.BOLD))
        print()
        for r in records[:20]:  # Limit to 20
            status = styled("ok", Style.GREEN) if r.success else styled("fail", Style.RED)
            domain = r.domain or "unknown"
            print(f"    {r.id[:12]}  [{status}]  {domain}")
        print()
        print(styled(f"  Total: {len(records)}", Style.DIM))
        print()

    elif subcommand == "show":
        if not args.hash:
            cli.error("Usage: motherlabs corpus show <hash>")
            sys.exit(1)

        record = corpus.get(args.hash)
        if not record:
            cli.error(f"No compilation found with hash: {args.hash}")
            sys.exit(1)

        print()
        print(styled(f"  Compilation: {record.canonical_hash[:12]}", Style.WHITE, Style.BOLD))
        print()
        print(styled(f"    Domain: {record.domain}", Style.GRAY))
        print(styled(f"    Success: {record.success}", Style.GRAY))
        print(styled(f"    Provider: {record.provider}", Style.GRAY))
        print(styled(f"    Model: {record.model}", Style.GRAY))
        print()

        if record.blueprint:
            comps = record.blueprint.get("components", [])
            print(styled(f"    Components ({len(comps)}):", Style.WHITE))
            for c in comps[:10]:
                print(styled(f"      - {c.get('name')} ({c.get('type')})", Style.GRAY))
        print()

    elif subcommand == "stats":
        stats = corpus.get_provider_stats()
        if not stats:
            cli.status("No provider stats available")
            return

        print()
        print(styled("  Provider Statistics:", Style.WHITE, Style.BOLD))
        print()
        for provider, pstats in stats.items():
            success_rate = pstats.get("success_rate", 0)
            total = pstats.get("total_compilations", 0)
            color = Style.GREEN if success_rate >= 0.8 else Style.YELLOW if success_rate >= 0.5 else Style.RED
            print(f"    {provider}:")
            print(styled(f"      Compilations: {total}", Style.GRAY))
            print(styled(f"      Success rate: {success_rate:.0%}", color))
        print()

    else:
        cli.error(f"Unknown corpus command: {subcommand}")
        cli.status("Usage: motherlabs corpus [list|show|stats]")
        sys.exit(1)


def cmd_config(args, cli: CLI, config: ConfigManager):
    """ConfigCommand: set provider, model, etc."""
    subcommand = args.config_command

    if subcommand == "list":
        print()
        print(styled("  Configuration:", Style.WHITE, Style.BOLD))
        print()
        for key, value in config.list_all().items():
            print(f"    {key}: {styled(str(value), Style.CYAN)}")
        print()
        print(styled(f"  Config file: {config.config_path}", Style.DIM))
        print()

    elif subcommand == "get":
        if not args.key:
            cli.error("Usage: motherlabs config get <key>")
            sys.exit(1)

        value = config.get(args.key)
        if value is None:
            cli.error(f"Unknown config key: {args.key}")
            sys.exit(1)

        print(f"  {args.key}: {styled(str(value), Style.CYAN)}")

    elif subcommand == "set":
        if not args.key or not args.value:
            cli.error("Usage: motherlabs config set <key> <value>")
            sys.exit(1)

        # Type conversion for known keys
        value = args.value
        if args.key == "output_format" and value not in ["json", "yaml", "markdown"]:
            cli.error(f"Invalid output format. Use: json, yaml, or markdown")
            sys.exit(1)

        config.set(args.key, value)
        print(styled(f"  Set {args.key} = {value}", Style.GREEN))

    else:
        cli.error(f"Unknown config command: {subcommand}")
        cli.status("Usage: motherlabs config [list|get|set]")
        sys.exit(1)


def cmd_viewport(args, cli: CLI, config: ConfigManager):
    """ViewportCommand: visualize a stored compilation's dimensional space."""
    from persistence.corpus import Corpus
    from cli.viewport import (
        render_dimensional_summary,
        render_node_map,
        render_interface_contracts,
        render_fragile_edges,
        render_emission_result,
        render_blueprint_tree,
        render_compilation_overview,
    )

    corpus = Corpus()
    compilation_id = args.compilation_id

    # Load data from corpus
    blueprint = corpus.load_blueprint(compilation_id)
    if not blueprint:
        cli.error(f"No compilation found: {compilation_id}")
        sys.exit(1)

    context_graph = corpus.load_context_graph(compilation_id) or {}
    dim_meta = context_graph.get("dimensional_metadata")
    imap = context_graph.get("interface_map")
    emission = context_graph.get("emission_result")

    # Determine which views to show
    show_map = getattr(args, "map", False)
    show_interfaces = getattr(args, "interfaces", False)
    show_emission = getattr(args, "emission", False)
    show_fragile = getattr(args, "fragile", False)
    show_tree = getattr(args, "tree", False)

    # Custom axes for node map
    axes_arg = getattr(args, "axes", None)
    x_axis = y_axis = None
    if axes_arg:
        parts = axes_arg.split(",")
        x_axis = parts[0].strip() if len(parts) > 0 else None
        y_axis = parts[1].strip() if len(parts) > 1 else None

    # If no specific view requested, show overview
    show_specific = show_map or show_interfaces or show_emission or show_fragile or show_tree
    if not show_specific:
        print(render_compilation_overview(blueprint, dim_meta, imap, emission))
        print()
        print(render_dimensional_summary(dim_meta))
        print()
        print(render_node_map(dim_meta, x_axis, y_axis))
        print()
        print(render_interface_contracts(imap))
        print()
        if emission:
            print(render_emission_result(emission))
            print()
        return

    # Show specific views
    if show_map:
        print(render_node_map(dim_meta, x_axis, y_axis))
        print()
    if show_interfaces:
        print(render_interface_contracts(imap))
        print()
    if show_emission:
        print(render_emission_result(emission))
        print()
    if show_fragile:
        print(render_fragile_edges(dim_meta))
        print()
    if show_tree:
        print(render_blueprint_tree(blueprint))
        print()


def cmd_emit(args, cli: CLI, config: ConfigManager):
    """EmitCommand: emit code from a stored blueprint via LLM agents."""
    from persistence.corpus import Corpus
    from core.engine import MotherlabsEngine
    from core.exceptions import MotherlabsError
    from core.agent_emission import serialize_emission_result
    from cli.viewport import render_emission_result

    corpus = Corpus()
    compilation_id = args.compilation_id

    # Load blueprint and interface map
    blueprint = corpus.load_blueprint(compilation_id)
    if not blueprint:
        cli.error(f"No compilation found: {compilation_id}")
        sys.exit(1)

    context_graph = corpus.load_context_graph(compilation_id) or {}
    imap_dict = context_graph.get("interface_map")
    dim_meta_dict = context_graph.get("dimensional_metadata")

    provider = getattr(args, "provider", None) or config.get("provider")
    model = getattr(args, "model", None) or config.get("model")
    live = getattr(args, "live", False)

    try:
        cli.phase("Emitting code from blueprint...")

        engine = MotherlabsEngine(provider=provider, model=model)

        # Deserialize interface map and dim_meta if available
        imap_obj = None
        dim_meta_obj = None
        if imap_dict:
            from core.interface_schema import deserialize_interface_map
            imap_obj = deserialize_interface_map(imap_dict)
        if dim_meta_dict:
            from core.dimensional import deserialize_dimensional_metadata
            dim_meta_obj = deserialize_dimensional_metadata(dim_meta_dict)

        result = engine.emit_code(
            blueprint,
            interface_map=imap_obj,
            dim_meta=dim_meta_obj,
        )

        # Serialize for rendering
        emission_dict = serialize_emission_result(result)
        print()
        print(render_emission_result(emission_dict))
        print()

        # Write emission to output
        output_dir = Path(getattr(args, "output", None) or config.get("output_dir") or "./output")
        output_dir.mkdir(parents=True, exist_ok=True)
        emission_file = output_dir / "emission.json"
        with open(emission_file, "w") as f:
            json.dump(emission_dict, f, indent=2)

        cli.status(f"Emission saved to {emission_file}")

    except KeyboardInterrupt:
        print()
        cli.status("Interrupted")
        sys.exit(130)
    except MotherlabsError as e:
        cli.rich_error(e)
        sys.exit(1)
    except Exception as e:
        cli.rich_error(e)
        if os.environ.get("DEBUG"):
            import traceback
            traceback.print_exc()
        sys.exit(1)


def cmd_health(args, cli: CLI, config: ConfigManager):
    """Show system health status."""
    from core.engine import MotherlabsEngine

    engine = MotherlabsEngine(llm_client=None, auto_store=False)
    snapshot = engine.get_health_snapshot()

    status = snapshot.get("status", "unknown")
    if status == "healthy":
        cli.insight(f"Status: {styled('healthy', Style.GREEN)}")
    elif status == "degraded":
        cli.insight(f"Status: {styled('degraded', Style.YELLOW)}")
    else:
        cli.insight(f"Status: {styled(status, Style.RED)}")

    cli.insight(f"Uptime: {snapshot.get('uptime_seconds', 0):.0f}s")
    cli.insight(f"Compilations: {snapshot.get('total_compilations', 0)}")
    cli.insight(f"Success rate: {snapshot.get('success_rate', 0):.0%}")
    cli.insight(f"Avg duration: {snapshot.get('avg_duration_seconds', 0):.1f}s")
    cli.insight(f"Cache hit rate: {snapshot.get('cache_hit_rate', 0):.0%}")
    cost = snapshot.get("recent_total_cost_usd", 0)
    if cost > 0:
        cli.insight(f"Recent cost: ${cost:.4f}")
    issues = snapshot.get("issues", [])
    if issues:
        for issue in issues:
            cli.insight(f"  {styled('!', Style.YELLOW)} {issue}")


def cmd_metrics(args, cli: CLI, config: ConfigManager):
    """Show compilation metrics."""
    from core.engine import MotherlabsEngine

    engine = MotherlabsEngine(llm_client=None, auto_store=False)
    metrics = engine.get_metrics()

    if not metrics:
        cli.insight("No metrics recorded yet.")
        return

    cli.insight(f"Total compilations: {metrics.get('total_compilations', 0)}")
    cli.insight(f"Success rate: {metrics.get('success_rate', 0):.0%}")
    cli.insight(f"Avg duration: {metrics.get('avg_duration_seconds', 0):.1f}s")
    cli.insight(f"Total tokens: {metrics.get('total_input_tokens', 0) + metrics.get('total_output_tokens', 0):,}")
    cli.insight(f"Total cost: ${metrics.get('total_cost_usd', 0):.4f}")
    cli.insight(f"Avg cost: ${metrics.get('avg_cost_usd', 0):.4f}")


def cmd_edit(args, cli: CLI, config: ConfigManager):
    """Edit a stored blueprint."""
    from core.engine import MotherlabsEngine
    from persistence.corpus import Corpus

    provider = config.get("provider")
    model = config.get("model")

    # Build edit operations
    edits = []
    for old, new in (args.rename or []):
        edits.append({"operation": "rename", "old_name": old, "new_name": new})
    for name in (args.remove or []):
        edits.append({"operation": "remove", "name": name})
    for desc in (getattr(args, 'add_constraint', None) or []):
        edits.append({"operation": "add_constraint", "description": desc})
    for name, comp_type in (args.add_component or []):
        edits.append({"operation": "add_component", "name": name, "type": comp_type})

    if not edits and not args.recompile:
        cli.error("No edit operations specified. Use --rename, --remove, --add-constraint, or --add-component")
        sys.exit(1)

    try:
        engine = MotherlabsEngine(provider=provider, model=model)

        if edits:
            cli.phase(f"Applying {len(edits)} edits...")
            edit_result = engine.edit_blueprint(args.compilation_id, edits)
            blueprint = edit_result.get("blueprint", {})
            comps = blueprint.get("components", [])
            cli.insight(f"Blueprint: {len(comps)} components after edits")
            warnings = edit_result.get("warnings", [])
            for w in warnings:
                cli.insight(f"  {styled('!', Style.YELLOW)} {w}")

        if args.recompile:
            cli.phase(f"Re-running stage {args.recompile}...")
            recompile_result = engine.recompile_stage(
                args.compilation_id, args.recompile, edits if edits else None
            )
            if recompile_result.get("success"):
                bp = recompile_result.get("blueprint", {})
                cli.insight(f"Recompiled: {len(bp.get('components', []))} components")
            else:
                cli.error(recompile_result.get("error", "Recompile failed"))

    except Exception as e:
        cli.rich_error(e)
        sys.exit(1)


def cmd_trust(args, cli: CLI, config: ConfigManager):
    """TrustCommand: inspect trust indicators for a stored compilation."""
    from persistence.corpus import Corpus
    from core.trust import compute_trust_indicators
    from cli.viewport import _bar, _sparkline

    corpus = Corpus()
    compilation_id = args.compilation_id

    # Load data from corpus
    blueprint = corpus.load_blueprint(compilation_id)
    if not blueprint:
        cli.error(f"No compilation found: {compilation_id}")
        sys.exit(1)

    context_graph = corpus.load_context_graph(compilation_id) or {}
    dim_meta = context_graph.get("dimensional_metadata", {})
    verification = context_graph.get("verification", {})
    intent_keywords = list(context_graph.get("keywords", []))

    try:
        trust = compute_trust_indicators(
            blueprint=blueprint,
            verification=verification,
            context_graph=context_graph,
            dimensional_metadata=dim_meta,
            intent_keywords=intent_keywords,
        )
    except Exception as e:
        cli.error(f"Trust computation failed: {e}")
        sys.exit(1)

    # Full trust report
    cli.trust_summary(trust)

    # Extended: gap report detail
    if trust.gap_report:
        print()
        print(styled("  Gap Report", Style.CYAN, Style.BOLD))
        for i, gap in enumerate(trust.gap_report[:10]):
            last = i == min(len(trust.gap_report), 10) - 1
            connector = "└─" if last else "├─"
            print(styled(f"  {connector} {gap}", Style.GRAY))
        if len(trust.gap_report) > 10:
            print(styled(f"  ... and {len(trust.gap_report) - 10} more", Style.DIM))

    # Extended: silence zones
    if trust.silence_zones:
        print()
        print(styled("  Silence Zones", Style.YELLOW, Style.BOLD))
        for i, zone in enumerate(trust.silence_zones[:5]):
            last = i == min(len(trust.silence_zones), 5) - 1
            connector = "└─" if last else "├─"
            print(styled(f"  {connector} {zone}", Style.GRAY))

    # Extended: confidence trajectory sparkline
    if trust.confidence_trajectory:
        spark = _sparkline(list(trust.confidence_trajectory))
        final = trust.confidence_trajectory[-1] if trust.confidence_trajectory else 0.0
        print()
        print(
            f"  Confidence: {styled(spark, Style.CYAN)} "
            f"{styled(f'{final:.0%}', Style.WHITE, Style.BOLD)}"
        )

    print()


def cmd_keys(args, cli: CLI, config: ConfigManager):
    """KeysCommand: manage API keys (create, list, revoke)."""
    from motherlabs_platform.auth import KeyStore

    db_path = os.environ.get("MOTHERLABS_AUTH_DB")
    store = KeyStore(db_path=db_path) if db_path else KeyStore()

    subcommand = args.keys_command

    if subcommand == "create":
        name = args.name
        rate_limit = args.rate_limit
        budget = args.budget

        key_id, raw_key = store.create_key(name, rate_limit=rate_limit, budget=budget)

        print()
        print(styled("  API Key Created", Style.GREEN, Style.BOLD))
        print()
        print(f"    Key ID:     {styled(key_id, Style.CYAN)}")
        print(f"    Name:       {styled(name, Style.WHITE)}")
        print(f"    Rate limit: {styled(str(rate_limit) + '/hr', Style.GRAY)}")
        print(f"    Budget:     {styled('$' + str(budget), Style.GRAY)}")
        print()
        print(styled("  Raw key (save this — it won't be shown again):", Style.YELLOW))
        print()
        print(f"    {styled(raw_key, Style.WHITE, Style.BOLD)}")
        print()

    elif subcommand == "list":
        keys = store.list_keys()
        if not keys:
            cli.status("No API keys created yet")
            return

        print()
        print(styled("  API Keys:", Style.WHITE, Style.BOLD))
        print()
        for k in keys:
            status = styled("active", Style.GREEN) if k.is_active else styled("revoked", Style.RED)
            budget_pct = (k.spent_usd / k.budget_usd * 100) if k.budget_usd > 0 else 0
            print(f"    {k.id}  {styled(k.name, Style.WHITE):<20}  [{status}]  "
                  f"${k.spent_usd:.2f}/${k.budget_usd:.0f} ({budget_pct:.0f}%)  "
                  f"{styled(str(k.rate_limit_per_hour) + '/hr', Style.DIM)}")
        print()

    elif subcommand == "revoke":
        key_id = args.key_id
        success = store.revoke_key(key_id)
        if success:
            print(styled(f"  Key {key_id} revoked", Style.GREEN))
        else:
            cli.error(f"Key not found: {key_id}")
            sys.exit(1)

    else:
        cli.error(f"Unknown keys command: {subcommand}")
        cli.status("Usage: motherlabs keys [create|list|revoke]")
        sys.exit(1)


def cmd_tools(args, cli: CLI, config: ConfigManager):
    """ToolsCommand: manage tool packages (list, export, import, show)."""
    subcommand = args.tools_command

    if subcommand == "list":
        from motherlabs_platform.tool_registry import ToolRegistry
        db_path = os.environ.get("MOTHERLABS_TOOLS_DB")
        registry = ToolRegistry(db_path=db_path) if db_path else ToolRegistry()

        domain = getattr(args, "domain", None)
        local_only = getattr(args, "local_only", False)
        tools = registry.list_tools(domain=domain, local_only=local_only)

        if not tools:
            cli.status("No tools registered")
            return

        print()
        print(styled("  Tool Registry:", Style.WHITE, Style.BOLD))
        print()
        for t in tools[:30]:
            badge = t.verification_badge
            if badge == "verified":
                badge_display = styled(badge, Style.GREEN)
            elif badge == "partial":
                badge_display = styled(badge, Style.YELLOW)
            else:
                badge_display = styled(badge, Style.RED)
            print(
                f"    {t.package_id[:12]}  {styled(t.name, Style.WHITE):<30}  "
                f"[{badge_display}]  {t.domain}  "
                f"{t.component_count}c/{t.relationship_count}r  "
                f"{styled(f'{t.trust_score:.0f}', Style.CYAN)}"
            )
        print()
        print(styled(f"  Total: {len(tools)}", Style.DIM))
        print()

    elif subcommand == "export":
        compilation_id = args.compilation_id
        if not compilation_id:
            cli.error("Usage: motherlabs tools export <compilation_id>")
            sys.exit(1)

        from core.tool_export import export_tool, export_tool_to_file
        from motherlabs_platform.instance_identity import InstanceIdentityStore
        from motherlabs_platform.tool_registry import ToolRegistry
        from persistence.corpus import Corpus

        try:
            identity_store = InstanceIdentityStore()
            identity = identity_store.get_or_create_self()

            corpus = Corpus()
            pkg = export_tool(
                compilation_id=compilation_id,
                corpus=corpus,
                instance_id=identity.instance_id,
                name=getattr(args, "name", None),
            )

            output_path = getattr(args, "output", None)
            file_path = export_tool_to_file(pkg, output_path)

            # Register locally
            db_path = os.environ.get("MOTHERLABS_TOOLS_DB")
            registry = ToolRegistry(db_path=db_path) if db_path else ToolRegistry()
            try:
                registry.register_tool(pkg, is_local=True)
            except Exception:
                pass  # Already registered

            print()
            print(styled("  Tool Exported", Style.GREEN, Style.BOLD))
            print()
            print(f"    Package ID:  {styled(pkg.package_id, Style.CYAN)}")
            print(f"    Name:        {styled(pkg.name, Style.WHITE)}")
            print(f"    Trust:       {styled(f'{pkg.trust_score:.1f}', Style.WHITE)} [{pkg.verification_badge}]")
            print(f"    Components:  {len(pkg.blueprint.get('components', []))}")
            print(f"    File:        {styled(file_path, Style.DIM)}")
            print()

        except ValueError as e:
            cli.error(str(e))
            sys.exit(1)
        except Exception as e:
            cli.error(f"Export failed: {e}")
            if os.environ.get("DEBUG"):
                import traceback
                traceback.print_exc()
            sys.exit(1)

    elif subcommand == "import":
        file_path = args.file_path
        if not file_path:
            cli.error("Usage: motherlabs tools import <path.mtool>")
            sys.exit(1)

        from core.tool_export import load_tool_from_file, import_tool
        from motherlabs_platform.instance_identity import InstanceIdentityStore
        from motherlabs_platform.tool_registry import ToolRegistry

        try:
            pkg = load_tool_from_file(file_path)
        except (FileNotFoundError, ValueError) as e:
            cli.error(str(e))
            sys.exit(1)

        identity_store = InstanceIdentityStore()
        identity = identity_store.get_or_create_self()
        db_path = os.environ.get("MOTHERLABS_TOOLS_DB")
        registry = ToolRegistry(db_path=db_path) if db_path else ToolRegistry()

        min_trust = getattr(args, "min_trust", 60.0)
        require_verified = getattr(args, "require_verified", False)

        result = import_tool(
            package=pkg,
            registry=registry,
            instance_id=identity.instance_id,
            min_trust_score=min_trust,
            require_verified=require_verified,
        )

        if result.allowed:
            print()
            print(styled("  Tool Imported", Style.GREEN, Style.BOLD))
            print()
            print(f"    Package ID:  {styled(pkg.package_id, Style.CYAN)}")
            print(f"    Name:        {styled(pkg.name, Style.WHITE)}")
            print(f"    Trust:       {styled(f'{pkg.trust_score:.1f}', Style.WHITE)} [{pkg.verification_badge}]")
            print(f"    Source:      {pkg.source_instance_id}")
            print(f"    Checks:      {', '.join(result.checks_performed)}")
            if result.warnings:
                print()
                for w in result.warnings:
                    cli.warning(w)
            print()
        else:
            print()
            print(styled(f"  Import Rejected", Style.RED, Style.BOLD))
            print()
            print(styled(f"    {result.rejection_reason}", Style.WHITE))
            print()
            print(styled(f"    Provenance:  {'OK' if result.provenance_valid else 'FAIL'}", Style.GRAY))
            print(styled(f"    Trust:       {'OK' if result.trust_sufficient else 'FAIL'}", Style.GRAY))
            print(styled(f"    Code Safety: {'OK' if result.code_safe else 'FAIL'}", Style.GRAY))
            print()
            sys.exit(1)

    elif subcommand == "show":
        package_id = args.package_id
        if not package_id:
            cli.error("Usage: motherlabs tools show <package_id>")
            sys.exit(1)

        from motherlabs_platform.tool_registry import ToolRegistry
        db_path = os.environ.get("MOTHERLABS_TOOLS_DB")
        registry = ToolRegistry(db_path=db_path) if db_path else ToolRegistry()
        pkg = registry.get_tool(package_id)

        if not pkg:
            cli.error(f"Tool not found: {package_id}")
            sys.exit(1)

        comps = pkg.blueprint.get("components", [])
        rels = pkg.blueprint.get("relationships", [])

        print()
        print(styled(f"  Tool: {pkg.name}", Style.WHITE, Style.BOLD))
        print()
        print(f"    Package ID:  {styled(pkg.package_id, Style.CYAN)}")
        print(f"    Version:     {pkg.version}")
        print(f"    Domain:      {pkg.domain}")
        print(f"    Trust:       {styled(f'{pkg.trust_score:.1f}', Style.WHITE)} [{pkg.verification_badge}]")
        print(f"    Fingerprint: {pkg.fingerprint}")
        print(f"    Source:      {pkg.source_instance_id}")
        print(f"    Created:     {pkg.created_at}")
        print()
        print(styled(f"    Components ({len(comps)}):", Style.WHITE))
        for c in comps[:10]:
            print(styled(f"      - {c.get('name')} ({c.get('type')})", Style.GRAY))
        print(styled(f"    Relationships: {len(rels)}", Style.WHITE))
        print(styled(f"    Code files: {len(pkg.generated_code)}", Style.WHITE))
        print()
        print(styled(f"    Provenance chain ({len(pkg.provenance_chain)}):", Style.WHITE))
        for rec in pkg.provenance_chain:
            print(styled(
                f"      {rec.instance_id[:12]}  {rec.timestamp}  "
                f"[{rec.verification_badge}]  {rec.provider}",
                Style.GRAY,
            ))
        print()

    else:
        cli.error(f"Unknown tools command: {subcommand}")
        cli.status("Usage: motherlabs tools [list|export|import|show]")
        sys.exit(1)


def cmd_goal(args, cli: CLI, config: ConfigManager):
    """GoalCommand: manage persistent goals for autonomous mode."""
    from mother.goals import GoalStore, compute_goal_health

    db_path = Path.home() / ".motherlabs" / "history.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    store = GoalStore(db_path)
    subcommand = args.goal_command

    if subcommand == "add":
        goal_id = store.add(
            description=args.description,
            source="cli",
            priority=args.priority,
        )
        print()
        print(styled("  Goal added", Style.WHITE, Style.BOLD))
        print()
        print(f"    ID:       {styled(str(goal_id), Style.CYAN)}")
        print(f"    Priority: {styled(args.priority, Style.WHITE)}")
        print(f"    {styled(args.description, Style.DIM)}")
        print()

    elif subcommand == "list":
        goals = store.active()
        if not goals:
            print()
            cli.status("No active goals")
            print()
            store.close()
            return

        now = time.time()
        print()
        print(styled("  Active Goals", Style.WHITE, Style.BOLD))
        print()
        for g in goals:
            health = compute_goal_health(g, now=now)
            health_pct = f"{health * 100:.0f}%"
            health_color = Style.GREEN if health > 0.6 else (Style.YELLOW if health > 0.3 else Style.RED)
            print(
                f"    {styled(str(g.goal_id), Style.CYAN):>6}  "
                f"{styled(health_pct, health_color):>6}  "
                f"{styled(g.priority, Style.DIM):<8}  "
                f"{g.description}"
            )
        print()

    elif subcommand == "dismiss":
        goal = store.get(args.goal_id)
        if not goal:
            cli.error(f"Goal {args.goal_id} not found")
            store.close()
            sys.exit(1)
        store.update_status(args.goal_id, "dismissed", completion_note="dismissed via CLI")
        print(styled(f"  Dismissed goal {args.goal_id}", Style.GREEN))

    else:
        cli.error("Usage: motherlabs goal [add|list|dismiss]")
        sys.exit(1)

    store.close()


def cmd_activate(args, cli: CLI, config: ConfigManager):
    """ActivateCommand: enable autonomous operating mode."""
    from mother.config import load_config, save_config

    mother_config = load_config()
    mother_config.autonomous_enabled = True
    if args.perception:
        mother_config.screen_monitoring = True

    saved_path = save_config(mother_config)

    print()
    print(styled("  Autonomous mode activated", Style.WHITE, Style.BOLD))
    print()
    print(f"    autonomous_enabled:  {styled('True', Style.GREEN)}")
    if args.perception:
        print(f"    screen_monitoring:   {styled('True', Style.GREEN)}")
    print(f"    config: {styled(str(saved_path), Style.DIM)}")
    print()
    cli.status("Add goals with: motherlabs goal add \"description\"")
    print()


def cmd_instance(args, cli: CLI, config: ConfigManager):
    """InstanceCommand: manage instance identity and peers."""
    subcommand = args.instance_command

    if subcommand == "info":
        from motherlabs_platform.instance_identity import (
            InstanceIdentityStore,
            build_trust_graph_digest,
            serialize_trust_graph_digest,
        )
        from motherlabs_platform.tool_registry import ToolRegistry

        identity_store = InstanceIdentityStore()
        identity = identity_store.get_or_create_self()

        db_path = os.environ.get("MOTHERLABS_TOOLS_DB")
        registry = ToolRegistry(db_path=db_path) if db_path else ToolRegistry()

        digest = build_trust_graph_digest(
            identity.instance_id, identity.name, registry,
        )

        print()
        print(styled("  Instance Info", Style.WHITE, Style.BOLD))
        print()
        print(f"    Instance ID: {styled(identity.instance_id, Style.CYAN)}")
        print(f"    Name:        {styled(identity.name, Style.WHITE)}")
        print(f"    Created:     {identity.created_at}")
        print()
        print(styled("  Trust Graph", Style.WHITE, Style.BOLD))
        print()
        print(f"    Tools:       {digest.tool_count} ({digest.verified_tool_count} verified)")
        print(f"    Avg Trust:   {styled(f'{digest.avg_trust_score:.1f}', Style.CYAN)}")
        if digest.domain_counts:
            for domain, count in sorted(digest.domain_counts.items()):
                print(f"    {domain}:  {count}")
        print()

    elif subcommand == "peers":
        from motherlabs_platform.instance_identity import InstanceIdentityStore

        store = InstanceIdentityStore()
        peers = store.list_peers()

        if not peers:
            cli.status("No peers registered")
            return

        print()
        print(styled("  Known Peers:", Style.WHITE, Style.BOLD))
        print()
        for p in peers:
            print(
                f"    {p.instance_id[:12]}  {styled(p.name, Style.WHITE):<20}  "
                f"{styled(p.api_endpoint, Style.DIM)}"
            )
        print()

    else:
        cli.error(f"Unknown instance command: {subcommand}")
        cli.status("Usage: motherlabs instance [info|peers]")
        sys.exit(1)


def cmd_agent(args, cli: CLI, config: ConfigManager):
    """AgentCommand: build a project from a description (intent → blueprint → code → files)."""
    from core.engine import MotherlabsEngine
    from core.exceptions import MotherlabsError
    from core.agent_orchestrator import AgentOrchestrator, AgentConfig, serialize_agent_result

    description = None

    if getattr(args, "file", None):
        try:
            with open(args.file) as f:
                description = f.read().strip()
            cli.status(f"Reading from {args.file}")
        except Exception as e:
            cli.error(f"Cannot read file: {e}")
            sys.exit(1)
    elif args.description:
        description = args.description
    else:
        description = cli.prompt()

    if not description:
        cli.error("No description provided")
        sys.exit(1)

    provider = getattr(args, "provider", None) or config.get("provider")
    model = getattr(args, "model", None) or config.get("model")
    output_dir = getattr(args, "output", None) or config.get("output_dir") or "./output"
    codegen_mode = getattr(args, "mode", "llm")
    enrich = not getattr(args, "no_enrich", False)
    dry_run = getattr(args, "dry_run", False)
    build = getattr(args, "build", False)
    domain = getattr(args, "domain", "software")

    # Resolve domain adapter
    domain_adapter = None
    if domain != "software":
        from core.adapter_registry import get_adapter
        import adapters  # noqa: F401 — triggers adapter registration
        try:
            domain_adapter = get_adapter(domain)
            cli.status(f"Domain: {domain}")
        except Exception as e:
            cli.error(f"Unknown domain '{domain}': {e}")
            sys.exit(1)

    agent_config = AgentConfig(
        codegen_mode=codegen_mode,
        enrich_input=enrich,
        write_project=not dry_run,
        output_dir=output_dir,
        build=build and not dry_run,
    )

    try:
        engine = MotherlabsEngine(provider=provider, model=model, domain_adapter=domain_adapter)
        orch = AgentOrchestrator(engine, agent_config)

        # Phase count depends on whether build is enabled
        total_phases = 5 if agent_config.build else 4

        def on_progress(phase, msg):
            phase_map = {
                "quality": f"[1/{total_phases}] Understanding intent",
                "enrich":  f"[1/{total_phases}] Understanding intent",
                "compile": f"[2/{total_phases}] Compiling blueprint",
                "emit":    f"[3/{total_phases}] Generating code",
                "write":   f"[4/{total_phases}] Writing project",
                "build":   f"[5/{total_phases}] Validating & fixing",
            }
            header = phase_map.get(phase, phase)
            if msg and not msg.startswith("["):
                cli.phase(header)
                cli.insight(msg)

        result = orch.run(description, on_progress=on_progress)

        if not result.success:
            cli.error(result.error or "Agent failed")
            sys.exit(1)

        # Summary
        n_components = len(result.blueprint.get("components", []))
        n_relationships = len(result.blueprint.get("relationships", []))
        n_files = len(result.project_manifest.files_written) if result.project_manifest else 0
        total_lines = result.project_manifest.total_lines if result.project_manifest else 0

        print()
        if result.project_manifest:
            project_name = Path(result.project_manifest.project_dir).name
            print(styled(f"  ✓ {project_name}/ — {n_components} components · {n_files} files", Style.GREEN, Style.BOLD))
            print()

            # File tree
            print(styled(f"  {project_name}/", Style.WHITE))
            files = sorted(result.project_manifest.files_written)
            for i, f in enumerate(files):
                last = i == len(files) - 1
                prefix = "└─" if last else "├─"
                print(styled(f"    {prefix} {f}", Style.GRAY))
            print()
        else:
            print(styled(f"  ✓ {n_components} components · {len(result.generated_code)} code blocks", Style.GREEN, Style.BOLD))
            print()

        # Trust summary
        if result.compile_result:
            _show_trust_summary(cli, result.compile_result)
        else:
            _show_trust_summary_from_blueprint(cli, result.blueprint)

        # Cost line
        cli.cost_line(engine)

        # Save agent result as JSON
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / "agent-result.json"
        with open(result_file, "w") as f:
            json.dump(serialize_agent_result(result), f, indent=2)
        cli.status(f"Agent result saved to {result_file}")

    except KeyboardInterrupt:
        print()
        cli.status("Interrupted")
        sys.exit(130)
    except MotherlabsError as e:
        cli.rich_error(e)
        sys.exit(1)
    except Exception as e:
        cli.rich_error(e)
        if os.environ.get("DEBUG"):
            import traceback
            traceback.print_exc()
        sys.exit(1)


def _show_trust_summary(cli: CLI, result) -> None:
    """Compute and display trust indicators from a CompileResult."""
    try:
        from core.trust import compute_trust_indicators
        trust = compute_trust_indicators(
            blueprint=result.blueprint,
            verification=result.verification or {},
            context_graph=result.context_graph or {},
            dimensional_metadata=result.dimensional_metadata or {},
            intent_keywords=list(result.context_graph.get("keywords", [])) if result.context_graph else [],
        )
        cli.trust_summary(trust)
    except Exception:
        pass  # Never let trust display crash the CLI


def _show_trust_summary_from_blueprint(cli: CLI, blueprint: Dict[str, Any]) -> None:
    """Compute and display trust indicators from a bare blueprint (agent path)."""
    try:
        from core.trust import compute_trust_indicators
        trust = compute_trust_indicators(
            blueprint=blueprint,
            verification={},
            context_graph={},
            dimensional_metadata={},
            intent_keywords=[],
        )
        cli.trust_summary(trust)
    except Exception:
        pass


def write_output(result, output_dir: Path, fmt: OutputFormat = OutputFormat.JSON, insights: List[str] = None) -> tuple:
    """Write output files and return (files, stats)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    files = []
    insights = insights or []

    # Blueprint in requested format
    if fmt == OutputFormat.JSON:
        blueprint_file = output_dir / "blueprint.json"
        with open(blueprint_file, "w") as f:
            f.write(OutputFormatter.to_json(result.blueprint))
    elif fmt == OutputFormat.YAML:
        blueprint_file = output_dir / "blueprint.yaml"
        with open(blueprint_file, "w") as f:
            f.write(OutputFormatter.to_yaml(result.blueprint))
    else:  # MARKDOWN
        blueprint_file = output_dir / "blueprint.md"
        with open(blueprint_file, "w") as f:
            f.write(OutputFormatter.to_markdown(result.blueprint, insights))
    files.append(str(blueprint_file))

    # Context graph (always JSON)
    context_json = output_dir / "context-graph.json"
    with open(context_json, "w") as f:
        json.dump(result.context_graph, f, indent=2)
    files.append(str(context_json))

    # Trace (markdown)
    trace_md = output_dir / "trace.md"
    with open(trace_md, "w") as f:
        f.write(format_trace_md(result.context_graph))
    files.append(str(trace_md))

    # Trust indicators (JSON)
    try:
        from core.trust import compute_trust_indicators, serialize_trust_indicators
        trust = compute_trust_indicators(
            blueprint=result.blueprint,
            verification=result.verification or {},
            context_graph=result.context_graph or {},
            dimensional_metadata=result.dimensional_metadata or {},
            intent_keywords=list(result.context_graph.get("keywords", [])) if result.context_graph else [],
        )
        trust_file = output_dir / "trust.json"
        with open(trust_file, "w") as f:
            json.dump(serialize_trust_indicators(trust), f, indent=2)
        files.append(str(trust_file))
    except Exception:
        pass  # Don't fail output writing if trust computation fails

    # Stats
    stats = {
        "components": len(result.blueprint.get("components", [])),
        "conflicts_resolved": len([
            c for c in result.blueprint.get("constraints", [])
            if "resolved" in str(c).lower()
        ]) or 0
    }

    return files, stats


def main():
    """Main CLI entry point - dispatches to subcommands."""
    config = ConfigManager()

    # Initialize logging (Phase 5.1, enhanced Phase 19)
    debug_mode = os.environ.get("DEBUG") or os.environ.get("MOTHERLABS_DEBUG")
    log_file = os.environ.get("MOTHERLABS_LOG")
    json_log = bool(os.environ.get("MOTHERLABS_JSON_LOG")) or "--json-log" in sys.argv
    setup_logging(debug=bool(debug_mode), log_file=log_file, json_format=json_log)

    parser = argparse.ArgumentParser(
        description="Motherlabs - semantic compiler",
        usage="motherlabs <command> [options]"
    )
    parser.add_argument("--json-log", action="store_true", help="Use structured JSON logging (Phase 19)")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # compile command
    compile_parser = subparsers.add_parser("compile", help="Compile natural language to blueprint")
    compile_parser.add_argument("description", nargs="?", help="What to build")
    compile_parser.add_argument("--file", "-f", help="Read description from file")
    compile_parser.add_argument("--output", "-o", help="Output directory")
    compile_parser.add_argument("--format", choices=["json", "yaml", "markdown"], help="Output format")
    compile_parser.add_argument("--provider", "-p", choices=["grok", "claude", "openai", "gemini"], help="LLM provider")
    compile_parser.add_argument("--model", "-m", help="Model override")
    compile_parser.add_argument("--viewport", action="store_true", help="Show viewport after compilation")
    compile_parser.add_argument("--emit", action="store_true", dest="emit_code", help="Emit code after compilation")
    compile_parser.add_argument("--write", action="store_true", dest="write_project", help="Write emitted code as runnable project files")
    compile_parser.add_argument("--domain", "-d", default="software", help="Domain adapter (software, process, ...)")

    # self-compile command
    self_parser = subparsers.add_parser("self-compile", help="Compile Motherlabs itself")
    self_parser.add_argument("--output", "-o", help="Output directory")
    self_parser.add_argument("--format", choices=["json", "yaml", "markdown"], help="Output format")
    self_parser.add_argument("--provider", "-p", choices=["grok", "claude", "openai", "gemini"], help="LLM provider")
    self_parser.add_argument("--model", "-m", help="Model override")
    self_parser.add_argument("--loop", type=int, default=0, metavar="N",
                             help="Run self-compile loop N times (convergence tracking)")

    # compile-tree command
    tree_parser = subparsers.add_parser("compile-tree", help="Compile as tree of subsystems")
    tree_parser.add_argument("description", nargs="?", help="What to build")
    tree_parser.add_argument("--file", "-f", help="Read description from file")
    tree_parser.add_argument("--output", "-o", help="Output directory")
    tree_parser.add_argument("--format", choices=["json", "yaml", "markdown"], help="Output format")
    tree_parser.add_argument("--provider", "-p", choices=["grok", "claude", "openai", "gemini"], help="LLM provider")
    tree_parser.add_argument("--model", "-m", help="Model override")
    tree_parser.add_argument("--max-children", type=int, default=8, help="Max subsystem decompositions")

    # corpus command
    corpus_parser = subparsers.add_parser("corpus", help="List/query stored compilations")
    corpus_parser.add_argument("corpus_command", choices=["list", "show", "stats"], help="Corpus subcommand")
    corpus_parser.add_argument("hash", nargs="?", help="Compilation hash (for show)")

    # config command
    config_parser = subparsers.add_parser("config", help="Manage configuration")
    config_parser.add_argument("config_command", choices=["list", "get", "set"], help="Config subcommand")
    config_parser.add_argument("key", nargs="?", help="Config key")
    config_parser.add_argument("value", nargs="?", help="Config value (for set)")

    # viewport command
    viewport_parser = subparsers.add_parser("viewport", help="Visualize a stored compilation")
    viewport_parser.add_argument("compilation_id", help="Compilation ID (hash)")
    viewport_parser.add_argument("--map", action="store_true", help="Show 2D node map")
    viewport_parser.add_argument("--interfaces", action="store_true", help="Show interface contracts")
    viewport_parser.add_argument("--emission", action="store_true", help="Show emission results")
    viewport_parser.add_argument("--fragile", action="store_true", help="Show fragile edges")
    viewport_parser.add_argument("--tree", action="store_true", help="Show blueprint tree")
    viewport_parser.add_argument("--axes", help="Custom axes for node map (x,y)")

    # emit command
    emit_parser = subparsers.add_parser("emit", help="Emit code from a stored blueprint")
    emit_parser.add_argument("compilation_id", help="Compilation ID (hash)")
    emit_parser.add_argument("--live", action="store_true", help="Show batch progress inline")
    emit_parser.add_argument("--output", "-o", help="Output directory")
    emit_parser.add_argument("--provider", "-p", choices=["grok", "claude", "openai", "gemini"], help="LLM provider")
    emit_parser.add_argument("--model", "-m", help="Model override")

    # agent command
    agent_parser = subparsers.add_parser("agent", help="Build a project from a description")
    agent_parser.add_argument("description", nargs="?", help="What to build")
    agent_parser.add_argument("--file", "-f", help="Read description from file")
    agent_parser.add_argument("--output", "-o", default="./output", help="Output directory")
    agent_parser.add_argument("--provider", "-p", choices=["grok", "claude", "openai", "gemini"], help="LLM provider")
    agent_parser.add_argument("--model", "-m", help="Model override")
    agent_parser.add_argument("--mode", choices=["llm", "template"], default="llm", help="Code generation mode")
    agent_parser.add_argument("--no-enrich", action="store_true", help="Skip input enrichment")
    agent_parser.add_argument("--dry-run", action="store_true", help="Compile and emit but don't write project files")
    agent_parser.add_argument("--build", action="store_true", help="Run build loop: install, test, fix (Phase 27)")
    agent_parser.add_argument("--domain", "-d", default="software", help="Domain adapter (software, process, ...)")

    # health command
    subparsers.add_parser("health", help="Show system health status")

    # metrics command
    subparsers.add_parser("metrics", help="Show compilation metrics")

    # edit command
    edit_parser = subparsers.add_parser("edit", help="Edit a stored blueprint")
    edit_parser.add_argument("compilation_id", help="Compilation ID to edit")
    edit_parser.add_argument("--rename", nargs=2, metavar=("OLD", "NEW"), action="append", help="Rename component")
    edit_parser.add_argument("--remove", action="append", metavar="NAME", help="Remove component")
    edit_parser.add_argument("--add-constraint", metavar="DESC", action="append", help="Add constraint")
    edit_parser.add_argument("--add-component", nargs=2, metavar=("NAME", "TYPE"), action="append", help="Add component")
    edit_parser.add_argument("--recompile", metavar="STAGE", help="Re-run stage after edits (EXPAND/DECOMPOSE/GROUND/CONSTRAIN/ARCHITECT)")

    # trust command
    trust_parser = subparsers.add_parser("trust", help="Inspect trust indicators for a stored compilation")
    trust_parser.add_argument("compilation_id", help="Compilation ID (hash)")

    # keys command
    keys_parser = subparsers.add_parser("keys", help="Manage API keys")
    keys_parser.add_argument("keys_command", choices=["create", "list", "revoke"], help="Keys subcommand")
    keys_parser.add_argument("--name", help="Key name (for create)")
    keys_parser.add_argument("--rate-limit", type=int, default=100, help="Requests per hour (default: 100)")
    keys_parser.add_argument("--budget", type=float, default=50.0, help="Budget in USD (default: 50)")
    keys_parser.add_argument("key_id", nargs="?", help="Key ID (for revoke)")

    # tools command
    tools_parser = subparsers.add_parser("tools", help="Manage tool packages")
    tools_parser.add_argument("tools_command", choices=["list", "export", "import", "show"], help="Tools subcommand")
    tools_parser.add_argument("compilation_id", nargs="?", help="Compilation ID (for export)")
    tools_parser.add_argument("file_path", nargs="?", help="File path (for import)")
    tools_parser.add_argument("package_id", nargs="?", help="Package ID (for show)")
    tools_parser.add_argument("--name", help="Tool name (for export)")
    tools_parser.add_argument("--output", "-o", help="Output file path (for export)")
    tools_parser.add_argument("--domain", "-d", help="Filter by domain (for list)")
    tools_parser.add_argument("--local-only", action="store_true", help="Show only locally compiled tools")
    tools_parser.add_argument("--min-trust", type=float, default=60.0, help="Minimum trust score for import")
    tools_parser.add_argument("--require-verified", action="store_true", help="Require 'verified' badge for import")

    # instance command
    instance_parser = subparsers.add_parser("instance", help="Instance identity and peers")
    instance_parser.add_argument("instance_command", choices=["info", "peers"], help="Instance subcommand")

    # goal command
    goal_parser = subparsers.add_parser("goal", help="Manage persistent goals")
    goal_sub = goal_parser.add_subparsers(dest="goal_command")
    goal_add = goal_sub.add_parser("add", help="Add a goal")
    goal_add.add_argument("description", help="Goal description")
    goal_add.add_argument("--priority", choices=["urgent", "high", "normal", "low"], default="normal")
    goal_sub.add_parser("list", help="List active goals")
    goal_dismiss = goal_sub.add_parser("dismiss", help="Dismiss a goal")
    goal_dismiss.add_argument("goal_id", type=int, help="Goal ID to dismiss")

    # activate command
    activate_parser = subparsers.add_parser("activate", help="Enable autonomous operating mode")
    activate_parser.add_argument("--perception", action="store_true", help="Also enable screen monitoring")

    args = parser.parse_args()
    cli = CLI()

    # No command - show help or interactive mode
    if not args.command:
        cli.banner()
        parser.print_help()
        print()
        print(styled("  Quick start:", Style.WHITE))
        print(styled('    motherlabs compile "Build a task manager"', Style.DIM))
        print(styled("    motherlabs self-compile", Style.DIM))
        print(styled("    motherlabs corpus list", Style.DIM))
        print(styled('    motherlabs viewport <compilation_id>', Style.DIM))
        print(styled('    motherlabs emit <compilation_id>', Style.DIM))
        print(styled('    motherlabs agent "Build a todo app with teams"', Style.DIM))
        print(styled('    motherlabs trust <compilation_id>', Style.DIM))
        print(styled('    motherlabs keys create --name "my-key"', Style.DIM))
        print()
        sys.exit(0)

    cli.banner()

    # Dispatch to command handlers
    if args.command == "compile":
        cmd_compile(args, cli, config)
    elif args.command == "self-compile":
        cmd_self_compile(args, cli, config)
    elif args.command == "compile-tree":
        cmd_compile_tree(args, cli, config)
    elif args.command == "corpus":
        cmd_corpus(args, cli, config)
    elif args.command == "config":
        cmd_config(args, cli, config)
    elif args.command == "viewport":
        cmd_viewport(args, cli, config)
    elif args.command == "emit":
        cmd_emit(args, cli, config)
    elif args.command == "agent":
        cmd_agent(args, cli, config)
    elif args.command == "health":
        cmd_health(args, cli, config)
    elif args.command == "metrics":
        cmd_metrics(args, cli, config)
    elif args.command == "edit":
        cmd_edit(args, cli, config)
    elif args.command == "trust":
        cmd_trust(args, cli, config)
    elif args.command == "keys":
        cmd_keys(args, cli, config)
    elif args.command == "tools":
        cmd_tools(args, cli, config)
    elif args.command == "instance":
        cmd_instance(args, cli, config)
    elif args.command == "goal":
        cmd_goal(args, cli, config)
    elif args.command == "activate":
        cmd_activate(args, cli, config)
    else:
        cli.error(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
