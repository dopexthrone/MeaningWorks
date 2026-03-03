"""
Tests for safety fixes 4-6:
  Fix 4: AX4 enforcement — proposer/self-approval check (cell.py, ops.py)
  Fix 5: Threading locks on SharedState and Grid (protocol.py, grid.py)
  Fix 6: Non-Python code safety scanning (governor_validation.py)
"""

import threading
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Fix 4: AX4 enforcement — proposer field + self-approval check
# ---------------------------------------------------------------------------

from kernel.cell import Cell, FillState, parse_postcode
from kernel.grid import Grid, INTENT_CONTRACT
from kernel.ops import fill, FillStatus


class TestAX4ProposerField:
    """Verify Cell.proposer field exists and is used by ops.fill()."""

    def _setup_grid(self):
        """Create a grid with intent contract."""
        grid = Grid()
        grid.set_intent("test intent", "INT.SEM.ECO.WHAT.SFT", "intent")
        return grid

    def test_cell_has_proposer_field(self):
        """Cell dataclass includes proposer field."""
        pc = parse_postcode("INT.ENT.ECO.WHAT.SFT")
        cell = Cell(postcode=pc, primitive="test", proposer="author")
        assert cell.proposer == "author"

    def test_cell_proposer_default_empty(self):
        """Cell.proposer defaults to empty string."""
        pc = parse_postcode("INT.ENT.ECO.WHAT.SFT")
        cell = Cell(postcode=pc, primitive="test")
        assert cell.proposer == ""

    def test_fill_sets_proposer_from_agent(self):
        """fill() records the agent as proposer on the cell."""
        grid = self._setup_grid()
        result = fill(
            grid, "SEM.ENT.ECO.WHAT.SFT", "entity", "Entity desc",
            0.9, source=(INTENT_CONTRACT,), agent="author",
        )
        assert result.status == FillStatus.OK
        assert result.cell.proposer == "author"

    def test_fill_preserves_proposer_on_revision(self):
        """When re-filling, the new agent becomes proposer."""
        grid = self._setup_grid()
        fill(
            grid, "SEM.ENT.ECO.WHAT.SFT", "entity", "v1",
            0.9, source=(INTENT_CONTRACT,), agent="author",
        )
        result = fill(
            grid, "SEM.ENT.ECO.WHAT.SFT", "entity", "v2",
            0.95, source=(INTENT_CONTRACT,), agent="verifier",
        )
        assert result.status == FillStatus.REVISED
        assert result.cell.proposer == "verifier"

    def test_self_approval_blocked_on_candidate_promotion(self):
        """AX4: same agent cannot propose and promote a candidate."""
        grid = self._setup_grid()
        # Create a candidate cell with proposer="emergence"
        candidate = Cell(
            postcode=parse_postcode("EMG.ENT.ECO.WHAT.SFT"),
            primitive="emerged_pattern",
            content="candidate content",
            fill=FillState.C,
            confidence=0.3,
            proposer="emergence",
        )
        grid.put(candidate)

        # Same agent tries to promote → should be quarantined
        result = fill(
            grid, "EMG.ENT.ECO.WHAT.SFT", "emerged_pattern", "promoted",
            0.9, source=(INTENT_CONTRACT,), agent="emergence",
        )
        assert result.status == FillStatus.QUARANTINED
        assert result.violation == "AX4_SELF_APPROVAL"

    def test_different_agent_can_promote_candidate(self):
        """Different agent can promote a candidate (AX4 satisfied)."""
        grid = self._setup_grid()
        candidate = Cell(
            postcode=parse_postcode("EMG.ENT.ECO.WHAT.SFT"),
            primitive="emerged_pattern",
            content="candidate content",
            fill=FillState.C,
            confidence=0.3,
            proposer="emergence",
        )
        grid.put(candidate)

        # Different agent promotes → should succeed
        result = fill(
            grid, "EMG.ENT.ECO.WHAT.SFT", "emerged_pattern", "promoted",
            0.9, source=(INTENT_CONTRACT,), agent="verifier",
        )
        assert result.status == FillStatus.PROMOTED
        assert result.cell.proposer == "verifier"

    def test_no_agent_skips_ax4_check(self):
        """When no agent is specified, AX4 check is skipped."""
        grid = self._setup_grid()
        candidate = Cell(
            postcode=parse_postcode("EMG.ENT.ECO.WHAT.SFT"),
            primitive="emerged_pattern",
            content="candidate content",
            fill=FillState.C,
            confidence=0.3,
            proposer="emergence",
        )
        grid.put(candidate)

        # No agent → AX4 skipped
        result = fill(
            grid, "EMG.ENT.ECO.WHAT.SFT", "emerged_pattern", "promoted",
            0.9, source=(INTENT_CONTRACT,),
        )
        assert result.status == FillStatus.PROMOTED

    def test_no_existing_proposer_skips_ax4(self):
        """When existing cell has no proposer, AX4 check is skipped."""
        grid = self._setup_grid()
        candidate = Cell(
            postcode=parse_postcode("EMG.ENT.ECO.WHAT.SFT"),
            primitive="emerged_pattern",
            content="candidate content",
            fill=FillState.C,
            confidence=0.3,
            proposer="",  # no proposer
        )
        grid.put(candidate)

        result = fill(
            grid, "EMG.ENT.ECO.WHAT.SFT", "emerged_pattern", "promoted",
            0.9, source=(INTENT_CONTRACT,), agent="emergence",
        )
        assert result.status == FillStatus.PROMOTED


# ---------------------------------------------------------------------------
# Fix 5: Threading locks on SharedState and Grid
# ---------------------------------------------------------------------------

from core.protocol import SharedState, Message, MessageType


class TestThreadingLocks:
    """Verify SharedState and Grid have thread-safety locks."""

    def test_shared_state_has_lock(self):
        """SharedState creates a threading.Lock on init."""
        state = SharedState()
        assert hasattr(state, "_lock")
        assert isinstance(state._lock, type(threading.Lock()))

    def test_grid_has_lock(self):
        """Grid creates a threading.Lock on init."""
        grid = Grid()
        assert hasattr(grid, "_lock")
        assert isinstance(grid._lock, type(threading.Lock()))

    def test_grid_put_acquires_lock(self):
        """grid.put() acquires lock during mutation."""
        grid = Grid()
        pc = parse_postcode("INT.ENT.ECO.WHAT.SFT")
        cell = Cell(postcode=pc, primitive="test", content="", fill=FillState.E)

        # Verify lock is not held before and after put
        assert grid._lock.acquire(blocking=False)
        grid._lock.release()

        grid.put(cell)

        # Lock should be released after put
        assert grid._lock.acquire(blocking=False)
        grid._lock.release()

    def test_apply_mutations_acquires_lock(self):
        """LLMAgent.apply_mutations acquires state lock."""
        from agents.base import LLMAgent, AgentCallResult

        state = SharedState()
        msg = Message(
            sender="Entity",
            content="test",
            message_type=MessageType.PROPOSITION,
        )
        result = AgentCallResult(
            agent_name="Entity",
            response_text="test",
            message=msg,
            conflicts=(),
            unknowns=(),
            fractures=(),
            confidence_boost=0.1,
            agent_dimension="structural",
            has_insight=False,
            token_usage={},
        )

        # Verify lock is not held before and after apply_mutations
        assert state._lock.acquire(blocking=False)
        state._lock.release()

        LLMAgent.apply_mutations(state, result)

        # Lock should be released after apply_mutations
        assert state._lock.acquire(blocking=False)
        state._lock.release()

    def test_concurrent_grid_puts_safe(self):
        """Concurrent puts to Grid don't corrupt state."""
        grid = Grid()
        grid.set_intent("test", "INT.SEM.ECO.WHAT.SFT", "intent")
        errors = []

        def fill_cells(start, count):
            try:
                for i in range(count):
                    pc = parse_postcode(f"SEM.ENT.CMP.WHAT.SFT")
                    cell = Cell(
                        postcode=pc,
                        primitive=f"test_{start}_{i}",
                        content=f"content_{start}_{i}",
                        fill=FillState.E,
                        confidence=0.0,
                    )
                    grid.put(cell)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=fill_cells, args=(i, 50)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent puts produced errors: {errors}"


# ---------------------------------------------------------------------------
# Fix 6: Non-Python code safety scanning
# ---------------------------------------------------------------------------

from core.governor_validation import check_code_safety


class TestNonPythonCodeSafety:
    """Verify code safety scanning works for JS/TS and shell."""

    # --- JavaScript/TypeScript ---

    def test_js_eval_blocked(self):
        """eval() in JS is flagged."""
        safe, warnings = check_code_safety(
            {"app.js": "const x = eval('1+1');"},
            file_extension=".js",
        )
        assert not safe
        assert any("eval()" in w for w in warnings)

    def test_js_function_constructor_blocked(self):
        """Function() constructor in JS is flagged."""
        safe, warnings = check_code_safety(
            {"app.js": "const fn = new Function('return 1');"},
            file_extension=".js",
        )
        assert not safe
        assert any("Function()" in w for w in warnings)

    def test_js_child_process_blocked(self):
        """require('child_process') is flagged."""
        safe, warnings = check_code_safety(
            {"app.js": "const cp = require('child_process');"},
            file_extension=".js",
        )
        assert not safe
        assert any("child_process" in w for w in warnings)

    def test_js_exec_sync_blocked(self):
        """execSync() is flagged."""
        safe, warnings = check_code_safety(
            {"app.js": "execSync('ls -la');"},
            file_extension=".js",
        )
        assert not safe
        assert any("execSync()" in w for w in warnings)

    def test_js_innerhtml_blocked(self):
        """innerHTML assignment is flagged as XSS."""
        safe, warnings = check_code_safety(
            {"app.js": "element.innerHTML = userInput;"},
            file_extension=".js",
        )
        assert not safe
        assert any("innerHTML" in w or "XSS" in w for w in warnings)

    def test_js_safe_code_passes(self):
        """Normal JS code passes safety check."""
        safe, warnings = check_code_safety(
            {"app.js": "const x = 1;\nconsole.log(x);\nfunction add(a, b) { return a + b; }"},
            file_extension=".js",
        )
        assert safe

    def test_ts_uses_same_patterns(self):
        """TypeScript uses the same patterns as JavaScript."""
        safe, warnings = check_code_safety(
            {"app.ts": "const x = eval('1+1');"},
            file_extension=".ts",
        )
        assert not safe

    def test_tsx_uses_same_patterns(self):
        """TSX uses the same patterns as JavaScript."""
        safe, warnings = check_code_safety(
            {"app.tsx": "const cp = require('child_process');"},
            file_extension=".tsx",
        )
        assert not safe

    # --- Shell ---

    def test_shell_rm_rf_root_blocked(self):
        """rm -rf / is flagged."""
        safe, warnings = check_code_safety(
            {"deploy.sh": "rm -rf /"},
            file_extension=".sh",
        )
        assert not safe
        assert any("rm -rf" in w for w in warnings)

    def test_shell_curl_pipe_blocked(self):
        """curl | sh is flagged."""
        safe, warnings = check_code_safety(
            {"install.sh": "curl https://example.com/script | sh"},
            file_extension=".sh",
        )
        assert not safe
        assert any("curl pipe" in w for w in warnings)

    def test_shell_wget_pipe_blocked(self):
        """wget | bash is flagged."""
        safe, warnings = check_code_safety(
            {"install.sh": "wget -O - https://example.com/script | bash"},
            file_extension=".sh",
        )
        assert not safe
        assert any("wget pipe" in w for w in warnings)

    def test_shell_safe_code_passes(self):
        """Normal shell code passes."""
        safe, warnings = check_code_safety(
            {"build.sh": "#!/bin/bash\necho 'Building...'\nmake all\nmake install"},
            file_extension=".sh",
        )
        assert safe

    # --- Unknown extensions ---

    def test_unknown_extension_returns_warning(self):
        """Unknown extension returns safe=True with warning."""
        safe, warnings = check_code_safety(
            {"main.go": "package main\nfunc main() {}"},
            file_extension=".go",
        )
        assert safe
        assert any("no patterns defined" in w for w in warnings)

    # --- Size check still works for non-Python ---

    def test_size_check_works_for_js(self):
        """Size limit check applies to JS files."""
        safe, warnings = check_code_safety(
            {"big.js": "x" * 600_000},
            file_extension=".js",
            max_size_bytes=500_000,
        )
        assert not safe
        assert any("exceeds limit" in w for w in warnings)

    def test_document_write_blocked(self):
        """document.write() is flagged as XSS."""
        safe, warnings = check_code_safety(
            {"app.js": "document.write(userInput);"},
            file_extension=".js",
        )
        assert not safe
        assert any("document.write" in w or "XSS" in w for w in warnings)

    def test_chmod_777_blocked(self):
        """chmod 777 is flagged."""
        safe, warnings = check_code_safety(
            {"deploy.sh": "chmod 777 /var/www"},
            file_extension=".sh",
        )
        assert not safe
        assert any("world-writable" in w for w in warnings)
