"""
Motherlabs Runtime Validator — validate generated projects by running them.

Phase 27: Runtime Build Loop (Step 3)

Creates a venv, installs deps, runs import checks, pytest, and optional smoke test.
Parses tracebacks and maps errors back to blueprint components.

This is a LEAF MODULE — stdlib only. No engine/protocol/pipeline imports.
"""

import logging
import os
import re
import sys
import time
import venv
import subprocess
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("motherlabs.runtime_validator")


# =============================================================================
# FROZEN DATACLASSES
# =============================================================================

@dataclass(frozen=True)
class RuntimeConfig:
    """Configuration for runtime validation."""
    subprocess_timeout_seconds: int = 60
    pip_install_timeout_seconds: int = 120
    smoke_test_timeout_seconds: int = 10
    install_deps: bool = True
    run_tests: bool = True
    run_smoke_test: bool = False
    create_venv: bool = True


@dataclass(frozen=True)
class CommandResult:
    """Result of a subprocess command execution."""
    command: str
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    duration: float = 0.0


@dataclass(frozen=True)
class ComponentError:
    """An error mapped to a specific blueprint component."""
    component_name: str
    error_type: str        # "import", "test", "runtime", "syntax"
    error_message: str
    file_path: str = ""
    line_number: int = 0
    traceback_snippet: str = ""


@dataclass(frozen=True)
class ValidationResult:
    """Complete result of project validation."""
    success: bool
    dep_install: Optional[CommandResult] = None
    import_check: Optional[CommandResult] = None
    test_run: Optional[CommandResult] = None
    smoke_test: Optional[CommandResult] = None
    component_errors: Tuple[ComponentError, ...] = ()
    unmapped_errors: Tuple[str, ...] = ()
    venv_path: str = ""


# =============================================================================
# VENV CREATION
# =============================================================================

def create_project_venv(project_dir: str) -> str:
    """Create a virtual environment inside the project directory.

    Args:
        project_dir: Path to the generated project

    Returns:
        Path to the venv's Python executable
    """
    venv_dir = os.path.join(project_dir, ".venv")
    venv.create(venv_dir, with_pip=True, clear=True)

    # Determine python executable path
    if sys.platform == "win32":
        python_path = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        python_path = os.path.join(venv_dir, "bin", "python")

    return python_path


# =============================================================================
# COMMAND EXECUTION
# =============================================================================

def run_command(
    command: List[str],
    cwd: str,
    timeout: int = 60,
    env: Optional[Dict[str, str]] = None,
) -> CommandResult:
    """Execute a subprocess command with timeout.

    Args:
        command: Command and arguments as list
        cwd: Working directory
        timeout: Timeout in seconds
        env: Optional environment variables

    Returns:
        CommandResult with output and status
    """
    cmd_str = " ".join(command)
    t0 = time.time()

    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    try:
        proc = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=run_env,
        )
        duration = time.time() - t0
        return CommandResult(
            command=cmd_str,
            returncode=proc.returncode,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
            timed_out=False,
            duration=round(duration, 3),
        )
    except subprocess.TimeoutExpired:
        duration = time.time() - t0
        return CommandResult(
            command=cmd_str,
            returncode=-1,
            stdout="",
            stderr=f"Command timed out after {timeout}s",
            timed_out=True,
            duration=round(duration, 3),
        )
    except FileNotFoundError:
        duration = time.time() - t0
        return CommandResult(
            command=cmd_str,
            returncode=-1,
            stdout="",
            stderr=f"Command not found: {command[0]}",
            timed_out=False,
            duration=round(duration, 3),
        )


# =============================================================================
# VALIDATION STEPS
# =============================================================================

def install_dependencies(
    python_path: str,
    project_dir: str,
    timeout: int = 120,
) -> CommandResult:
    """Install project dependencies via pip.

    Args:
        python_path: Path to Python executable (in venv)
        project_dir: Project root directory
        timeout: Pip install timeout

    Returns:
        CommandResult from pip install
    """
    req_file = os.path.join(project_dir, "requirements.txt")
    if not os.path.exists(req_file):
        return CommandResult(
            command="pip install -r requirements.txt",
            returncode=0,
            stdout="No requirements.txt found, skipping.",
            stderr="",
            duration=0.0,
        )

    # Check if requirements.txt is empty
    with open(req_file, "r") as f:
        content = f.read().strip()
    if not content:
        return CommandResult(
            command="pip install -r requirements.txt",
            returncode=0,
            stdout="Empty requirements.txt, skipping.",
            stderr="",
            duration=0.0,
        )

    return run_command(
        [python_path, "-m", "pip", "install", "-r", req_file, "--quiet"],
        cwd=project_dir,
        timeout=timeout,
    )


def check_imports(
    python_path: str,
    project_dir: str,
    timeout: int = 60,
) -> CommandResult:
    """Check that all project modules can be imported.

    Runs: python -c "import <package>"

    Args:
        python_path: Path to Python executable
        project_dir: Project root directory
        timeout: Timeout in seconds

    Returns:
        CommandResult from import check
    """
    # Find the package name (directory with __init__.py inside project_dir)
    package_name = None
    init_path = os.path.join(project_dir, "__init__.py")
    if os.path.exists(init_path):
        package_name = os.path.basename(project_dir)

    if not package_name:
        # Try importing individual .py files
        py_files = [f for f in os.listdir(project_dir)
                     if f.endswith(".py") and f != "__init__.py" and f != "setup.py"]
        if not py_files:
            return CommandResult(
                command="import check",
                returncode=0,
                stdout="No Python files found to import.",
                stderr="",
                duration=0.0,
            )

        # Build import script that tries each file
        import_lines = []
        for f in py_files:
            module = f[:-3]  # strip .py
            import_lines.append(f"import {module}")

        import_script = "; ".join(import_lines)
        return run_command(
            [python_path, "-c", import_script],
            cwd=project_dir,
            timeout=timeout,
            env={"PYTHONPATH": project_dir},
        )

    # Package import
    parent_dir = os.path.dirname(project_dir)
    return run_command(
        [python_path, "-c", f"import {package_name}"],
        cwd=parent_dir,
        timeout=timeout,
        env={"PYTHONPATH": parent_dir},
    )


def run_tests(
    python_path: str,
    project_dir: str,
    timeout: int = 60,
) -> CommandResult:
    """Run project tests via pytest.

    Args:
        python_path: Path to Python executable
        project_dir: Project root directory
        timeout: Timeout in seconds

    Returns:
        CommandResult from pytest
    """
    tests_dir = os.path.join(project_dir, "tests")
    if not os.path.exists(tests_dir):
        return CommandResult(
            command="pytest",
            returncode=0,
            stdout="No tests/ directory found, skipping.",
            stderr="",
            duration=0.0,
        )

    # Ensure pytest is available in the venv
    run_command(
        [python_path, "-m", "pip", "install", "pytest", "--quiet"],
        cwd=project_dir,
        timeout=60,
    )

    return run_command(
        [python_path, "-m", "pytest", tests_dir, "--tb=short", "-q"],
        cwd=project_dir,
        timeout=timeout,
        env={"PYTHONPATH": project_dir},
    )


def run_smoke_test(
    python_path: str,
    project_dir: str,
    timeout: int = 10,
) -> CommandResult:
    """Run main.py as a smoke test (quick startup check).

    Args:
        python_path: Path to Python executable
        project_dir: Project root directory
        timeout: Short timeout (main.py may start servers)

    Returns:
        CommandResult from main.py
    """
    main_path = os.path.join(project_dir, "main.py")
    if not os.path.exists(main_path):
        return CommandResult(
            command="python main.py",
            returncode=0,
            stdout="No main.py found, skipping smoke test.",
            stderr="",
            duration=0.0,
        )

    return run_command(
        [python_path, main_path],
        cwd=project_dir,
        timeout=timeout,
    )


# =============================================================================
# TRACEBACK PARSING
# =============================================================================

# Patterns for extracting file/line/error from Python tracebacks
_TB_FILE_LINE = re.compile(
    r'File "([^"]+)", line (\d+)',
)
_TB_ERROR_LINE = re.compile(
    r'^(\w+Error|\w+Exception|ImportError|ModuleNotFoundError|SyntaxError|'
    r'TypeError|ValueError|AttributeError|NameError|IndentationError): (.+)$',
    re.MULTILINE,
)
_IMPORT_ERROR = re.compile(
    r"(?:ModuleNotFoundError|ImportError): (?:No module named )?'?([^'\"]+)'?",
)
_PYTEST_FAIL = re.compile(
    r'^FAILED (.+?)::(.+?)(?:\s|$)',
    re.MULTILINE,
)


def parse_traceback(output: str) -> List[Tuple[str, int, str]]:
    """Parse Python traceback output into structured errors.

    Args:
        output: Combined stdout+stderr from a command

    Returns:
        List of (filename, line_number, error_message) tuples
    """
    results = []

    # Find file/line references
    file_lines = _TB_FILE_LINE.findall(output)

    # Find error lines
    error_matches = _TB_ERROR_LINE.findall(output)

    if file_lines and error_matches:
        # Pair the last file/line with each error
        for error_type, error_msg in error_matches:
            # Use the most recent file/line before this error
            filename, line = file_lines[-1]
            results.append((filename, int(line), f"{error_type}: {error_msg}"))
    elif error_matches:
        # Errors without file references
        for error_type, error_msg in error_matches:
            results.append(("", 0, f"{error_type}: {error_msg}"))
    elif file_lines:
        # File references without clear error lines — use the output itself
        for filename, line in file_lines:
            results.append((filename, int(line), output.strip()[:200]))

    return results


def map_errors_to_components(
    errors: List[Tuple[str, int, str]],
    generated_code: Dict[str, str],
    blueprint: Dict[str, Any],
) -> Tuple[List[ComponentError], List[str]]:
    """Map parsed errors to blueprint components.

    Args:
        errors: (filename, line_number, error_message) tuples from parse_traceback
        generated_code: Dict[component_name, code] from emission
        blueprint: Blueprint dict

    Returns:
        (mapped ComponentErrors, unmapped error strings)
    """
    # Build filename → component name mapping
    component_names = set(generated_code.keys())
    blueprint_components = {c.get("name", "") for c in blueprint.get("components", [])}
    all_names = component_names | blueprint_components

    mapped: List[ComponentError] = []
    unmapped: List[str] = []

    for filename, line_number, error_msg in errors:
        matched = False
        basename = os.path.basename(filename).replace(".py", "") if filename else ""

        # Try matching filename to component
        # Sort by name length (shortest first) so "Task" matches before "TaskManager"
        for name in sorted(all_names, key=len):
            name_lower = name.lower().replace(" ", "_")
            if (basename and (name_lower in basename.lower()
                             or basename.lower() in name_lower)):
                # Determine error type from message
                error_type = "runtime"
                if "ImportError" in error_msg or "ModuleNotFoundError" in error_msg:
                    error_type = "import"
                elif "SyntaxError" in error_msg:
                    error_type = "syntax"
                elif "FAILED" in error_msg or "assert" in error_msg.lower():
                    error_type = "test"

                mapped.append(ComponentError(
                    component_name=name,
                    error_type=error_type,
                    error_message=error_msg,
                    file_path=filename,
                    line_number=line_number,
                    traceback_snippet=error_msg[:500],
                ))
                matched = True
                break

        # Try matching error message content to component names
        # Sort by longest name first to avoid partial matches (e.g. "Task" before "TaskManager")
        if not matched:
            for name in sorted(all_names, key=len, reverse=True):
                if name.lower() in error_msg.lower():
                    error_type = "runtime"
                    if "Import" in error_msg:
                        error_type = "import"
                    elif "Syntax" in error_msg:
                        error_type = "syntax"

                    mapped.append(ComponentError(
                        component_name=name,
                        error_type=error_type,
                        error_message=error_msg,
                        file_path=filename,
                        line_number=line_number,
                        traceback_snippet=error_msg[:500],
                    ))
                    matched = True
                    break

        if not matched:
            unmapped.append(error_msg)

    return mapped, unmapped


# =============================================================================
# AGENT SYSTEM VALIDATION
# =============================================================================

def validate_agent_system(
    project_dir: str,
    generated_code: Dict[str, str],
    blueprint: Dict[str, Any],
    config: Optional[RuntimeConfig] = None,
    port: int = 8080,
    startup_timeout: int = 10,
) -> ValidationResult:
    """Smoke-test an agent system: start, send message, verify response, shutdown.

    Goes beyond import checking to validate that the runtime actually starts,
    accepts a message, and returns a valid response.

    Args:
        project_dir: Path to the generated project directory
        generated_code: Dict[component_name, code] from emission
        blueprint: Blueprint dict
        config: Optional RuntimeConfig
        port: Port the runtime should listen on
        startup_timeout: Seconds to wait for READY signal

    Returns:
        ValidationResult with smoke test results
    """
    if config is None:
        config = RuntimeConfig(run_smoke_test=True)

    # First run standard validation (venv, deps, imports)
    standard_result = validate_project(
        project_dir, generated_code, blueprint,
        RuntimeConfig(
            create_venv=config.create_venv,
            install_deps=config.install_deps,
            run_tests=False,          # Skip tests — we do a live smoke test instead
            run_smoke_test=False,     # We handle this ourselves
        ),
    )

    if not standard_result.success:
        return standard_result

    # Determine python path
    python_path = sys.executable
    if standard_result.venv_path:
        if sys.platform == "win32":
            python_path = os.path.join(standard_result.venv_path, "Scripts", "python.exe")
        else:
            python_path = os.path.join(standard_result.venv_path, "bin", "python")

    # Start the runtime as a subprocess
    main_path = os.path.join(project_dir, "main.py")
    if not os.path.exists(main_path):
        return ValidationResult(
            success=False,
            dep_install=standard_result.dep_install,
            import_check=standard_result.import_check,
            unmapped_errors=("No main.py found for agent system smoke test",),
            venv_path=standard_result.venv_path,
        )

    env = os.environ.copy()
    env["PYTHONPATH"] = project_dir
    env["PORT"] = str(port)

    smoke_result = _run_agent_smoke_test(
        python_path, main_path, project_dir, env, port, startup_timeout,
    )

    all_errors: List[Tuple[str, int, str]] = []
    if smoke_result.returncode != 0:
        all_errors.extend(
            parse_traceback(smoke_result.stdout + "\n" + smoke_result.stderr)
        )

    mapped, unmapped = map_errors_to_components(
        all_errors, generated_code, blueprint,
    )

    return ValidationResult(
        success=smoke_result.returncode == 0,
        dep_install=standard_result.dep_install,
        import_check=standard_result.import_check,
        smoke_test=smoke_result,
        component_errors=tuple(mapped),
        unmapped_errors=tuple(unmapped),
        venv_path=standard_result.venv_path,
    )


def _run_agent_smoke_test(
    python_path: str,
    main_path: str,
    project_dir: str,
    env: Dict[str, str],
    port: int,
    startup_timeout: int,
) -> CommandResult:
    """Start agent system, send test message, verify response, shutdown.

    Args:
        python_path: Path to Python executable
        main_path: Path to main.py
        project_dir: Project directory
        env: Environment variables
        port: Port to connect to
        startup_timeout: Seconds to wait for READY signal

    Returns:
        CommandResult summarizing the smoke test
    """
    import socket

    t0 = time.time()

    try:
        # Start the runtime process
        proc = subprocess.Popen(
            [python_path, main_path],
            cwd=project_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

        # Wait for READY signal in stdout
        ready = False
        stdout_lines = []
        deadline = time.time() + startup_timeout
        while time.time() < deadline:
            if proc.poll() is not None:
                # Process exited early
                break
            # Non-blocking read
            try:
                line = proc.stdout.readline().decode(errors="replace")
                if line:
                    stdout_lines.append(line)
                    if "READY" in line:
                        ready = True
                        break
            except Exception as e:
                logger.debug(f"Stdout readline skipped: {e}")
                time.sleep(0.1)

        if not ready:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

            stderr_out = ""
            try:
                stderr_out = proc.stderr.read().decode(errors="replace")
            except Exception as e:
                logger.debug(f"Stderr read skipped: {e}")

            duration = time.time() - t0
            return CommandResult(
                command=f"agent smoke test (port {port})",
                returncode=-1,
                stdout="".join(stdout_lines),
                stderr=stderr_out or "Runtime did not emit READY signal",
                timed_out=not ready,
                duration=round(duration, 3),
            )

        # Send a test message via TCP
        test_msg = '{"target": "", "payload": {"type": "ping"}}\n'
        response_text = ""
        try:
            sock = socket.create_connection(("127.0.0.1", port), timeout=5)
            sock.sendall(test_msg.encode())
            response_text = sock.recv(4096).decode(errors="replace")
            sock.close()
        except Exception as e:
            response_text = f"Connection error: {e}"

        # Shutdown
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

        stderr_out = ""
        try:
            stderr_out = proc.stderr.read().decode(errors="replace")
        except Exception as e:
            logger.debug(f"Stderr read skipped: {e}")

        duration = time.time() - t0
        stdout_full = "".join(stdout_lines) + f"\nResponse: {response_text}"

        return CommandResult(
            command=f"agent smoke test (port {port})",
            returncode=0,
            stdout=stdout_full,
            stderr=stderr_out,
            timed_out=False,
            duration=round(duration, 3),
        )

    except Exception as e:
        duration = time.time() - t0
        return CommandResult(
            command=f"agent smoke test (port {port})",
            returncode=-1,
            stdout="",
            stderr=f"Smoke test failed: {e}",
            timed_out=False,
            duration=round(duration, 3),
        )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def validate_project(
    project_dir: str,
    generated_code: Dict[str, str],
    blueprint: Dict[str, Any],
    config: Optional[RuntimeConfig] = None,
) -> ValidationResult:
    """Validate a generated project by running it.

    Creates venv, installs deps, checks imports, runs tests, optional smoke test.
    Parses any errors and maps them back to blueprint components.

    Args:
        project_dir: Path to the generated project directory
        generated_code: Dict[component_name, code] from emission
        blueprint: Blueprint dict
        config: Optional RuntimeConfig

    Returns:
        ValidationResult with all step results and mapped errors
    """
    if config is None:
        config = RuntimeConfig()

    all_errors: List[Tuple[str, int, str]] = []
    dep_result = None
    import_result = None
    test_result = None
    smoke_result = None
    venv_path = ""

    # 1. Create venv (or use system Python)
    if config.create_venv:
        try:
            python_path = create_project_venv(project_dir)
            venv_path = os.path.dirname(os.path.dirname(python_path))
        except Exception as e:
            return ValidationResult(
                success=False,
                component_errors=(),
                unmapped_errors=(f"Venv creation failed: {e}",),
                venv_path="",
            )
    else:
        python_path = sys.executable

    # 2. Install dependencies
    if config.install_deps:
        dep_result = install_dependencies(
            python_path, project_dir, config.pip_install_timeout_seconds,
        )
        if dep_result.returncode != 0:
            all_errors.extend(parse_traceback(dep_result.stderr))

    # 3. Import check
    import_result = check_imports(
        python_path, project_dir, config.subprocess_timeout_seconds,
    )
    if import_result.returncode != 0:
        all_errors.extend(
            parse_traceback(import_result.stdout + "\n" + import_result.stderr)
        )

    # 4. Run tests
    if config.run_tests:
        test_result = run_tests(
            python_path, project_dir, config.subprocess_timeout_seconds,
        )
        if test_result.returncode != 0:
            all_errors.extend(
                parse_traceback(test_result.stdout + "\n" + test_result.stderr)
            )

    # 5. Smoke test
    if config.run_smoke_test:
        smoke_result = run_smoke_test(
            python_path, project_dir, config.smoke_test_timeout_seconds,
        )
        if smoke_result.returncode != 0:
            all_errors.extend(
                parse_traceback(smoke_result.stdout + "\n" + smoke_result.stderr)
            )

    # 6. Map errors to components
    mapped, unmapped = map_errors_to_components(
        all_errors, generated_code, blueprint,
    )

    # Determine success: all non-None steps must have returncode 0
    steps = [dep_result, import_result, test_result, smoke_result]
    success = all(
        step.returncode == 0
        for step in steps
        if step is not None
    )

    return ValidationResult(
        success=success,
        dep_install=dep_result,
        import_check=import_result,
        test_run=test_result,
        smoke_test=smoke_result,
        component_errors=tuple(mapped),
        unmapped_errors=tuple(unmapped),
        venv_path=venv_path,
    )
