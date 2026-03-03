"""
Motherlabs Agent Orchestrator — coordinates enrichment, compilation, and project writing.

Phase 3 of Agent Ship: The agent wraps the compiler. It doesn't replace it.

NOT a leaf module. Imports engine, input_enrichment, project_writer, input_quality.
"""

import os
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Callable

from core.input_quality import InputQualityAnalyzer
from core.input_enrichment import (
    EnrichmentResult,
    ENRICHMENT_SYSTEM_PROMPT,
    build_enrichment_prompt,
    parse_enrichment_response,
)
from core.project_writer import (
    ProjectConfig,
    ProjectManifest,
    write_project,
    validate_all_code,
)
from core.build_loop import (
    BuildConfig,
    BuildResult,
    BuildIteration,
    FixAttempt,
    build_fix_prompt,
    identify_components_to_fix,
    apply_fix_to_project,
    serialize_build_result,
)
from core.runtime_validator import (
    RuntimeConfig,
    ValidationResult,
    validate_project,
)
from core.agent_emission import extract_code_from_response

logger = logging.getLogger("motherlabs.agent")


# =============================================================================
# FROZEN DATACLASSES
# =============================================================================

@dataclass(frozen=True)
class AgentConfig:
    """Configuration for the agent orchestrator."""
    codegen_mode: str = "llm"           # "llm" or "template"
    enrich_input: bool = True
    write_project: bool = True
    output_dir: str = "./output"
    language: str = "python"
    build: bool = False                  # Phase 27: Enable runtime build loop


@dataclass(frozen=True)
class AgentResult:
    """Complete result from an agent run."""
    success: bool
    project_manifest: Optional[ProjectManifest] = None
    blueprint: Dict[str, Any] = field(default_factory=dict)
    generated_code: Dict[str, str] = field(default_factory=dict)
    enrichment: Optional[EnrichmentResult] = None
    compile_result: Any = None          # CompileResult from engine
    quality_score: float = 0.0
    error: Optional[str] = None
    timing: Dict[str, float] = field(default_factory=dict)
    build_result: Optional[BuildResult] = None  # Phase 27
    trust: Optional[Any] = None         # TrustIndicators from core.trust


# =============================================================================
# SERIALIZATION
# =============================================================================

def serialize_agent_result(result: AgentResult) -> dict:
    """Serialize AgentResult to JSON-safe dict for API/IDE consumption."""
    d = {
        "success": result.success,
        "blueprint": result.blueprint,
        "generated_code": dict(result.generated_code),
        "quality_score": result.quality_score,
        "error": result.error,
        "timing": dict(result.timing),
    }

    if result.project_manifest:
        d["project_manifest"] = {
            "project_dir": result.project_manifest.project_dir,
            "files_written": list(result.project_manifest.files_written),
            "entry_point": result.project_manifest.entry_point,
            "total_lines": result.project_manifest.total_lines,
            "file_contents": dict(result.project_manifest.file_contents),
        }
    else:
        d["project_manifest"] = None

    if result.enrichment:
        d["enrichment"] = {
            "original_input": result.enrichment.original_input,
            "enriched_input": result.enrichment.enriched_input,
            "expansion_ratio": result.enrichment.expansion_ratio,
        }
    else:
        d["enrichment"] = None

    if result.build_result:
        d["build_result"] = serialize_build_result(result.build_result)
    else:
        d["build_result"] = None

    if result.trust:
        t = result.trust
        d["trust"] = {
            "overall_score": t.overall_score,
            "provenance_depth": t.provenance_depth,
            "fidelity_scores": dict(t.fidelity_scores),
            "gap_report": list(t.gap_report),
            "dimensional_coverage": dict(t.dimensional_coverage),
            "verification_badge": t.verification_badge,
            "confidence_trajectory": list(t.confidence_trajectory),
            "silence_zones": list(t.silence_zones),
            "derivation_chain_length": t.derivation_chain_length,
            "component_count": t.component_count,
            "relationship_count": t.relationship_count,
            "constraint_count": t.constraint_count,
        }
    else:
        d["trust"] = None

    return d


# =============================================================================
# ORCHESTRATOR
# =============================================================================

class AgentOrchestrator:
    """Coordinates enrichment → compilation → code emission → project writing.

    The agent wraps the compiler. It doesn't replace it.
    """

    def __init__(self, engine, config: Optional[AgentConfig] = None):
        """Initialize orchestrator with engine and config.

        Args:
            engine: MotherlabsEngine instance (with llm client)
            config: Optional AgentConfig (defaults used if None)
        """
        self.engine = engine
        self.config = config or AgentConfig()

    def run(
        self,
        description: str,
        on_progress: Optional[Callable[[str, str], None]] = None,
    ) -> AgentResult:
        """Run the full agent pipeline: enrich → compile → emit → write.

        Args:
            description: User's natural language description
            on_progress: Optional callback(phase_name, message)

        Returns:
            AgentResult with all artifacts
        """
        timing: Dict[str, float] = {}
        enrichment_result = None

        def _progress(phase: str, msg: str):
            if on_progress:
                on_progress(phase, msg)

        # 1. Quality gate
        _progress("quality", "Analyzing input quality...")
        t0 = time.time()

        analyzer = InputQualityAnalyzer()
        quality = analyzer.analyze(description)
        timing["quality"] = round(time.time() - t0, 3)

        if not quality.is_acceptable:
            return AgentResult(
                success=False,
                quality_score=quality.overall,
                error=f"Input quality too low ({quality.overall:.0%}). {quality.suggestion}",
                timing=timing,
            )

        _progress("quality", f"Quality: {quality.overall:.0%}")

        # 2. Enrich if hollow
        effective_description = description
        if self.config.enrich_input and quality.is_hollow:
            _progress("enrich", "Enriching sparse input...")
            t0 = time.time()
            try:
                enrichment_result = self._enrich(description)
                effective_description = enrichment_result.enriched_input
                _progress("enrich", f"Enriched (x{enrichment_result.expansion_ratio:.1f})")
            except Exception as e:
                logger.warning(f"Enrichment failed, using original input: {e}")
                effective_description = description
            timing["enrich"] = round(time.time() - t0, 3)

        # 3. Compile (with retry on transient provider errors)
        _progress("compile", "Compiling blueprint...")
        t0 = time.time()
        compile_result = None
        for compile_attempt in range(2):  # max 2 attempts
            try:
                compile_result = self.engine.compile(
                    effective_description,
                    use_corpus_suggestions=True,
                )
                break
            except Exception as e:
                # Retry once on transient provider/timeout errors
                from core.exceptions import ProviderError, TimeoutError as MotherlabsTimeout
                if compile_attempt == 0 and isinstance(e, (ProviderError, MotherlabsTimeout)):
                    logger.warning(f"Transient error during compilation: {e}. Retrying...")
                    _progress("compile", "Retrying after transient error...")
                    time.sleep(5.0)
                    continue
                timing["compile"] = round(time.time() - t0, 3)
                return AgentResult(
                    success=False,
                    quality_score=quality.overall,
                    enrichment=enrichment_result,
                    error=f"Compilation failed: {e}",
                    timing=timing,
                )
        timing["compile"] = round(time.time() - t0, 3)

        if not compile_result.success:
            return AgentResult(
                success=False,
                blueprint=compile_result.blueprint,
                quality_score=quality.overall,
                enrichment=enrichment_result,
                compile_result=compile_result,
                error=compile_result.error or "Compilation failed",
                timing=timing,
            )

        blueprint = compile_result.blueprint
        components = blueprint.get("components", [])
        _progress("compile", f"{len(components)} components · {len(blueprint.get('relationships', []))} relationships")

        # 3b. Compute trust indicators
        trust_indicators = None
        try:
            from core.trust import compute_trust_indicators
            intent_kw = list(compile_result.context_graph.get("keywords", [])) if compile_result.context_graph else []
            trust_indicators = compute_trust_indicators(
                blueprint=blueprint,
                verification=compile_result.verification or {},
                context_graph=compile_result.context_graph or {},
                dimensional_metadata=compile_result.dimensional_metadata or {},
                intent_keywords=intent_kw,
            )
        except Exception as e:
            logger.debug(f"Trust indicator computation skipped: {e}")

        # 4. Emit code
        _progress("emit", "Generating code...")
        t0 = time.time()
        generated_code: Dict[str, str] = {}
        try:
            generated_code = self._emit_code(blueprint, compile_result)
        except Exception as e:
            logger.warning(f"Code emission failed: {e}")
            timing["emit"] = round(time.time() - t0, 3)
            return AgentResult(
                success=False,
                blueprint=blueprint,
                quality_score=quality.overall,
                enrichment=enrichment_result,
                compile_result=compile_result,
                error=f"Code emission failed: {e}",
                timing=timing,
            )
        timing["emit"] = round(time.time() - t0, 3)

        total_lines = sum(code.count('\n') + 1 for code in generated_code.values())
        _progress("emit", f"{len(generated_code)} components emitted ({total_lines:,} lines)")

        # 4b. Post-emission syntax repair
        if generated_code:
            generated_code = self._repair_syntax_errors(
                generated_code, blueprint, _progress,
            )

        # 4c. Code safety scan before writing to disk
        if generated_code:
            try:
                from core.governor_validation import check_code_safety
                _file_ext = ".py"
                if self.engine.domain_adapter:
                    _file_ext = self.engine.domain_adapter.materialization.file_extension
                safe, safety_warnings = check_code_safety(
                    generated_code, file_extension=_file_ext,
                )
                if safety_warnings:
                    for sw in safety_warnings:
                        logger.warning(f"Code safety: {sw}")
                if not safe:
                    _progress("safety", f"Code safety check failed: {len(safety_warnings)} issue(s)")
                    return AgentResult(
                        success=False,
                        blueprint=blueprint,
                        quality_score=quality.overall,
                        enrichment=enrichment_result,
                        compile_result=compile_result,
                        error=f"Code safety check failed: {'; '.join(safety_warnings[:3])}",
                        timing=timing,
                    )
            except Exception as e:
                # Fail-closed: if safety check itself fails, do not write to disk
                logger.error(f"Code safety check unavailable: {e}")
                return AgentResult(
                    success=False,
                    blueprint=blueprint,
                    quality_score=quality.overall,
                    enrichment=enrichment_result,
                    compile_result=compile_result,
                    error=f"Code safety check unavailable: {e}",
                    timing=timing,
                )

        # 5. Write project
        manifest = None
        if self.config.write_project and generated_code:
            _progress("write", "Writing project...")
            t0 = time.time()
            try:
                _runtime_cap = None
                _ent_types = None
                _file_ext = ".py"
                if self.engine.domain_adapter:
                    _ent_types = self.engine.domain_adapter.vocabulary.entity_types
                    _file_ext = self.engine.domain_adapter.materialization.file_extension
                    _runtime_cap = getattr(self.engine.domain_adapter, 'runtime', None)
                project_config = ProjectConfig(
                    language=self.config.language,
                    runtime_capabilities=_runtime_cap,
                )
                manifest = write_project(
                    generated_code, blueprint,
                    self.config.output_dir, project_config,
                    entity_types=_ent_types,
                    file_extension=_file_ext,
                )
                _progress("write", f"Written to {manifest.project_dir}")
            except Exception as e:
                logger.warning(f"Project writing failed: {e}")
            timing["write"] = round(time.time() - t0, 3)

        # 6. Build loop (Phase 27)
        build_result = None
        if self.config.build and manifest and generated_code:
            _progress("build", "Validating and fixing project...")
            t0 = time.time()
            try:
                build_result = self._build(
                    manifest.project_dir, generated_code, blueprint,
                    _progress,
                )
                if build_result.final_manifest:
                    manifest = build_result.final_manifest
                if build_result.final_code:
                    generated_code = build_result.final_code
                if build_result.success:
                    _progress("build", f"Build passed ({len(build_result.iterations)} iterations)")
                else:
                    unfixed = build_result.components_unfixed
                    _progress("build", f"Build incomplete: {len(unfixed)} components unfixed")
            except Exception as e:
                logger.warning(f"Build loop failed: {e}")
                _progress("build", f"Build loop error: {e}")
            timing["build"] = round(time.time() - t0, 3)

        # 7. Return result
        return AgentResult(
            success=True,
            project_manifest=manifest,
            blueprint=blueprint,
            generated_code=generated_code,
            enrichment=enrichment_result,
            compile_result=compile_result,
            quality_score=quality.overall,
            timing=timing,
            build_result=build_result,
            trust=trust_indicators,
        )

    def _enrich(self, description: str) -> EnrichmentResult:
        """Run input enrichment via LLM."""
        prompt = build_enrichment_prompt(description)
        response = self.engine.llm.complete_with_system(
            system_prompt=ENRICHMENT_SYSTEM_PROMPT,
            user_prompt=prompt,
        )
        return parse_enrichment_response(response, description)

    def _build(
        self,
        project_dir: str,
        generated_code: Dict[str, str],
        blueprint: Dict[str, Any],
        _progress: Callable[[str, str], None],
    ) -> BuildResult:
        """Run the build loop: validate → fix → re-validate → repeat.

        Args:
            project_dir: Path to the written project
            generated_code: Current generated code dict
            blueprint: Blueprint dict
            _progress: Progress callback

        Returns:
            BuildResult with iteration history
        """
        from core.protocol_spec import PROTOCOL

        build_spec = PROTOCOL.build
        runtime_config = RuntimeConfig(
            subprocess_timeout_seconds=build_spec.subprocess_timeout_seconds,
            pip_install_timeout_seconds=build_spec.pip_install_timeout_seconds,
            smoke_test_timeout_seconds=build_spec.smoke_test_timeout_seconds,
            create_venv=build_spec.create_venv,
        )
        build_config = BuildConfig(
            max_iterations=build_spec.max_iterations,
            max_fixes_per_component=build_spec.max_fixes_per_component,
            runtime_config=runtime_config,
        )

        current_code = dict(generated_code)
        iterations: List[BuildIteration] = []
        fix_history: Dict[str, int] = {}  # component_name → fix attempt count
        all_fixed: set = set()
        total_attempts = 0

        for i in range(build_config.max_iterations):
            _progress("build", f"Iteration {i + 1}/{build_config.max_iterations}: validating...")

            # Validate
            validation = validate_project(
                project_dir, current_code, blueprint, runtime_config,
            )

            if validation.success:
                iterations.append(BuildIteration(
                    iteration=i + 1,
                    validation=validation,
                ))
                break

            # Identify components to fix
            to_fix = identify_components_to_fix(
                validation, fix_history, build_config.max_fixes_per_component,
            )

            if not to_fix:
                # All errors are unmapped or exceeded max fixes
                iterations.append(BuildIteration(
                    iteration=i + 1,
                    validation=validation,
                ))
                break

            # Fix each component
            fix_attempts: List[FixAttempt] = []
            fixed_this_iter: List[str] = []
            broken_this_iter: List[str] = []

            for error in to_fix:
                name = error.component_name
                code = current_code.get(name, "")
                if not code:
                    continue

                _progress("build", f"Fixing {name} ({error.error_type}: {error.error_message[:60]})")

                prompt = build_fix_prompt(name, error, code, blueprint, generated_code=current_code)

                try:
                    response = self.engine.llm.complete_with_system(
                        system_prompt="You are a code repair agent. Fix the error in the given Python code. Return ONLY the corrected Python code in a ```python block.",
                        user_content=prompt,
                    )
                    fixed_code = extract_code_from_response(response)

                    if fixed_code and fixed_code != code:
                        _ent_types = None
                        _file_ext = ".py"
                        if self.engine.domain_adapter:
                            _ent_types = self.engine.domain_adapter.vocabulary.entity_types
                            _file_ext = self.engine.domain_adapter.materialization.file_extension
                        current_code, manifest = apply_fix_to_project(
                            project_dir, name, fixed_code, current_code, blueprint,
                            entity_types=_ent_types,
                            file_extension=_file_ext,
                        )
                        fix_attempts.append(FixAttempt(
                            component_name=name,
                            iteration=i + 1,
                            error=error.error_message,
                            prompt=prompt[:500],
                            original_code=code[:500],
                            fixed_code=fixed_code[:500],
                            succeeded=True,
                        ))
                        fixed_this_iter.append(name)
                        all_fixed.add(name)
                    else:
                        fix_attempts.append(FixAttempt(
                            component_name=name,
                            iteration=i + 1,
                            error=error.error_message,
                            prompt=prompt[:500],
                            original_code=code[:500],
                            fixed_code=fixed_code[:500] if fixed_code else "",
                            succeeded=False,
                        ))
                        broken_this_iter.append(name)
                except Exception as e:
                    logger.warning(f"Fix attempt failed for {name}: {e}")
                    fix_attempts.append(FixAttempt(
                        component_name=name,
                        iteration=i + 1,
                        error=error.error_message,
                        prompt=prompt[:500],
                        original_code=code[:500],
                        fixed_code="",
                        succeeded=False,
                    ))
                    broken_this_iter.append(name)

                fix_history[name] = fix_history.get(name, 0) + 1
                total_attempts += 1

            iterations.append(BuildIteration(
                iteration=i + 1,
                validation=validation,
                fixes_attempted=tuple(fix_attempts),
                components_fixed=tuple(fixed_this_iter),
                components_still_broken=tuple(broken_this_iter),
            ))

        # Determine final state
        final_validation = iterations[-1].validation if iterations else None
        success = final_validation.success if final_validation else False

        # Components that were ever broken and never fixed
        all_broken = set()
        for it in iterations:
            for e in it.validation.component_errors:
                all_broken.add(e.component_name)
        unfixed = all_broken - all_fixed

        # Get final manifest
        final_manifest = None
        if iterations:
            # Re-read manifest from disk state
            from core.project_writer import ProjectConfig
            parent = os.path.dirname(project_dir)
            pname = os.path.basename(project_dir)
            try:
                _ent_types = None
                _file_ext = ".py"
                _runtime_cap = None
                if self.engine.domain_adapter:
                    _ent_types = self.engine.domain_adapter.vocabulary.entity_types
                    _file_ext = self.engine.domain_adapter.materialization.file_extension
                    _runtime_cap = getattr(self.engine.domain_adapter, 'runtime', None)
                final_manifest = write_project(
                    current_code, blueprint, parent,
                    ProjectConfig(project_name=pname, runtime_capabilities=_runtime_cap),
                    entity_types=_ent_types,
                    file_extension=_file_ext,
                )
            except Exception as e:
                logger.debug(f"Project write skipped: {e}")

        return BuildResult(
            success=success,
            iterations=tuple(iterations),
            final_code=current_code,
            final_manifest=final_manifest,
            components_fixed=tuple(sorted(all_fixed)),
            components_unfixed=tuple(sorted(unfixed)),
            total_fix_attempts=total_attempts,
        )

    def _repair_syntax_errors(
        self,
        generated_code: Dict[str, str],
        blueprint: Dict[str, Any],
        _progress: Callable[[str, str], None],
        max_repair_attempts: int = 2,
    ) -> Dict[str, str]:
        """Validate emitted code syntax and re-emit broken components.

        Runs validate_all_code on the emitted code dict. For each component
        with a syntax error, asks the LLM to fix the code (up to max_repair_attempts).
        Returns the repaired code dict.

        Args:
            generated_code: Component name -> emitted code
            blueprint: Blueprint dict (for context in repair prompts)
            _progress: Progress callback
            max_repair_attempts: Max repair passes per component
        """
        file_ext = ".py"
        if self.engine.domain_adapter:
            file_ext = self.engine.domain_adapter.materialization.file_extension

        errors = validate_all_code(generated_code, file_extension=file_ext)
        if not errors:
            return generated_code

        repaired = dict(generated_code)
        repair_count = 0
        repair_failures = 0

        for _pass in range(max_repair_attempts):
            errors = validate_all_code(repaired, file_extension=file_ext)
            if not errors:
                break

            for err in errors:
                # Parse "ComponentName.py:LINE: MSG" format
                parts = err.split(":", 2)
                if len(parts) < 3:
                    continue
                comp_file = parts[0].strip()
                comp_name = comp_file.replace(file_ext, "")
                if comp_name not in repaired:
                    continue

                line_num = parts[1].strip()
                err_msg = parts[2].strip()
                broken_code = repaired[comp_name]

                _progress("repair", f"Repairing {comp_name} (syntax error line {line_num})...")
                logger.info(f"Syntax repair: {comp_name} — {err_msg}")

                # Build repair prompt with broken code + error
                repair_prompt = (
                    f"The following code for component '{comp_name}' has a syntax error:\n"
                    f"  Line {line_num}: {err_msg}\n\n"
                    f"```\n{broken_code}\n```\n\n"
                    f"Fix the syntax error and return the complete corrected code. "
                    f"Do NOT change the logic or add new features — only fix the syntax."
                )

                preamble = ""
                if self.engine.domain_adapter and self.engine.domain_adapter.prompts.emission_preamble:
                    preamble = self.engine.domain_adapter.prompts.emission_preamble

                system_prompt = preamble or "Fix the syntax error in the code. Return only the corrected code."

                try:
                    response = self.engine.llm.complete_with_system(
                        system_prompt=system_prompt,
                        user_content=repair_prompt,
                        max_tokens=16384,
                        temperature=0.0,
                    )
                    fixed_code = extract_code_from_response(response)
                    if fixed_code.strip():
                        repaired[comp_name] = fixed_code
                        repair_count += 1
                    else:
                        repair_failures += 1
                except Exception as e:
                    logger.warning(f"Syntax repair failed for {comp_name}: {e}")
                    repair_failures += 1

        # Final validation report
        final_errors = validate_all_code(repaired, file_extension=file_ext)
        if repair_count > 0 or repair_failures > 0:
            logger.info(
                f"Syntax repair: {repair_count} fixed, {repair_failures} failed, "
                f"{len(final_errors)} remaining errors"
            )
        if repair_count > 0:
            _progress("repair", f"{repair_count} components repaired")

        return repaired

    def _emit_code(
        self,
        blueprint: Dict[str, Any],
        compile_result,
    ) -> Dict[str, str]:
        """Emit code using configured mode."""
        if self.config.codegen_mode == "llm":
            # LLM emission — returns EmissionResult with generated_code dict
            emission_result = self.engine.emit_code(blueprint, layered=True)
            return dict(emission_result.generated_code)
        else:
            # Template codegen — returns single string, split by class
            from codegen.generator import BlueprintCodeGenerator
            gen = BlueprintCodeGenerator(blueprint)
            full_code = gen.generate()
            if not full_code.strip():
                logger.warning("E5001: Template code generation produced empty output")
            return _split_template_output(full_code, blueprint)


def _split_template_output(
    full_code: str,
    blueprint: Dict[str, Any],
) -> Dict[str, str]:
    """Split template codegen output (one big string) into per-component dict.

    Scans for class definitions that match blueprint component names.
    """
    import re

    components = {c["name"] for c in blueprint.get("components", [])}
    result: Dict[str, str] = {}

    # Find all class definitions
    class_pattern = re.compile(r'^class\s+(\w+)', re.MULTILINE)
    matches = list(class_pattern.finditer(full_code))

    for i, match in enumerate(matches):
        class_name = match.group(1)
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_code)
        class_code = full_code[start:end].rstrip()

        # Only include blueprint components (skip boilerplate like BaseAgent)
        if class_name in components:
            result[class_name] = class_code

    # If no matches, return the whole thing under a generic key
    if not result and full_code.strip():
        result["generated"] = full_code

    return result
