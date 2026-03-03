"""
CodeGen Module: Transform blueprints into Python code stubs.

Stage 1: Generate dataclass/class stubs from components
Stage 2: Generate method signatures from relationships
Stage 3: Generate validation logic from constraints
"""

from .generator import BlueprintCodeGenerator

__all__ = ['BlueprintCodeGenerator']
