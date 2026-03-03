"""
Code Comparison Tool for Motherlabs Self-Improvement Loop.

Phase 4.2: Compare generated code against actual code to measure
self-rebuild capability.
"""
import ast
from typing import Dict, List, Any, Optional, Tuple


class CodeComparisonTool:
    """Tool to compare generated code against actual code."""

    def __init__(self, actual_code: str, generated_code: str):
        """
        Initialize comparison tool.

        Args:
            actual_code: The actual source code to compare against
            generated_code: The generated code to evaluate
        """
        self.actual_code = actual_code
        self.generated_code = generated_code
        self.actual_tree = ast.parse(actual_code)
        try:
            self.generated_tree = ast.parse(generated_code)
            self.syntax_valid = True
        except SyntaxError:
            self.generated_tree = None
            self.syntax_valid = False

    def extract_class(self, tree: ast.AST, class_name: str) -> Optional[ast.ClassDef]:
        """Extract a class definition by name (case-insensitive)."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name.lower() == class_name.lower():
                return node
        return None

    def analyze_class(self, class_def: Optional[ast.ClassDef]) -> Dict[str, Any]:
        """
        Extract structural info from a class.

        Returns dict with:
        - fields: List of field definitions
        - methods: List of method definitions
        - decorators: List of decorator names
        - bases: List of base class names
        """
        if not class_def:
            return {'fields': [], 'methods': [], 'decorators': [], 'bases': []}

        fields = []
        methods = []

        for item in class_def.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                field_name = item.target.id
                field_type = ast.unparse(item.annotation) if item.annotation else "Any"
                default = ast.unparse(item.value) if item.value else None
                fields.append({
                    'name': field_name,
                    'type': field_type,
                    'default': default
                })
            elif isinstance(item, ast.FunctionDef):
                params = []
                for arg in item.args.args:
                    if arg.arg != 'self':
                        params.append({
                            'name': arg.arg,
                            'type': ast.unparse(arg.annotation) if arg.annotation else None
                        })
                methods.append({
                    'name': item.name,
                    'params': params,
                    'return_type': ast.unparse(item.returns) if item.returns else None
                })

        decorators = [ast.unparse(d) for d in class_def.decorator_list]
        bases = [ast.unparse(b) for b in class_def.bases]

        return {
            'fields': fields,
            'methods': methods,
            'decorators': decorators,
            'bases': bases
        }

    def compare_classes(self, class_name: str) -> Dict[str, Any]:
        """
        Compare a specific class between actual and generated code.

        Returns dict with:
        - class_found: Whether class exists in generated code
        - actual: Structural info from actual class
        - generated: Structural info from generated class
        - field_score: 0-1 score for field matching
        - method_score: 0-1 score for method matching
        - type_score: 0-1 score for type correctness
        - overall_score: Average of the three scores
        - missing_fields: Fields in actual but not generated
        - missing_methods: Methods in actual but not generated
        - extra_fields: Fields in generated but not actual
        - extra_methods: Methods in generated but not actual
        """
        actual_class = self.extract_class(self.actual_tree, class_name)
        generated_class = self.extract_class(self.generated_tree, class_name) if self.generated_tree else None

        actual_info = self.analyze_class(actual_class)
        generated_info = self.analyze_class(generated_class)

        # Field comparison
        actual_field_names = {f['name'] for f in actual_info['fields']}
        generated_field_names = {f['name'] for f in generated_info['fields']}
        field_overlap = actual_field_names & generated_field_names

        # Method comparison
        actual_method_names = {m['name'] for m in actual_info['methods']}
        generated_method_names = {m['name'] for m in generated_info['methods']}
        method_overlap = actual_method_names & generated_method_names

        # Score calculation
        field_score = len(field_overlap) / len(actual_field_names) if actual_field_names else 1.0
        method_score = len(method_overlap) / len(actual_method_names) if actual_method_names else 1.0

        # Type correctness
        type_correct = 0
        type_total = 0
        for field in actual_info['fields']:
            if field['name'] in generated_field_names:
                type_total += 1
                gen_field = next((f for f in generated_info['fields'] if f['name'] == field['name']), None)
                if gen_field and gen_field['type'] == field['type']:
                    type_correct += 1
        type_score = type_correct / type_total if type_total else 1.0

        return {
            'class_found': generated_class is not None,
            'actual': actual_info,
            'generated': generated_info,
            'field_score': field_score,
            'method_score': method_score,
            'type_score': type_score,
            'overall_score': (field_score + method_score + type_score) / 3,
            'missing_fields': actual_field_names - generated_field_names,
            'missing_methods': actual_method_names - generated_method_names,
            'extra_fields': generated_field_names - actual_field_names,
            'extra_methods': generated_method_names - actual_method_names
        }

    def compare_all_classes(self) -> Dict[str, Dict[str, Any]]:
        """Compare all classes found in actual code."""
        results = {}
        for node in ast.walk(self.actual_tree):
            if isinstance(node, ast.ClassDef):
                results[node.name] = self.compare_classes(node.name)
        return results

    def get_summary(self) -> Dict[str, Any]:
        """Get overall comparison summary."""
        if not self.syntax_valid:
            return {
                'syntax_valid': False,
                'overall_score': 0,
                'class_scores': {}
            }

        class_results = self.compare_all_classes()

        if not class_results:
            return {
                'syntax_valid': True,
                'overall_score': 0,
                'class_scores': {}
            }

        # Calculate overall score
        scores = [r['overall_score'] for r in class_results.values() if r['class_found']]
        overall = sum(scores) / len(scores) if scores else 0

        return {
            'syntax_valid': True,
            'overall_score': overall,
            'classes_found': sum(1 for r in class_results.values() if r['class_found']),
            'classes_total': len(class_results),
            'class_scores': {name: r['overall_score'] for name, r in class_results.items()}
        }


def compare_blueprint_output(actual_file: str, blueprint_file: str) -> Dict[str, Any]:
    """
    Convenience function to compare a blueprint's generated code against actual code.

    Args:
        actual_file: Path to actual source file
        blueprint_file: Path to blueprint JSON file

    Returns:
        Comparison summary
    """
    import json
    from codegen.generator import BlueprintCodeGenerator

    with open(actual_file, 'r') as f:
        actual_code = f.read()

    with open(blueprint_file, 'r') as f:
        blueprint = json.load(f)

    gen = BlueprintCodeGenerator(blueprint)
    generated_code = gen.generate()

    tool = CodeComparisonTool(actual_code, generated_code)
    return tool.get_summary()
