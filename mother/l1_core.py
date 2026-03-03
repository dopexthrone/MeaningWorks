import re
from difflib import SequenceMatcher

def track_losses(input_text: str, blueprint: dict) -> list[tuple[str, float]]:
    """
    Compute compression losses between input description and blueprint.
    Returns list of (category, severity 0.0-1.0).
    """
    losses = []

    # Entity compression (strengthened with fuzzy matching)
    pattern = r'\b[A-Z][a-z]*(?:[A-Z][a-z]*)*\b'
    raw_entities = re.findall(pattern, input_text)
    action_words = {'implement', 'handle', 'add', 'create', 'build', 'use', 'test', 'make', 'develop', 'design', 'fix', 'improve', 'reduce', 'optimize', 'deploy', 'ensure', 'require', 'validate', 'check'}
    input_entities = set()
    for w in raw_entities:
        # split camelcase
        camel_words = re.sub(r'([a-z])([A-Z])', r'\1 \2', w).split()
        for word in camel_words:
            lw = word.lower()
            if lw not in action_words:
                input_entities.add(lw)
    components = blueprint.get('components', [])
    bp_entities = set()
    for c in components:
        # name dash/underscore split
        name = c.get('name', '')
        parts = name.lower().replace('-', ' ').replace('_', ' ').replace(':', ' ').split()
        bp_entities.update(p.strip() for p in parts if p.strip())
        # camel split name
        camel_name = re.sub(r'([a-z])([A-Z])', r'\\1 \\2', name.lower()).split()
        bp_entities.update(p.strip() for p in camel_name if p.strip())
        # description
        desc = c.get('description', '')
        raw_entities = re.findall(pattern, desc)
        for w in raw_entities:
            camel_words = re.sub(r'([a-z])([A-Z])', r'\\1 \\2', w).split()
            for word in camel_words:
                lw = word.lower()
                if lw not in action_words:
                    bp_entities.add(lw)
        # methods names and params
        methods = c.get('methods', [])
        for m in methods:
            mname = m.get('name', '')
            parts = mname.lower().replace('-', ' ').replace('_', ' ').split()
            bp_entities.update(p.strip() for p in parts if p.strip())
            camel_mname = re.sub(r'([a-z])([A-Z])', r'\\1 \\2', mname.lower()).split()
            bp_entities.update(p.strip() for p in camel_mname if p.strip())
            params = m.get('params', {})
            for pkey in params:
                p_lower = pkey.lower()
                parts = p_lower.replace('-', ' ').replace('_', ' ').split()
                bp_entities.update(p.strip() for p in parts if p.strip())
                camel_p = re.sub(r'([a-z])([A-Z])', r'\\1 \\2', p_lower).split()
                bp_entities.update(p.strip() for p in camel_p if p.strip())
    if input_entities:
        input_list = list(input_entities)
        matched = 0
        for inp in input_list:
            for bp in bp_entities:
                if SequenceMatcher(None, inp, bp).ratio() > 0.75:
                    matched += 1
                    break
        ratio = (len(input_list) - matched) / len(input_list)
        if ratio > 0:
            losses.append(('entity', round(ratio, 2)))

    # Constraints
    constraint_keywords = ['must', 'before', 'validate', 'handle', 'ensure', 'require']
    has_constraint = any(kw in input_text.lower() for kw in constraint_keywords)
    if has_constraint:
        bp_text = ' '.join(str(c.get('description', '')) + ' ' + str(c.get('methods', '')) for c in components)
        handling_keywords = ['validate', 'handle', 'check', 'auth', 'ensure']
        has_handling = any(kw in bp_text.lower() for kw in handling_keywords)
        if not has_handling:
            losses.append(('constraint', 0.7))

    # Trivial input: no losses
    if len(input_text.strip()) < 20:
        return []

    return losses
