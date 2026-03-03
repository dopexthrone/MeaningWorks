"""Unified naming for code emission — LEAF MODULE (stdlib only).

Single source of truth for name transformations across project_writer,
materialization, agent_emission, and runtime_scaffold.
"""

import re


def sanitize_name(name: str) -> str:
    """Strip path prefixes, special chars, normalize to word-separated form.

    Handles component names like '/build', '~/.motherlabs/inputHistory',
    'mother>Prompt', '/tools', '/status' by extracting meaningful words.
    """
    # Strip leading path components — take last segment
    if '/' in name:
        name = name.rsplit('/', 1)[-1]
    # Replace special chars (>, <, ~, ., @, #, etc.) with spaces as separators
    name = re.sub(r'[^a-zA-Z0-9_\s-]', ' ', name)
    # Collapse whitespace
    name = re.sub(r'\s+', ' ', name).strip()
    # If empty after sanitization, fallback
    return name or 'component'


def to_snake(name: str) -> str:
    """Convert any name to snake_case. THE canonical implementation."""
    name = sanitize_name(name)
    # Insert underscore before uppercase letters preceded by lowercase/digit
    s = re.sub(r'(?<=[a-z0-9])([A-Z])', r'_\1', name)
    # Replace remaining non-alphanumeric with underscore, lowercase
    s = re.sub(r'[^a-z0-9_]', '_', s.lower())
    # Collapse multiple underscores and strip
    s = re.sub(r'_+', '_', s).strip('_')
    return s or 'component'


def to_pascal(name: str) -> str:
    """Convert any name to PascalCase. THE canonical implementation."""
    name = sanitize_name(name)
    # If no separators and already starts uppercase, assume PascalCase
    if not re.search(r'[\s_-]', name) and name and name[0].isupper():
        return re.sub(r'[^a-zA-Z0-9]', '', name)
    # Convert from spaces/underscores/hyphens
    parts = re.split(r'[\s_-]+', name)
    return ''.join(p.capitalize() for p in parts if p)


def slugify(text: str) -> str:
    """Convert text to valid Python package name."""
    slug = re.sub(r'[^a-z0-9]+', '_', text.lower().strip()).strip('_')
    if slug and not slug[0].isalpha():
        slug = 'p_' + slug
    return slug or 'project'
