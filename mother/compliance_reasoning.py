"""
Compliance reasoning — regulatory exposure, blueprint compliance, data sovereignty.

LEAF module. Genome #120 (legal-awareness), #121 (compliance-checking),
#125 (data-sovereignty-respecting).

All functions are pure — no external API calls, no LLM invocations.
Heuristic keyword analysis over structured inputs.
"""

from typing import Dict, FrozenSet, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Heuristic constants
# ---------------------------------------------------------------------------

_REGULATORY_DOMAINS: Dict[str, FrozenSet[str]] = {
    "healthcare": frozenset({
        "health", "medical", "patient", "clinical", "diagnosis",
        "treatment", "hospital", "pharmacy", "ehr", "hipaa",
    }),
    "finance": frozenset({
        "finance", "banking", "payment", "trading", "investment",
        "loan", "credit", "insurance", "fintech", "kyc",
    }),
    "privacy": frozenset({
        "personal", "user-data", "tracking", "cookie", "consent",
        "profile", "analytics", "behavioral", "biometric", "pii",
    }),
    "ai": frozenset({
        "ai", "machine-learning", "ml", "model", "algorithm",
        "automated-decision", "prediction", "classification", "llm", "neural",
    }),
    "education": frozenset({
        "education", "student", "school", "university", "lms",
        "grade", "enrollment", "academic", "ferpa", "coppa",
    }),
    "food": frozenset({
        "food", "restaurant", "nutrition", "ingredient", "allergen",
        "recipe", "delivery", "fda", "menu", "catering",
    }),
}

_JURISDICTION_REGULATIONS: Dict[str, List[str]] = {
    "us": [
        "SOC 2 Type II", "CCPA/CPRA (California Privacy)",
        "Section 508 (Accessibility)", "FTC Act (Consumer Protection)",
    ],
    "eu": [
        "GDPR (General Data Protection)", "Digital Services Act",
        "AI Act", "ePrivacy Directive", "WCAG 2.1 AA (Accessibility)",
    ],
    "uk": [
        "UK GDPR", "Data Protection Act 2018",
        "Online Safety Act", "Equality Act (Accessibility)",
    ],
    "canada": [
        "PIPEDA (Privacy)", "AIDA (AI & Data Act)",
        "CASL (Anti-Spam)", "Accessibility for Ontarians",
    ],
    "australia": [
        "Privacy Act 1988", "Consumer Data Right",
        "Online Safety Act", "Disability Discrimination Act",
    ],
    "global": [
        "ISO 27001 (Information Security)", "OWASP Top 10 (Web Security)",
        "WCAG 2.1 (Accessibility)",
    ],
}

# Domain-specific regulations
_DOMAIN_REGULATIONS: Dict[str, List[str]] = {
    "healthcare": ["HIPAA (US)", "HITECH Act", "FDA 21 CFR Part 11", "HL7/FHIR Standards"],
    "finance": ["PCI DSS", "SOX", "AML/KYC", "MiFID II (EU)", "Dodd-Frank (US)"],
    "privacy": ["GDPR Art. 6 (Lawful Basis)", "CCPA Right to Delete", "Privacy by Design"],
    "ai": ["EU AI Act Risk Classification", "NIST AI RMF", "Algorithmic Accountability"],
    "education": ["FERPA (US)", "COPPA (US)", "Student Data Privacy"],
    "food": ["FDA Labeling", "EU Food Safety (EFSA)", "Allergen Disclosure"],
}

_DATA_SOVEREIGNTY_RULES: Dict[str, FrozenSet[str]] = {
    "eu": frozenset({"eu", "eea", "europe", "germany", "france", "ireland", "netherlands"}),
    "china": frozenset({"china", "cn", "mainland-china"}),
    "russia": frozenset({"russia", "ru"}),
    "india": frozenset({"india", "in"}),
    "us": frozenset({"us", "usa", "united-states"}),  # US does not enforce locality
}

# Zones that enforce data locality (must keep data within zone)
_ENFORCING_ZONES: FrozenSet[str] = frozenset({"eu", "china", "russia", "india"})

_COMPLIANCE_PATTERNS: Dict[str, FrozenSet[str]] = {
    "data_storage": frozenset({
        "database", "storage", "cache", "persistence", "db",
        "repository", "store", "warehouse", "archive",
    }),
    "auth": frozenset({
        "auth", "authentication", "login", "password", "session",
        "token", "oauth", "sso", "identity", "credentials",
    }),
    "payment": frozenset({
        "payment", "billing", "stripe", "checkout", "invoice",
        "subscription", "charge", "refund", "transaction",
    }),
    "user_data": frozenset({
        "user", "profile", "account", "registration", "personal",
        "preferences", "settings", "consent", "gdpr",
    }),
    "external_api": frozenset({
        "api", "webhook", "integration", "third-party", "external",
        "endpoint", "rest", "graphql", "sdk",
    }),
}

# Compliance requirements by pattern category
_PATTERN_REQUIREMENTS: Dict[str, List[str]] = {
    "data_storage": [
        "Encryption at rest required",
        "Backup and recovery plan needed",
        "Data retention policy required",
    ],
    "auth": [
        "OWASP authentication best practices",
        "Password hashing (bcrypt/argon2)",
        "Session management and timeout",
        "MFA consideration for sensitive operations",
    ],
    "payment": [
        "PCI DSS compliance required",
        "No raw card data storage",
        "Secure payment processor integration",
        "Refund and dispute handling",
    ],
    "user_data": [
        "Privacy policy required",
        "Consent mechanism needed",
        "Data deletion capability (right to be forgotten)",
        "Data portability support",
    ],
    "external_api": [
        "API key rotation policy",
        "Rate limiting and abuse prevention",
        "Data flow documentation",
        "Third-party security assessment",
    ],
}


# ---------------------------------------------------------------------------
# #120 — Legal-awareness: regulatory exposure assessment
# ---------------------------------------------------------------------------

def assess_regulatory_exposure(
    domain_keywords: List[str],
    geography: str = "global",
) -> List[str]:
    """Assess regulatory exposure from domain keywords and geography.

    Returns list of applicable regulatory frameworks with context.
    """
    if not domain_keywords:
        return ["No domain keywords provided — unable to assess regulatory exposure"]

    findings: List[str] = []
    keywords_lower = frozenset(k.lower() for k in domain_keywords)

    # Cross-reference domain keywords with regulatory domains
    matched_domains: List[str] = []
    for domain, domain_kw in _REGULATORY_DOMAINS.items():
        overlap = keywords_lower & domain_kw
        if overlap:
            matched_domains.append(domain)
            domain_regs = _DOMAIN_REGULATIONS.get(domain, [])
            for reg in domain_regs:
                findings.append(f"[{domain}] {reg} — triggered by: {', '.join(sorted(overlap))}")

    # Map geography to jurisdiction regulations
    geo_lower = geography.lower().strip()
    geo_regs = _JURISDICTION_REGULATIONS.get(geo_lower)
    if geo_regs:
        for reg in geo_regs:
            findings.append(f"[{geo_lower}] {reg}")
    elif geo_lower != "global":
        # Unknown geography — apply global baseline
        for reg in _JURISDICTION_REGULATIONS["global"]:
            findings.append(f"[global] {reg} (geography '{geography}' not recognized — applying global baseline)")

    # Always apply global baseline if no geo-specific rules
    if not geo_regs and geo_lower == "global":
        for reg in _JURISDICTION_REGULATIONS["global"]:
            findings.append(f"[global] {reg}")

    return findings


# ---------------------------------------------------------------------------
# #121 — Compliance-checking: blueprint component compliance
# ---------------------------------------------------------------------------

def check_blueprint_compliance(
    component_names: List[str],
    constraints: Optional[List[str]] = None,
    domain: str = "",
) -> List[str]:
    """Check blueprint components for compliance requirements.

    Matches component names against compliance patterns, cross-references
    with domain requirements, and notes already-addressed concerns.

    Returns list of compliance notes/requirements.
    """
    if not component_names:
        return []

    notes: List[str] = []
    constraint_text = " ".join(c.lower() for c in (constraints or []))
    components_lower = [c.lower() for c in component_names]
    all_component_words = frozenset(" ".join(components_lower).split())

    # Match components against compliance patterns
    for category, pattern_keywords in _COMPLIANCE_PATTERNS.items():
        matched_components = []
        for comp in component_names:
            comp_words = frozenset(comp.lower().replace("-", " ").replace("_", " ").split())
            if comp_words & pattern_keywords:
                matched_components.append(comp)

        if matched_components:
            reqs = _PATTERN_REQUIREMENTS.get(category, [])
            for req in reqs:
                # Check if constraint already addresses this
                req_words = frozenset(req.lower().split())
                if constraint_text and (req_words & frozenset(constraint_text.split())):
                    notes.append(
                        f"[{category}] {req} — likely addressed in constraints "
                        f"(components: {', '.join(matched_components)})"
                    )
                else:
                    notes.append(
                        f"[{category}] {req} "
                        f"(components: {', '.join(matched_components)})"
                    )

    # Domain-specific requirements
    if domain:
        domain_lower = domain.lower()
        domain_regs = _DOMAIN_REGULATIONS.get(domain_lower, [])
        for reg in domain_regs:
            notes.append(f"[domain:{domain_lower}] {reg}")

    return notes


# ---------------------------------------------------------------------------
# #125 — Data-sovereignty-respecting: data locality validation
# ---------------------------------------------------------------------------

def validate_data_locality(
    data_locations: List[str],
    jurisdiction: str,
) -> Tuple[bool, List[str]]:
    """Validate that data locations comply with jurisdiction sovereignty rules.

    Args:
        data_locations: Where data is stored (e.g., ["us-east-1", "eu-west-1"]).
        jurisdiction: Governing jurisdiction (e.g., "eu", "china", "us").

    Returns (is_compliant, violations).
    """
    if not data_locations:
        return True, []

    juris_lower = jurisdiction.lower().strip()

    # US does not enforce data locality
    if juris_lower in _DATA_SOVEREIGNTY_RULES.get("us", set()):
        return True, []

    # Check if jurisdiction enforces locality
    enforcing_zone: Optional[str] = None
    for zone in _ENFORCING_ZONES:
        if juris_lower in _DATA_SOVEREIGNTY_RULES.get(zone, set()):
            enforcing_zone = zone
            break

    if not enforcing_zone:
        # Unknown or non-enforcing jurisdiction
        return True, []

    # Check each data location
    allowed_locations = _DATA_SOVEREIGNTY_RULES[enforcing_zone]
    violations: List[str] = []

    for loc in data_locations:
        loc_lower = loc.lower().strip()
        # Check if location is within the allowed zone
        in_zone = False
        for allowed in allowed_locations:
            if allowed in loc_lower or loc_lower in allowed:
                in_zone = True
                break

        if not in_zone:
            violations.append(
                f"Data in '{loc}' violates {enforcing_zone.upper()} data sovereignty — "
                f"must be within {', '.join(sorted(allowed_locations))}"
            )

    is_compliant = len(violations) == 0
    return is_compliant, violations
