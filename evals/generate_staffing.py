#!/usr/bin/env python3
"""Generate the STAFFING bench: synthetic consultant CVs + assignment briefs.

The staffing showcase (ROADMAP 9.0) matches consultant CVs against client
assignment requests. Real CVs are GDPR personal data, so the bench is
fully synthetic — and synthetic beats real for eval integrity, because
the ground truth is TRUE BY CONSTRUCTION:

1. Personas are STRUCTURED SPECS first (skills, years, languages,
   domains), deterministically derived from archetype tables with a
   fixed seed. CVs are rendered FROM the specs afterwards.
2. Expected match tiers are computed by a mechanical oracle over the
   specs (strong = every must-have met; partial = exactly one missing),
   never by judgment calls over prose.
3. Rendered text is VERIFIED against the spec: a CV must mention every
   spec skill (exact spelling, word-boundary regex) and may not mention
   any other taxonomy skill — otherwise the oracle's labels would drift
   from what the matcher can actually read. Briefs are checked the same
   way (every requirement named, nothing extra). Violations trigger a
   re-render; persistent violations fail the run.
4. One assignment (a08) is deliberately unsatisfiable — zero strong
   matches exist — so the matcher's honesty ("no perfect candidate;
   closest miss X") is itself measurable.

Generated artifacts (corpus_staffing/, staffing_personas.json,
golden_staffing.jsonl) are committed and reviewable like any golden set.
Re-running is idempotent: existing CVs and briefs are kept (and
re-verified), only missing ones are rendered.

Usage (from the repository root; full render costs ~$0.60 one-time):

    python evals/generate_staffing.py --specs-only   # free: oracle matrix
    python evals/generate_staffing.py --limit 2      # render 2 sample CVs
    python evals/generate_staffing.py                # render everything
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

EVALS_DIR = Path(__file__).parent
CORPUS_DIR = EVALS_DIR / "corpus_staffing"
PERSONAS_PATH = EVALS_DIR / "staffing_personas.json"
GOLDEN_PATH = EVALS_DIR / "golden_staffing.jsonl"

SEED = 20260715  # bench is deterministic; change = new bench version
CURRENT_YEAR = 2026

# --------------------------------------------------------------------------
# Skill taxonomy. Canonical names are what specs, briefs, and the oracle
# share; the regexes verify rendered prose against them. Overrides exist
# where naive word-boundary matching lies (e.g. \bEmbedded C\b would match
# inside "Embedded C++"; \bCAN\b must not match "can" or "CANoe").
# --------------------------------------------------------------------------

_PATTERN_OVERRIDES = {
    "C++": r"C\+\+",
    "Embedded C": r"(?i)\bEmbedded[ -]C\b(?!\+)",
    "CAN bus": r"\bCAN\b",
    "LIN": r"\bLIN\b",
    "Rust": r"\bRust\b",
    "Helm": r"\bHelm\b",
    "React": r"\bReact\b",
    "AWS": r"\bAWS\b",
    "Azure": r"(?i)\bAzure\b(?!\s+DevOps)",
    "Ethernet TSN": r"(?i)\b(?:Ethernet\s+)?TSN\b",
    "device drivers": r"(?i)\bdevice driver",
    "GitLab CI": r"(?i)\bGitLab CI",
    "ASPICE": r"(?i)\bA(?:utomotive\s+)?SPICE\b",
    "dSPACE HIL": r"(?i)\bdSPACE\b",
    "Vector CANoe": r"(?i)\bCANoe\b",
    "Golang": r"(?i)\bGolang\b",
    "Node.js": r"(?i)\bNode\.js\b",
}

SKILLS = [
    # embedded / automotive
    "Embedded C",
    "C++",
    "Rust",
    "AUTOSAR Classic",
    "AUTOSAR Adaptive",
    "SOME/IP",
    "CAN bus",
    "LIN",
    "Ethernet TSN",
    "ISO 26262",
    "ISO 21434",
    "MISRA C",
    "ASPICE",
    "Vector CANoe",
    "dSPACE HIL",
    "Simulink",
    "QNX",
    "FreeRTOS",
    "Zephyr",
    "Embedded Linux",
    "Yocto",
    "device drivers",
    "secure boot",
    # telecom
    "5G RAN",
    "5G Core",
    "O-RAN",
    "3GPP",
    "DPDK",
    "gRPC",
    # cloud / devops
    "Kubernetes",
    "Docker",
    "Helm",
    "Terraform",
    "Ansible",
    "AWS",
    "Azure",
    "Jenkins",
    "GitLab CI",
    "Azure DevOps",
    "ArgoCD",
    "Prometheus",
    "Grafana",
    "Kafka",
    "MQTT",
    "Python",
    "Golang",
    "Bash",
    "Artifactory",
    # adjacent (bench noise)
    "Robot Framework",
    "React",
    "Node.js",
    "TypeScript",
    "PostgreSQL",
]

PATTERNS = {
    s: re.compile(_PATTERN_OVERRIDES.get(s, rf"(?i)\b{re.escape(s)}\b")) for s in SKILLS
}

# --------------------------------------------------------------------------
# Pools. All names fictional; clients are anonymized descriptors, which is
# how real consultant CVs are written anyway.
# --------------------------------------------------------------------------

FIRST_NAMES = [
    "Anders",
    "Elin",
    "Johan",
    "Sara",
    "Mikael",
    "Astrid",
    "Henrik",
    "Freja",
    "Lars",
    "Ingrid",
    "Oskar",
    "Maja",
    "Erik",
    "Linnea",
    "Jonas",
    "Amira",
    "Nikolai",
    "Priya",
    "Tomas",
    "Yusuf",
    "Katarina",
    "Wei",
    "Marta",
    "Sofia",
    "Petri",
    "Aino",
    "Bjorn",
    "Sigrid",
    "Emil",
    "Noor",
    "Viktor",
    "Hanna",
    "Aleksi",
    "Dario",
    "Magnus",
    "Leila",
    "Ola",
    "Camille",
    "Sebastian",
    "Ida",
]
LAST_NAMES = [
    "Lindqvist",
    "Berg",
    "Johansson",
    "Virtanen",
    "Hansen",
    "Nystrom",
    "Karlsson",
    "Haddad",
    "Eriksson",
    "Nielsen",
    "Holm",
    "Patel",
    "Andersson",
    "Korhonen",
    "Dahl",
    "Oberg",
    "Chen",
    "Lund",
    "Bakke",
    "Sandberg",
    "Nilsson",
    "Kowalski",
    "Aaltonen",
    "Petrov",
    "Strand",
    "Okafor",
    "Sorensen",
    "Blomqvist",
    "Larsen",
    "Rahman",
    "Sjoberg",
    "Novak",
    "Hagg",
    "Lindholm",
    "Fischer",
    "Makinen",
    "Solberg",
    "Costa",
    "Engstrom",
    "Vik",
]

CLIENTS = {
    "automotive": [
        "a major Swedish truck OEM",
        "a German premium car manufacturer",
        "a Nordic passenger-car OEM",
        "a Swedish electric-vehicle startup",
        "a tier-1 automotive supplier",
        "an ADAS software company",
    ],
    "telecom": [
        "a Nordic telecom equipment vendor",
        "a Finnish network operator",
        "a European 5G infrastructure vendor",
        "a Swedish telecom operator",
    ],
    "generic": [
        "a Nordic bank",
        "a Danish logistics company",
        "a Swedish medtech company",
        "a Norwegian energy company",
        "a Finnish industrial-IoT company",
    ],
}

SCHOOLS = {
    "Sweden": [
        "Chalmers University of Technology",
        "KTH Royal Institute of Technology",
        "Lund University",
        "Linkoping University",
        "Uppsala University",
    ],
    "Finland": ["Aalto University", "Tampere University"],
    "Norway": ["NTNU Trondheim", "University of Oslo"],
    "Denmark": ["Technical University of Denmark (DTU)"],
}

CERT_RULES = [
    ("AWS", "AWS Certified Solutions Architect - Associate", 0.6),
    ("Azure", "Microsoft Certified: Azure Administrator Associate", 0.5),
    ("Kubernetes", "CKA (Certified Kubernetes Administrator)", 0.5),
    ("Terraform", "HashiCorp Certified: Terraform Associate", 0.4),
    ("ISO 26262", "ISO 26262 functional-safety training (TUV)", 0.7),
    ("ISO 21434", "ISO 21434 cybersecurity engineering training", 0.5),
    ("Robot Framework", "ISTQB Foundation Level", 0.8),
    ("ASPICE", "Automotive SPICE provisional assessor training", 0.4),
]

# --------------------------------------------------------------------------
# Archetypes: count, always-present core skills, and (skill, probability)
# pools. Probabilities are tuned so the oracle constraints below hold for
# the fixed seed — the printed matrix is the thing to look at when tuning.
# --------------------------------------------------------------------------

ARCHETYPES = [
    {
        "key": "auto_classic",
        "count": 8,
        "title": "Embedded Software Engineer",
        "domains": [(("automotive",), 1.0)],
        "core": ["Embedded C", "AUTOSAR Classic", "CAN bus"],
        "pool": [
            ("Vector CANoe", 0.8),
            ("MISRA C", 0.7),
            ("ISO 26262", 0.45),
            ("LIN", 0.5),
            ("C++", 0.6),
            ("Simulink", 0.4),
            ("Python", 0.45),
            ("Jenkins", 0.35),
            ("dSPACE HIL", 0.3),
            ("ASPICE", 0.3),
            ("FreeRTOS", 0.25),
            ("SOME/IP", 0.4),
        ],
        "years": (2, 16),
        "swedish": 0.75,
        "locations": ["Gothenburg, Sweden"] * 3 + ["Stockholm, Sweden", "Lund, Sweden"],
    },
    {
        "key": "auto_adaptive",
        "count": 4,
        "title": "Software Engineer, Software-Defined Vehicle",
        "domains": [(("automotive",), 1.0)],
        "core": ["C++", "AUTOSAR Adaptive", "SOME/IP"],
        "pool": [
            ("Ethernet TSN", 0.6),
            ("QNX", 0.5),
            ("ISO 21434", 0.5),
            ("Embedded Linux", 0.5),
            ("ISO 26262", 0.4),
            ("CAN bus", 0.4),
            ("Yocto", 0.35),
            ("Python", 0.4),
            ("gRPC", 0.3),
        ],
        "years": (3, 12),
        "swedish": 0.5,
        "locations": ["Gothenburg, Sweden", "Gothenburg, Sweden", "Stockholm, Sweden"],
    },
    {
        "key": "embedded_linux",
        "count": 5,
        "title": "Embedded Linux Engineer",
        "domains": [
            (("automotive",), 0.4),
            (("telecom",), 0.4),
            (("automotive", "telecom"), 0.2),
        ],
        "core": ["Embedded C", "Embedded Linux", "Yocto"],
        "pool": [
            ("device drivers", 0.7),
            ("C++", 0.6),
            ("Python", 0.5),
            ("secure boot", 0.45),
            ("Docker", 0.4),
            ("FreeRTOS", 0.35),
            ("Zephyr", 0.3),
            ("Rust", 0.3),
            ("MQTT", 0.3),
            ("GitLab CI", 0.3),
        ],
        "years": (3, 14),
        "swedish": 0.5,
        "locations": [
            "Gothenburg, Sweden",
            "Stockholm, Sweden",
            "Helsinki, Finland",
            "Oslo, Norway",
        ],
    },
    {
        "key": "telecom_ran",
        "count": 4,
        "title": "5G RAN Software Engineer",
        "domains": [(("telecom",), 1.0)],
        "core": ["C++", "5G RAN", "3GPP"],
        "pool": [
            ("DPDK", 0.6),
            ("O-RAN", 0.6),
            ("Kubernetes", 0.8),
            ("Docker", 0.6),
            ("Python", 0.5),
            ("GitLab CI", 0.4),
            ("gRPC", 0.4),
            ("Golang", 0.35),
        ],
        "years": (3, 15),
        "swedish": 0.4,
        "locations": [
            "Stockholm, Sweden",
            "Stockholm, Sweden",
            "Helsinki, Finland",
            "Oslo, Norway",
        ],
    },
    {
        "key": "telecom_core",
        "count": 3,
        "title": "Cloud-Native Developer, 5G Core",
        "domains": [(("telecom",), 1.0)],
        "core": ["Golang", "5G Core", "Kubernetes"],
        "pool": [
            ("gRPC", 0.7),
            ("Docker", 0.7),
            ("Helm", 0.6),
            ("GitLab CI", 0.5),
            ("Prometheus", 0.5),
            ("Kafka", 0.4),
            ("3GPP", 0.4),
            ("Azure", 0.3),
        ],
        "years": (2, 10),
        "swedish": 0.35,
        "locations": ["Stockholm, Sweden", "Helsinki, Finland"],
    },
    {
        "key": "devops_embedded",
        "count": 5,
        "title": "DevOps Engineer",
        "domains": [(("automotive",), 0.7), (("telecom",), 0.3)],
        "core": ["Python", "Jenkins", "Docker"],
        "pool": [
            ("Bash", 0.7),
            ("GitLab CI", 0.6),
            ("Yocto", 0.7),
            ("Artifactory", 0.5),
            ("Kubernetes", 0.45),
            ("Ansible", 0.4),
            ("Grafana", 0.35),
            ("Azure DevOps", 0.3),
            ("dSPACE HIL", 0.3),
            ("Vector CANoe", 0.2),
        ],
        "years": (2, 12),
        "swedish": 0.6,
        "locations": [
            "Gothenburg, Sweden",
            "Gothenburg, Sweden",
            "Stockholm, Sweden",
            "Linkoping, Sweden",
        ],
    },
    {
        "key": "cloud_platform",
        "count": 6,
        "title": "Cloud Platform Engineer",
        "domains": [(("automotive",), 0.4), (("telecom",), 0.3), ((), 0.3)],
        "core": ["Terraform", "Kubernetes", "Docker"],
        "pool": [
            ("AWS", 0.65),
            ("Python", 0.6),
            ("Azure", 0.5),
            ("Helm", 0.5),
            ("Prometheus", 0.5),
            ("Grafana", 0.45),
            ("Kafka", 0.4),
            ("Golang", 0.4),
            ("GitLab CI", 0.4),
            ("ArgoCD", 0.35),
            ("MQTT", 0.3),
            ("Ansible", 0.3),
        ],
        "years": (2, 12),
        "swedish": 0.45,
        "locations": [
            "Stockholm, Sweden",
            "Stockholm, Sweden",
            "Gothenburg, Sweden",
            "Copenhagen, Denmark",
            "Oslo, Norway",
        ],
    },
    {
        "key": "functional_safety",
        "count": 3,
        "title": "Functional Safety Engineer",
        "domains": [(("automotive",), 1.0)],
        "core": ["ISO 26262", "Embedded C", "MISRA C"],
        "pool": [
            ("ASPICE", 0.7),
            ("AUTOSAR Classic", 0.6),
            ("CAN bus", 0.6),
            ("ISO 21434", 0.4),
            ("Simulink", 0.4),
            ("Vector CANoe", 0.4),
        ],
        "years": (7, 18),
        "swedish": 0.85,
        "locations": ["Gothenburg, Sweden", "Gothenburg, Sweden", "Stockholm, Sweden"],
    },
    {
        "key": "qa_automation",
        "count": 1,
        "title": "Test Automation Engineer",
        "domains": [(("automotive",), 1.0)],
        "core": ["Python", "Robot Framework"],
        "pool": [
            ("Jenkins", 0.9),
            ("Docker", 0.5),
            ("CAN bus", 0.5),
            ("Bash", 0.5),
        ],
        "years": (3, 9),
        "swedish": 0.6,
        "locations": ["Gothenburg, Sweden"],
    },
    {
        "key": "fullstack",
        "count": 1,
        "title": "Fullstack Developer",
        "domains": [((), 1.0)],
        "core": ["TypeScript", "React", "Node.js"],
        "pool": [
            ("PostgreSQL", 0.8),
            ("Docker", 0.5),
            ("Azure", 0.4),
            ("Python", 0.3),
        ],
        "years": (2, 8),
        "swedish": 0.5,
        "locations": ["Stockholm, Sweden"],
    },
]

# --------------------------------------------------------------------------
# Assignments. must = list of requirement items; each item is a list of
# alternatives (any one satisfies it). a08 is deliberately unsatisfiable.
# --------------------------------------------------------------------------

ASSIGNMENTS = [
    {
        "id": "a01",
        "title": "Senior embedded developer, truck ECU platform",
        "client": "a major Swedish truck OEM",
        "location": "Gothenburg",
        "duration": "12 months",
        "domain": "automotive",
        "min_years": 5,
        "language": None,
        "must": [
            ["AUTOSAR Classic"],
            ["Embedded C"],
            ["CAN bus"],
            ["ISO 26262"],
            ["MISRA C"],
        ],
        "nice": ["Vector CANoe", "ASPICE", "Simulink"],
        "brief_note": None,
        "notes": "Bread-and-butter automotive embedded profile; several "
        "strong matches should exist.",
    },
    {
        "id": "a02",
        "title": "AUTOSAR Adaptive engineer, software-defined vehicle",
        "client": "a Nordic passenger-car OEM",
        "location": "Gothenburg",
        "duration": "18 months",
        "domain": "automotive",
        "min_years": 5,
        "language": None,
        "must": [["AUTOSAR Adaptive"], ["C++"], ["SOME/IP"]],
        "nice": ["Ethernet TSN", "QNX", "ISO 21434", "Yocto"],
        "brief_note": None,
        "notes": "Small strong pool by design (adaptive is a scarcer "
        "skill); partials are classic-platform people.",
    },
    {
        "id": "a03",
        "title": "5G RAN developer, cloud-native baseband",
        "client": "a Nordic telecom equipment vendor",
        "location": "Stockholm (Kista)",
        "duration": "12 months",
        "domain": "telecom",
        "min_years": 3,
        "language": None,
        "must": [["5G RAN"], ["C++", "Golang"], ["Kubernetes"]],
        "nice": ["O-RAN", "DPDK", "GitLab CI", "Helm"],
        "brief_note": None,
        "notes": "C++ or Golang both accepted; 5G Core people are the "
        "engineered near-miss (wrong network layer).",
    },
    {
        "id": "a04",
        "title": "DevOps engineer, embedded build & test infrastructure",
        "client": "a German premium car manufacturer",
        "location": "Gothenburg (hybrid)",
        "duration": "12 months",
        "domain": "automotive",
        "min_years": 3,
        "language": None,
        "must": [["Jenkins", "GitLab CI"], ["Docker"], ["Python"], ["Yocto"]],
        "nice": ["Artifactory", "Kubernetes", "dSPACE HIL", "Ansible"],
        "brief_note": None,
        "notes": "Yocto in a CI context separates embedded DevOps from "
        "generic DevOps.",
    },
    {
        "id": "a05",
        "title": "Cloud platform engineer, connected-vehicle backend",
        "client": "a Swedish electric-vehicle startup",
        "location": "Stockholm (remote-friendly)",
        "duration": "9 months",
        "domain": None,
        "min_years": 4,
        "language": None,
        "must": [["AWS", "Azure"], ["Kubernetes"], ["Terraform"]],
        "nice": ["Kafka", "MQTT", "Helm", "Golang"],
        "brief_note": "Experience from automotive or connected-vehicle "
        "projects is meriting but not required.",
        "notes": "Domain kept a soft preference on purpose: tests that "
        "the matcher does not over-filter on domain.",
    },
    {
        "id": "a06",
        "title": "Embedded Linux BSP engineer, radio units",
        "client": "a European 5G infrastructure vendor",
        "location": "Helsinki or remote",
        "duration": "12 months",
        "domain": None,
        "min_years": 4,
        "language": None,
        "must": [["Embedded Linux"], ["Yocto"], ["Embedded C"], ["device drivers"]],
        "nice": ["secure boot", "Rust", "Zephyr", "C++"],
        "brief_note": None,
        "notes": "BSP work: driver experience is the discriminator.",
    },
    {
        "id": "a07",
        "title": "Functional safety lead, ADAS platform",
        "client": "a tier-1 automotive supplier",
        "location": "Gothenburg",
        "duration": "24 months",
        "domain": "automotive",
        "min_years": 7,
        "language": "Swedish",
        "must": [["ISO 26262"], ["MISRA C"], ["ASPICE"]],
        "nice": ["AUTOSAR Classic", "ISO 21434", "Vector CANoe", "Simulink"],
        "brief_note": None,
        "notes": "Swedish is a hard requirement (safety audits and "
        "supplier meetings run in Swedish).",
    },
    {
        "id": "a08",
        "title": "Edge platform engineer, secure telecom workloads",
        "client": "a Finnish network operator",
        "location": "Helsinki",
        "duration": "12 months",
        "domain": "telecom",
        "min_years": 5,
        "language": None,
        "must": [["5G RAN", "5G Core"], ["Kubernetes"], ["secure boot"]],
        "nice": ["DPDK", "Helm", "GitLab CI", "Rust"],
        "brief_note": None,
        "notes": "DELIBERATELY UNSATISFIABLE: no persona combines 5G + "
        "Kubernetes + secure boot. The correct output is an "
        "honest 'no full match' plus the nearest partials.",
    },
]

# --------------------------------------------------------------------------
# Persona construction (deterministic).
# --------------------------------------------------------------------------


def _weighted_choice(rng, pairs):
    values = [v for v, _ in pairs]
    weights = [w for _, w in pairs]
    return rng.choices(values, weights=weights, k=1)[0]


def _engagements(rng, persona_skills, title, years, domains):
    """Split the career into engagements whose stacks cover every skill."""
    n = max(2, min(5, 1 + years // 3))
    # Newest first; each engagement gets a contiguous slice of years.
    spans, end = [], CURRENT_YEAR
    remaining = years
    for i in range(n):
        left = n - i - 1
        dur = remaining - left if left else remaining
        if left:
            dur = max(1, min(dur, round(remaining / (left + 1)) + rng.randint(0, 1)))
        start = end - dur
        spans.append((start, end))
        end, remaining = start, remaining - dur
    # Round-robin the skills so the union of stacks covers all of them.
    stacks = [[] for _ in range(n)]
    for idx, skill in enumerate(persona_skills):
        stacks[idx % n].append(skill)
    for stack in stacks:
        for core in persona_skills[:2]:
            if core not in stack:
                stack.append(core)
    engagements = []
    for i, ((start, end_y), stack) in enumerate(zip(spans, stacks)):
        # Clients stay inside the persona's domains: an ISO 26262 stack
        # at "a Nordic bank" would leak incoherence into the bench.
        domain = rng.choice(list(domains)) if domains else "generic"
        pool = CLIENTS.get(domain, CLIENTS["generic"])
        period = f"{start}-present" if i == 0 else f"{start}-{end_y}"
        engagements.append(
            {
                "role": title,
                "client": rng.choice(pool),
                "period": period,
                "stack": stack,
            }
        )
    return engagements


def build_personas():
    rng = random.Random(SEED)
    first = FIRST_NAMES[:]
    last = LAST_NAMES[:]
    rng.shuffle(first)
    rng.shuffle(last)

    personas, n = [], 0
    for arch in ARCHETYPES:
        for _ in range(arch["count"]):
            n += 1
            years = rng.randint(*arch["years"])
            skills = list(arch["core"])
            for skill, p in arch["pool"]:
                if rng.random() < p and skill not in skills:
                    skills.append(skill)
            location = rng.choice(arch["locations"])
            country = location.split(", ")[-1]
            languages = [{"language": "English", "level": "fluent"}]
            local = {
                "Sweden": "Swedish",
                "Finland": "Finnish",
                "Norway": "Norwegian",
                "Denmark": "Danish",
            }[country]
            if local == "Swedish":
                if rng.random() < arch["swedish"]:
                    languages.append({"language": "Swedish", "level": "native"})
            else:
                languages.append({"language": local, "level": "native"})
                if rng.random() < arch["swedish"] * 0.4:
                    languages.append({"language": "Swedish", "level": "professional"})
            if rng.random() < 0.15:
                languages.append({"language": "German", "level": "basic"})

            prefix = "Senior " if years >= 8 else ("Junior " if years < 3 else "")
            title = prefix + arch["title"]
            domains = _weighted_choice(rng, arch["domains"])
            certs = [c for s, c, p in CERT_RULES if s in skills and rng.random() < p]
            degree = "MSc" if rng.random() < 0.7 else "BSc"
            school = rng.choice(SCHOOLS[country])
            grad = CURRENT_YEAR - years - rng.randint(0, 2)
            name = f"{first[n - 1]} {last[n - 1]}"
            personas.append(
                {
                    "id": f"cv{n:02d}",
                    "name": name,
                    "title": title,
                    "archetype": arch["key"],
                    "years": years,
                    "location": location,
                    "domains": list(domains),
                    "languages": languages,
                    "skills": skills,
                    "certifications": certs,
                    "education": f"{degree} Computer Science and Engineering, "
                    f"{school}, {grad}",
                    "engagements": _engagements(rng, skills, title, years, domains),
                }
            )
    return personas


# --------------------------------------------------------------------------
# The oracle: expected tiers computed from specs, never from prose.
# --------------------------------------------------------------------------


def _speaks(persona, language):
    return any(
        entry["language"] == language
        and entry["level"] in ("native", "fluent", "professional")
        for entry in persona["languages"]
    )


def missing_requirements(persona, assignment):
    """Return the list of unmet must-have items (empty = strong match)."""
    missing = []
    for group in assignment["must"]:
        if not any(s in persona["skills"] for s in group):
            missing.append(" / ".join(group))
    if assignment["min_years"] and persona["years"] < assignment["min_years"]:
        missing.append(f"{assignment['min_years']}+ years of experience")
    if assignment["language"] and not _speaks(persona, assignment["language"]):
        missing.append(f"{assignment['language']} language")
    if assignment["domain"] and assignment["domain"] not in persona["domains"]:
        missing.append(f"{assignment['domain']} domain experience")
    return missing


def expected_tiers(personas, assignment):
    strong, partial = [], []
    for p in personas:
        missing = missing_requirements(p, assignment)
        nice_hits = sum(1 for s in assignment["nice"] if s in p["skills"])
        entry = {
            "id": p["id"],
            "name": p["name"],
            "years": p["years"],
            "nice_hits": nice_hits,
        }
        if not missing:
            strong.append(entry)
        elif len(missing) == 1:
            partial.append({**entry, "missing": missing[0]})
    key = lambda e: (-e["nice_hits"], -e["years"])  # noqa: E731
    return sorted(strong, key=key), sorted(partial, key=key)


def check_bench(personas):
    """Print the oracle matrix and enforce the bench-shape constraints."""
    problems = []
    for a in ASSIGNMENTS:
        strong, partial = expected_tiers(personas, a)
        strong_ids = ", ".join(f"{e['id']}(+{e['nice_hits']})" for e in strong) or "-"
        partial_ids = (
            ", ".join(f"{e['id']}(-{e['missing']})" for e in partial[:6]) or "-"
        )
        print(f"  {a['id']}: strong={len(strong)} [{strong_ids}]")
        print(f"       partial={len(partial)} [{partial_ids}]")
        if a["id"] == "a08":
            if strong:
                problems.append(f"{a['id']}: must have 0 strong matches")
            if len(partial) < 2:
                problems.append(f"{a['id']}: needs >=2 partial matches")
        else:
            if not 2 <= len(strong) <= 5:
                problems.append(f"{a['id']}: {len(strong)} strong matches (want 2-5)")
            if not partial:
                problems.append(f"{a['id']}: needs >=1 partial match")
    if problems:
        raise SystemExit("bench constraints violated:\n  " + "\n  ".join(problems))


# --------------------------------------------------------------------------
# Rendering + integrity checks.
# --------------------------------------------------------------------------

CV_PROMPT = """You are writing a FICTIONAL consultant CV for a synthetic \
staffing-demo corpus. The person does not exist. Every fact comes from the \
spec below — invent nothing beyond phrasing and modest, plausible outcomes.

Spec (JSON):
{spec}

Write the CV in Markdown, in the style Nordic IT consultancies use:

# <name>
**<title>** - <location>
Contact: <first>.<last>@synthetic.example

## Profile
3-4 sentences: {years} years of experience, domains, what the consultant
is strongest at. Concrete and plain; no buzzword soup.

## Core competencies
Grouped bullet lists covering the spec's skills.

## Selected engagements
One entry per engagement, newest first:
**<role> - <client descriptor>** (<period>)
followed by 3-5 bullets describing responsibilities and one concrete,
modest outcome. Use ONLY the technologies in that engagement's stack.
Outcomes are deliverables (an ECU shipped to start of production, a
pipeline migrated, an assessment passed) — never invented percentage or
metric claims. Vary the bullet phrasing between engagements.

## Certifications
## Education
## Languages

HARD RULES:
- Mention every skill in "skills" at least once, spelled exactly as given
  (e.g. "AUTOSAR Classic", "GitLab CI", "Golang", "CAN bus").
- Do NOT name any technology, protocol, standard, or certification that
  is not in the spec. Generic practices (Agile, Scrum, Git, Jira, code
  review, unit testing) are fine.
- Client descriptors stay exactly as given; never invent company names.
- 500-800 words. Return ONLY the Markdown document."""

BRIEF_PROMPT = """You are writing a FICTIONAL client assignment request for \
a synthetic staffing-demo corpus, as a client would send it to an IT \
consultancy's staffing desk.

Spec (JSON):
{spec}

Format, in Markdown, 200-300 words total:

# Assignment request: <title>
An opening paragraph: who the client is (use the descriptor verbatim),
1-2 invented but plausible sentences of product/context, the team
situation, duration, location, and a start about two months out.

## Requirements
Bullet list, one bullet per requirement item. Name every must-have skill
exactly as given. Where a requirement item lists alternatives, put them
in ONE bullet phrased with "or" ("X or Y") — they are alternatives, not
both required. State the experience level ONCE, as its own bullet
("at least {min_years} years of relevant experience") — never attach a
year count to individual skills. {language_line}

## Meriting
Bullet list of the nice-to-have skills, named exactly as given.
{brief_note}

HARD RULES: requirements read as mandatory ("must", "required");
meriting items read as optional; do NOT name any technology or standard
beyond those in the spec. Return ONLY the Markdown."""


def _llm(model: str):
    from dotenv import load_dotenv

    load_dotenv()
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model, temperature=0.6)


def _strip_fences(text: str) -> str:
    return (
        "\n".join(
            line for line in text.splitlines() if not line.strip().startswith("```")
        ).strip()
        + "\n"
    )


def taxonomy_mentions(text: str) -> set:
    return {s for s, rx in PATTERNS.items() if rx.search(text)}


def check_rendered(text: str, required: set, allowed: set):
    """Return (missing, leaked) taxonomy skills for a rendered document."""
    found = taxonomy_mentions(text)
    return sorted(required - found), sorted(found - allowed)


def _or_group_problems(text: str, must) -> list:
    """Alternatives must READ as alternatives — one line, joined by 'or'.

    The oracle scores multi-skill requirement items as any-of; a brief
    that renders them as separate "must have X" bullets silently turns
    OR into AND and detaches the labels from the prose.
    """
    problems = []
    for group in must:
        if len(group) < 2:
            continue
        ok = any(
            all(PATTERNS[s].search(line) for s in group)
            and re.search(r"\bor\b", line, re.IGNORECASE)
            for line in text.splitlines()
        )
        if not ok:
            problems.append(
                f"the alternatives {group} must appear together in one "
                f"bullet phrased '{' or '.join(group)}'"
            )
    return problems


def render_checked(
    llm, prompt: str, required: set, allowed: set, label: str, extra_check=None
) -> str:
    """Render, verify against the taxonomy, and retry with feedback."""
    messages = [("user", prompt)]
    for attempt in range(1, 4):
        text = _strip_fences(llm.invoke(messages).content)
        missing, leaked = check_rendered(text, required, allowed)
        extra = extra_check(text) if extra_check else []
        if not missing and not leaked and not extra:
            if attempt > 1:
                print(f"    {label}: clean on attempt {attempt}")
            return text
        messages += [
            ("assistant", text),
            (
                "user",
                "Revise the document. It MUST mention each of these, spelled "
                f"exactly as given: {missing or 'none missing'}. It must NOT "
                f"mention any of: {leaked or 'nothing to remove'}. "
                f"Also fix: {extra or 'nothing else'}. "
                "Return ONLY the corrected Markdown.",
            ),
        ]
    raise RuntimeError(
        f"{label}: integrity check failed after 3 attempts "
        f"(missing={missing}, leaked={leaked}, extra={extra})"
    )


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def render_cvs(llm, personas, limit=None):
    CORPUS_DIR.mkdir(exist_ok=True)
    rendered = 0
    for persona in personas:
        path = CORPUS_DIR / f"{persona['id']}_{_slug(persona['name'])}.md"
        skills = set(persona["skills"])
        if path.exists():
            missing, leaked = check_rendered(
                path.read_text(encoding="utf-8"), skills, skills
            )
            status = (
                "ok"
                if not missing and not leaked
                else (f"INTEGRITY DRIFT missing={missing} leaked={leaked}")
            )
            print(f"  cv: {path.name} exists ({status})")
            continue
        if limit is not None and rendered >= limit:
            continue
        prompt = CV_PROMPT.format(
            spec=json.dumps(persona, ensure_ascii=False, indent=2),
            years=persona["years"],
        )
        text = render_checked(llm, prompt, skills, skills, persona["id"])
        path.write_text(text, encoding="utf-8")
        rendered += 1
        print(f"  cv: wrote {path.name} ({len(text.split())} words)")
    return rendered


def render_briefs(llm, existing_briefs):
    briefs = {}
    for a in ASSIGNMENTS:
        required = {s for group in a["must"] for s in group} | set(a["nice"])
        cached = existing_briefs.get(a["id"])
        if (
            cached
            and check_rendered(cached, required, required) == ([], [])
            and not _or_group_problems(cached, a["must"])
        ):
            briefs[a["id"]] = cached
            print(f"  brief: {a['id']} cached (ok)")
            continue
        spec = {
            k: a[k]
            for k in (
                "title",
                "client",
                "location",
                "duration",
                "min_years",
                "language",
                "must",
                "nice",
            )
        }
        language_line = (
            f"State that {a['language']} is a mandatory requirement."
            if a["language"]
            else ""
        )
        note = f"Also mention: {a['brief_note']}" if a["brief_note"] else ""
        prompt = BRIEF_PROMPT.format(
            spec=json.dumps(spec, ensure_ascii=False, indent=2),
            min_years=a["min_years"],
            language_line=language_line,
            brief_note=note,
        )
        briefs[a["id"]] = render_checked(
            llm,
            prompt,
            required,
            required,
            a["id"],
            extra_check=lambda text, must=a["must"]: _or_group_problems(text, must),
        )
        print(f"  brief: wrote {a['id']} " f"({len(briefs[a['id']].split())} words)")
    return briefs


def write_golden(personas, briefs):
    with open(GOLDEN_PATH, "w", encoding="utf-8") as f:
        for a in ASSIGNMENTS:
            strong, partial = expected_tiers(personas, a)
            record = {
                "id": a["id"],
                "title": a["title"],
                "client": a["client"],
                "location": a["location"],
                "duration": a["duration"],
                "domain": a["domain"],
                "min_years": a["min_years"],
                "language": a["language"],
                "must_have": a["must"],
                "nice_to_have": a["nice"],
                "brief": briefs[a["id"]],
                "expected_strong": strong,
                "expected_partial": partial,
                "notes": a["notes"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  golden: wrote {GOLDEN_PATH.name} ({len(ASSIGNMENTS)} " "assignments)")


def _load_existing_briefs():
    if not GOLDEN_PATH.exists():
        return {}
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        return {
            record["id"]: record["brief"]
            for record in (json.loads(line) for line in f if line.strip())
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument(
        "--specs-only",
        action="store_true",
        help="build personas + oracle matrix, no rendering",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="render at most N new CVs (sampling runs)",
    )
    args = parser.parse_args()

    personas = build_personas()
    print(
        f"Personas: {len(personas)} "
        f"({', '.join(a['key'] + ':' + str(a['count']) for a in ARCHETYPES)})"
    )
    print("Oracle matrix:")
    check_bench(personas)
    PERSONAS_PATH.write_text(
        json.dumps(personas, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"  specs: wrote {PERSONAS_PATH.name}")
    if args.specs_only:
        return 0

    llm = _llm(args.model)
    print("Rendering CVs...")
    render_cvs(llm, personas, limit=args.limit)
    print("Rendering briefs...")
    briefs = render_briefs(llm, _load_existing_briefs())
    write_golden(personas, briefs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
