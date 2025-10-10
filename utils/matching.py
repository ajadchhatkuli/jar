from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class Triplet:
    subject: str
    verb: str
    object: str


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _normalize_verb(verb: str) -> str:
    v = verb.strip().lower()
    if not v:
        return v

    irregular = {
        "adds": "add",
        "applies": "apply",
        "checks": "check",
        "cleans": "clean",
        "clears": "clear",
        "coats": "coat",
        "connects": "connect",
        "directs": "direct",
        "displays": "display",
        "drains": "drain",
        "fills": "fill",
        "illuminates": "illuminate",
        "inspects": "inspect",
        "installs": "install",
        "loosens": "loosen",
        "lubricates": "lubricate",
        "measures": "measure",
        "mounts": "mount",
        "opens": "open",
        "places": "place",
        "positions": "position",
        "pours": "pour",
        "reads": "read",
        "reattaches": "reattach",
        "reinstalls": "reinstall",
        "removes": "remove",
        "rinses": "rinse",
        "tightens": "tighten",
        "tops": "top",
        "unboxes": "unbox",
        "uses": "use",
        "washes": "wash",
        "shines": "shine",
    }
    if v in irregular:
        return irregular[v]

    if v.endswith("ies") and len(v) > 3:
        return v[:-3] + "y"
    if v.endswith("es") and len(v) > 2:
        # Preserve trailing vowels like 'shine' -> 'shine'
        base = v[:-1]
        if base.endswith("s") and not base.endswith("ss"):
            base = base[:-1]
        return base
    if v.endswith("s") and len(v) > 1:
        return v[:-1]
    return v


def parse_triplet(text: str) -> Optional[Triplet]:
    if not isinstance(text, str):
        return None
    parts = text.strip().split(" ", 2)
    if len(parts) < 3:
        return None
    subject, verb, obj = parts
    subject = subject.strip()
    verb = verb.strip()
    obj = obj.strip()
    if not (subject and verb and obj):
        return None
    return Triplet(subject=subject, verb=verb, object=obj)


def load_step_triplets(howto_path: str) -> Dict[int, List[Triplet]]:
    data = json.loads(Path(howto_path).read_text())
    steps = data.get("steps", []) if isinstance(data, dict) else data
    out: Dict[int, List[Triplet]] = {}
    for step in steps:
        try:
            step_id = int(step.get("id"))
        except (TypeError, ValueError):
            continue
        triplets: List[Triplet] = []
        for sub in step.get("sub-steps", []):
            triplet = parse_triplet(sub)
            if triplet is None:
                continue
            triplets.append(triplet)
        if triplets:
            out[step_id] = triplets
    return out


def match_steps(
    step_triplets: Dict[int, List[Triplet]],
    actions: Iterable[Dict[str, Any]],
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Return mapping of step_id -> list of matches where each match contains the
    original triplet and the action that satisfied it. A step is included only
    when every triplet has a matching action.
    """
    normalized_actions: List[Dict[str, Any]] = []
    for action in actions:
        subject = str(action.get("subject", "")).strip()
        verb = str(action.get("verb", "")).strip()
        obj = str(action.get("object", "")).strip()
        if not (subject and verb and obj):
            continue
        normalized_actions.append(
            {
                "normalized": (
                    _normalize_text(subject),
                    _normalize_verb(verb),
                    _normalize_text(obj),
                ),
                "action": action,
            }
        )

    matches: Dict[int, List[Dict[str, Any]]] = {}
    for step_id, triplets in step_triplets.items():
        step_matches: List[Dict[str, Any]] = []
        for triplet in triplets:
            triplet_norm = (
                _normalize_text(triplet.subject),
                _normalize_verb(triplet.verb),
                _normalize_text(triplet.object),
            )
            found_action: Optional[Dict[str, Any]] = None
            for candidate in normalized_actions:
                if candidate["normalized"] == triplet_norm:
                    found_action = candidate["action"]
                    break
            if not found_action:
                step_matches = []
                break
            step_matches.append(
                {
                    "triplet": {
                        "subject": triplet.subject,
                        "verb": triplet.verb,
                        "object": triplet.object,
                    },
                    "action": {
                        "subject": found_action.get("subject"),
                        "verb": found_action.get("verb"),
                        "object": found_action.get("object"),
                        "time": found_action.get("time"),
                        "details": found_action.get("details"),
                    },
                }
            )
        if step_matches:
            matches[step_id] = step_matches
    return matches
