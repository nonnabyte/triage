"""Re-word the eval prompts into other layouts, to test that routing does not depend on our template.

Each eval prompt is taken apart with the dataset's own markers (this is test tooling: it knows the template
the router must not rely on) and written out again in layouts with different section labels, no labels at
all, the question before the document, and so on. The content is unchanged, so a router that reads the
request rather than the template gives every variant the same task type and the same decision.
"""

from __future__ import annotations

import re
from collections.abc import Callable

Variant = tuple[str, str]  # (layout name, prompt)


def _qa(prompt: str) -> list[Variant]:
    m = re.search(r"(?:PASSAGE|POLICY TEXT):\s*'?(.*?)'?\s*QUESTION:\s*(.*)$", prompt, re.S)
    if not m:
        return []
    passage, question = m.group(1).strip(), m.group(2).strip()
    return [
        (
            "labels renamed",
            f"Context:\n{passage}\n\nQ: {question}\nReply with the shortest answer, or NOT_IN_DOCUMENT "
            "when the context has no answer.",
        ),
        (
            "no labels, document first",
            f"{passage}\n\nBased only on the text above: {question} Give the shortest answer span, or "
            "NOT_IN_DOCUMENT if the text does not say.",
        ),
        (
            "no labels, question first",
            f"{question} Use only the article below; if it is not covered reply NOT_IN_DOCUMENT.\n\n"
            f"{passage}",
        ),
    ]


def _contract(prompt: str) -> list[Variant]:
    m = re.search(r"CONTRACT:\n(.*)\n\nTASK:\n.*?related to \"([^\"]+)\".*?Details:\s*(.*)$", prompt, re.S)
    if m:
        contract, category, details = m.group(1).strip(), m.group(2), m.group(3).strip()
        return [
            (
                "labels renamed",
                f'Document:\n{contract}\n\nInstruction: Find the clauses on "{category}" and return them as '
                f"a JSON array of exact quotes ([] if there are none). {details}",
            ),
            (
                "no labels, document first",
                f"{contract}\n\nFrom the agreement above, return a JSON array of every passage about "
                f'"{category}", quoted exactly, or [] if there is none. {details}',
            ),
            (
                "no labels, ask first",
                f"{details} Return a JSON array of every passage in the agreement below that concerns "
                f'"{category}", quoted exactly, or [] if none does.\n\n{contract}',
            ),
        ]
    m = re.search(r"CONTRACT TEXT:\s*'(.*)'\s*TASK:\s*(.*)$", prompt, re.S)
    if not m:
        return []
    contract, task = m.group(1).strip(), m.group(2).strip()
    return [
        ("labels renamed", f"Agreement:\n{contract}\n\nRequest: {task}"),
        ("no labels, document first", f"{contract}\n\n{task}"),
        ("no labels, ask first", f'{task}\n\n"""\n{contract}\n"""'),
    ]


def _coding(prompt: str) -> list[Variant]:
    m = re.search(r"TASK:\n(.*?)\n\nFUNCTION INTERFACES:\n(.*)$", prompt, re.S)
    if m:  # a described task plus the interface to implement
        task, interface = m.group(1).strip(), m.group(2).strip()
        return [
            (
                "labels renamed",
                "Implement this in Python and reply with the code only.\n\n"
                f"Spec: {task}\n\nSignature:\n{interface}",
            ),
            ("no labels", f"{task} Use this interface:\n\n{interface}\n\nPython only, no explanation."),
        ]
    head, sep, code = prompt.partition("\n\n")
    if sep and re.search(r"^\s*def \w+\(", code, re.M):  # a function to complete from its docstring
        return [
            (
                "fenced",
                f"Finish the function below. Reply with the full code.\n\n```python\n{code.strip()}\n```",
            ),
            ("code first", f"{code.strip()}\n\nImplement the function above so that its docstring holds."),
        ]
    return [
        ("wrapped", f"{prompt}\n\nReply with the code only."),
        ("polite", f"Could you help with this? {prompt}"),
    ]


def _classification(prompt: str) -> list[Variant]:
    m = re.search(r"LABELS:\n(.*?)\n\nQUERY:\n(.*)$", prompt, re.S)
    if not m:
        return [("wrapped", f"{prompt}\n\nNo explanation, please.")]
    labels, query = [ln.strip() for ln in m.group(1).splitlines() if ln.strip()], m.group(2).strip()
    bullets = "\n".join(f"- {label}" for label in labels)
    return [
        (
            "message first",
            f"Message: {query}\n\nWhich category does this message belong to? Options:\n{bullets}\n\n"
            "Answer with just the category.",
        ),
        (
            "inline list",
            f"Categorise the text into one of the following labels: {', '.join(labels)}.\n\n"
            f'Text: "{query}"\n\nOutput only the label.',
        ),
    ]


def _math(prompt: str) -> list[Variant]:
    m = re.search(r"PROBLEM:\n(.*)$", prompt, re.S)
    if not m:
        return [("wrapped", f"{prompt}\n\nShow no working.")]
    problem = m.group(1).strip()
    return [
        ("no labels", f"{problem}\n\nGive just the number."),
        ("labels renamed", f"Question: {problem}\nAnswer with only the number, no units."),
    ]


_BY_USE_CASE: dict[str, Callable[[str], list[Variant]]] = {
    "document_qa": _qa,
    "contract_extraction": _contract,
    "coding": _coding,
    "misc_classification": _classification,
    "misc_math": _math,
}


def variants(use_case: str, prompt: str) -> list[Variant]:
    """Re-worded versions of an eval prompt; empty for a use case with no template to vary."""
    make = _BY_USE_CASE.get(use_case)
    return make(prompt) if make else []
