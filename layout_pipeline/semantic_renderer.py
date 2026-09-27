from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup, Tag

from .schema import LayoutNode, atomic_nodes


@dataclass(frozen=True)
class AtomicSnippetAudit:
    atomic_id: int
    accepted: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return {"atomic_id": self.atomic_id, "accepted": self.accepted, "reasons": list(self.reasons)}


@dataclass(frozen=True)
class FinalLayoutRender:
    html: str
    audits: tuple[AtomicSnippetAudit, ...]


class SemanticLayoutRenderer:
    def render_placeholders(self, root: LayoutNode) -> str:
        return self._document(root, {}, True)[0]

    def render_final(self, root: LayoutNode, snippets: dict[int, str]) -> FinalLayoutRender:
        html, audits = self._document(root, snippets, False)
        return FinalLayoutRender(html, audits)

    def sanitize_snippet(self, snippet: str) -> tuple[str, list[str]]:
        return self._sanitize_fragment(snippet)

    def _document(
        self,
        root: LayoutNode,
        snippets: dict[int, str],
        placeholder: bool,
    ) -> tuple[str, tuple[AtomicSnippetAudit, ...]]:
        audits: list[AtomicSnippetAudit] = []
        body = self._node_html(root, snippets, placeholder, audits, True)
        html = (
            "<!DOCTYPE html><html><head><meta charset=\"UTF-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            "<script src=\"https://cdn.tailwindcss.com\"></script>"
            "<style>html,body{margin:0;padding:0;background:#fff;}"
            "*,*::before,*::after{box-sizing:border-box;}"
            ".layout-node{min-width:0;min-height:0;overflow:hidden;}"
            ".layout-atomic-content{width:100%;height:100%;overflow:hidden;}"
            "</style></head><body>"
            f"{body}</body></html>"
        )
        return html, tuple(audits)

    def _node_html(
        self,
        node: LayoutNode,
        snippets: dict[int, str],
        placeholder: bool,
        audits: list[AtomicSnippetAudit],
        root: bool,
    ) -> str:
        width = node.width
        height = node.height
        declarations = ["box-sizing:border-box", "overflow:hidden", f"flex-grow:{node.portion}", "flex-basis:0"]
        if root:
            declarations.extend((f"width:{width}px", f"height:{height}px", "flex-basis:auto"))
        if node.layout_type == "row":
            declarations.extend(("display:flex", "flex-direction:row"))
        elif node.layout_type == "column":
            declarations.extend(("display:flex", "flex-direction:column"))
        attributes = f'class="layout-node layout-{node.layout_type}" data-layout-id="{node.id}" data-layout-type="{node.layout_type}"'
        style = ";".join(declarations)
        if node.layout_type == "atomic":
            content = self._placeholder(node) if placeholder else self._atomic_content(node, snippets, audits)
            return f'<div {attributes} data-atomic-root="true" style="{style}"><div class="layout-atomic-content">{content}</div></div>'
        if node.layout_type == "spacer":
            return f'<div {attributes} aria-hidden="true" style="{style}"></div>'
        children = "".join(self._node_html(child, snippets, placeholder, audits, False) for child in node.children)
        return f'<div {attributes} style="{style}">{children}</div>'

    @staticmethod
    def _placeholder(node: LayoutNode) -> str:
        return (
            '<div style="width:100%;height:100%;display:flex;align-items:center;justify-content:center;'
            'border:1px solid rgba(0,0,0,.25);background:rgba(127,127,127,.08);font:12px sans-serif">'
            f"atomic {node.id}</div>"
        )

    def _atomic_content(
        self,
        node: LayoutNode,
        snippets: dict[int, str],
        audits: list[AtomicSnippetAudit],
    ) -> str:
        snippet = snippets.get(node.id, "")
        sanitized, reasons = self._sanitize_fragment(snippet)
        accepted = not reasons
        audits.append(AtomicSnippetAudit(node.id, accepted, tuple(reasons)))
        return sanitized if accepted else ""

    @staticmethod
    def _sanitize_fragment(snippet: str) -> tuple[str, list[str]]:
        cleaned = re.sub(r"```html\s*", "", snippet, flags=re.IGNORECASE)
        cleaned = re.sub(r"```(?![A-Za-z0-9_])", "", cleaned)
        reasons: list[str] = []
        if "```" in cleaned:
            reasons.append("markdown_fence")
        if re.search(r"position\s*:\s*(fixed|absolute)", cleaned, re.IGNORECASE):
            reasons.append("forbidden_position")
        soup = BeautifulSoup(cleaned, "html.parser")
        for name in ("html", "head", "body"):
            for document_tag in soup.find_all(name):
                document_tag.unwrap()
        for tag in soup.find_all(True):
            if not isinstance(tag, Tag):
                continue
            style = tag.get("style", "")
            if re.search(r"(?:^|;)\s*(width|height)\s*:\s*(?:100vw|100vh)", style, re.IGNORECASE):
                reasons.append("viewport_sized_fragment")
                break
        return str(soup), reasons


def leaf_nodes(root: LayoutNode) -> tuple[LayoutNode, ...]:
    return tuple(node for node in atomic_nodes(root) if node.element_ids)


def parent_by_leaf_id(root: LayoutNode) -> dict[int, LayoutNode]:
    parents: dict[int, LayoutNode] = {}

    def visit(node: LayoutNode) -> None:
        for child in node.children:
            if child.layout_type == "atomic":
                parents[child.id] = node
            visit(child)

    visit(root)
    return parents
