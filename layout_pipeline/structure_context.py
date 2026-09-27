from __future__ import annotations

from .schema import ElementNode, LayoutNode


class LeafStructureContextBuilder:
    def __init__(self, root: LayoutNode, elements: list[ElementNode] | tuple[ElementNode, ...] = ()):
        self.root = root
        self.elements = {element.id: element for element in elements}

    def build(self, leaf: LayoutNode, parent: LayoutNode | None) -> str:
        owned = [self.elements[element_id] for element_id in leaf.element_ids if element_id in self.elements]
        class_counts: dict[str, int] = {}
        for element in owned:
            class_counts[element.uied_class] = class_counts.get(element.uied_class, 0) + 1
        parent_type = parent.layout_type if parent is not None else "none"
        return (
            f"Page structure: parent={parent_type}; atomic_id={leaf.id}; "
            f"bbox={list(leaf.bbox)}; portion={leaf.portion}; ownership={leaf.ownership_label}; "
            f"elements={len(owned)}; classes={class_counts}. "
            "Generate only the visual content inside this atomic region. "
            "Do not recreate the page, neighboring regions, or outer layout."
        )
