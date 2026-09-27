from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from .types import LayoutNode


class DCGenSegmentationNode:
    def __init__(self, image: Image.Image, bbox: tuple[int, int, int, int], children=None):
        self.img = image
        self.bbox = bbox
        self.children = children or []
        self.depth = self.get_depth()

    def get_img(self, cut_out: bool = False, outline=(0, 255, 0)) -> Image.Image:
        if cut_out:
            return self.img.crop(self.bbox)
        drawn = self.img.copy()
        ImageDraw.Draw(drawn).rectangle(self.bbox, outline=outline, width=5)
        return drawn

    def is_leaf(self) -> bool:
        return not self.children

    def get_depth(self) -> int:
        return 1 if not self.children else 1 + max(child.get_depth() for child in self.children)

    def to_json_tree(self, path: str | None = None) -> dict:
        tree = {
            "bbox": list(self.bbox),
            "children": [child.to_json_tree() for child in self.children],
        }
        if path:
            Path(path).write_text(json.dumps(tree, indent=2), encoding="utf-8")
        return tree

    def display_tree(self, save_path: str | None = None) -> Image.Image:
        drawn = self.img.copy()
        image_draw = ImageDraw.Draw(drawn)

        def draw_node(node: "DCGenSegmentationNode") -> None:
            for child in node.children:
                image_draw.rectangle(child.bbox, outline=(0, 255, 0), width=3)
                draw_node(child)

        draw_node(self)
        if save_path:
            drawn.save(save_path)
        return drawn


def adapt_layout_tree(image: Image.Image, layout_root: LayoutNode) -> DCGenSegmentationNode:
    def adapt(node: LayoutNode) -> DCGenSegmentationNode:
        return DCGenSegmentationNode(image, node.bbox, [adapt(child) for child in node.children])

    adapted = adapt(layout_root)
    if adapted.is_leaf():
        adapted.children = [DCGenSegmentationNode(image, adapted.bbox)]
        adapted.depth = adapted.get_depth()
    return adapted
