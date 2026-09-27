from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from .types import ElementBox, LayoutNode


RAW_COLOR = (128, 128, 128)
NORMALIZED_COLOR = (0, 102, 255)
SPLIT_COLORS = [(255, 0, 0), (255, 140, 0)]
LEAF_COLOR = (0, 180, 0)


def draw_boxes(image: Image.Image, boxes: list[ElementBox], color=NORMALIZED_COLOR, width: int = 2) -> Image.Image:
    drawn = image.copy().convert("RGB")
    canvas = ImageDraw.Draw(drawn)
    for box in boxes:
        canvas.rectangle(box.bbox, outline=color, width=width)
    return drawn


def draw_layout_tree(image: Image.Image, root: LayoutNode) -> Image.Image:
    drawn = image.copy().convert("RGB")
    canvas = ImageDraw.Draw(drawn)

    def draw_node(node: LayoutNode) -> None:
        if node.children and node.split_axis and node.split_position is not None:
            color = SPLIT_COLORS[min(node.depth, len(SPLIT_COLORS) - 1)]
            if node.split_axis == "x":
                canvas.line((node.split_position, node.bbox[1], node.split_position, node.bbox[3]), fill=color, width=3)
            else:
                canvas.line((node.bbox[0], node.split_position, node.bbox[2], node.split_position), fill=color, width=3)
        if not node.children:
            canvas.rectangle(node.bbox, outline=LEAF_COLOR, width=2)
        for child in node.children:
            draw_node(child)

    draw_node(root)
    return drawn


def save_image(image: Image.Image, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)
