import unittest

from PIL import Image

from segmentation.layoutcoder_mvp import LayoutNode, adapt_layout_tree
from utils import DCGenGrid


class DCGenIntegrationTest(unittest.TestCase):
    def test_atomic_adapter_produces_replaceable_leaf_container(self):
        adapted = adapt_layout_tree(Image.new("RGB", (100, 80), "white"), LayoutNode((0, 0, 100, 80), "atomic"))
        grid = DCGenGrid(adapted, prompt_seg="leaf", prompt_refine="root")
        leaves = []

        def collect(node):
            if not node["children"]:
                leaves.append(node)
            for child in node["children"]:
                collect(child)

        collect(grid.img_seg_tree)
        self.assertEqual(len(leaves), 1)
        html = grid.code_substitution(grid.html_template, {leaves[0]["id"]: "<span>fixed fragment</span>"})
        self.assertIn("fixed fragment", html)
        self.assertNotEqual(html.strip(), "")


if __name__ == "__main__":
    unittest.main()
