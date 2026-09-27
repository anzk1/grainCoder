import unittest

from PIL import Image

from segmentation.layoutcoder_mvp import LayoutNode, adapt_layout_tree


class DCGenAdapterTest(unittest.TestCase):
    def test_serialized_fields_are_restricted(self):
        root = LayoutNode((0, 0, 100, 100), "row", [
            LayoutNode((0, 0, 50, 100), "atomic", depth=1),
            LayoutNode((50, 0, 100, 100), "atomic", depth=1),
        ])
        tree = adapt_layout_tree(Image.new("RGB", (100, 100)), root).to_json_tree()

        def assert_shape(node):
            self.assertEqual(set(node), {"bbox", "children"})
            for child in node["children"]:
                assert_shape(child)

        assert_shape(tree)

    def test_atomic_root_gets_synthetic_child(self):
        adapted = adapt_layout_tree(Image.new("RGB", (100, 100)), LayoutNode((0, 0, 100, 100), "atomic"))
        self.assertEqual(len(adapted.children), 1)
        self.assertEqual(adapted.children[0].bbox, adapted.bbox)
        self.assertTrue(adapted.children[0].is_leaf())


if __name__ == "__main__":
    unittest.main()
