import unittest

from layout_pipeline.schema import ElementNode, LayoutNode
from layout_pipeline.structure_context import LeafStructureContextBuilder


class StructureContextTest(unittest.TestCase):
    def test_uses_canonical_atomic_ownership(self):
        atomic = LayoutNode(1, (0, 0, 50, 50), "atomic", element_ids=(3,), ownership_label="header")
        parent = LayoutNode(0, (0, 0, 100, 50), "row", (atomic,))
        context = LeafStructureContextBuilder(parent, [ElementNode(3, (5, 5, 20, 20), "Text")])

        prompt = context.build(atomic, parent)

        self.assertIn("ownership=header", prompt)
        self.assertIn("'Text': 1", prompt)


if __name__ == "__main__":
    unittest.main()
