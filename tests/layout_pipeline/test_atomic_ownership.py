import unittest

from layout_pipeline.atomic_ownership import assign_atomic_ownership, replace_empty_atomics_with_spacers
from layout_pipeline.schema import ElementNode, LayoutNode, atomic_nodes


class AtomicOwnershipTest(unittest.TestCase):
    def test_assigns_each_element_once_and_absorbs_icon_only_region(self):
        root = LayoutNode(
            0,
            (0, 0, 100, 100),
            "row",
            (
                LayoutNode(1, (0, 0, 50, 100), "atomic"),
                LayoutNode(2, (50, 0, 100, 100), "atomic"),
            ),
        )
        elements = [
            ElementNode(0, (10, 10, 30, 25), "Text", "Title"),
            ElementNode(1, (65, 45, 80, 60), "Compo"),
        ]

        owned = assign_atomic_ownership(root, elements, (100, 100), icon_max_size=48)
        atomics = atomic_nodes(owned)

        self.assertEqual(atomics[0].element_ids, (0, 1))
        self.assertEqual(atomics[1].element_ids, ())

    def test_icon_regions_do_not_absorb_each_other(self):
        root = LayoutNode(
            0,
            (0, 0, 150, 100),
            "row",
            (
                LayoutNode(1, (0, 0, 50, 100), "atomic"),
                LayoutNode(2, (50, 0, 100, 100), "atomic"),
                LayoutNode(3, (100, 0, 150, 100), "atomic"),
            ),
        )
        elements = [
            ElementNode(0, (10, 10, 30, 25), "Text", "Title"),
            ElementNode(1, (65, 45, 80, 60), "Compo"),
            ElementNode(2, (115, 45, 130, 60), "Compo"),
        ]

        owned = assign_atomic_ownership(root, elements, (150, 100), icon_max_size=48)
        atomics = atomic_nodes(owned)

        self.assertEqual(atomics[0].element_ids, (0, 1, 2))
        self.assertEqual(atomics[1].element_ids, ())
        self.assertEqual(atomics[2].element_ids, ())

    def test_converts_empty_owned_atomic_to_layout_only_spacer(self):
        root = LayoutNode(
            0,
            (0, 0, 100, 100),
            "row",
            (
                LayoutNode(1, (0, 0, 50, 100), "atomic"),
                LayoutNode(2, (50, 0, 100, 100), "atomic", element_ids=(0,)),
            ),
        )

        converted = replace_empty_atomics_with_spacers(root)

        self.assertEqual(converted.children[0].layout_type, "spacer")
        self.assertEqual(converted.children[1].layout_type, "atomic")


if __name__ == "__main__":
    unittest.main()
