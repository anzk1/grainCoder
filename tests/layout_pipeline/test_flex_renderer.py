import unittest

from layout_pipeline.schema import LayoutNode
from layout_pipeline.semantic_renderer import SemanticLayoutRenderer


class FlexRendererTest(unittest.TestCase):
    def test_renders_portions_through_nested_flex(self):
        root = LayoutNode(
            0,
            (0, 0, 100, 50),
            "row",
            (
                LayoutNode(1, (0, 0, 30, 50), "atomic", portion=3, element_ids=(0,)),
                LayoutNode(2, (30, 0, 100, 50), "atomic", portion=7, element_ids=(1,)),
            ),
        )

        html = SemanticLayoutRenderer().render_final(root, {1: "<p>A</p>", 2: "<p>B</p>"}).html

        self.assertIn("flex-direction:row", html)
        self.assertIn("flex-grow:3", html)
        self.assertIn("flex-grow:7", html)
        self.assertNotIn("position:absolute", html)

    def test_sanitizes_fences_and_document_wrappers(self):
        renderer = SemanticLayoutRenderer()
        cases = {
            "leading": "```html\n<div>Leading</div>",
            "trailing": "<div>Trailing</div>\n```",
            "middle": "<div>Before</div>```html<div>After</div>```",
            "nested": "<html><head><style>.x{color:red}</style></head><body><div>Body</div></body></html>",
            "plain": "<section>Plain</section>",
        }
        for name, snippet in cases.items():
            with self.subTest(name=name):
                sanitized, reasons = renderer.sanitize_snippet(snippet)
                self.assertEqual(reasons, [])
                self.assertNotIn("```", sanitized)
                self.assertNotIn("<html", sanitized)
                self.assertNotIn("<head", sanitized)
                self.assertNotIn("<body", sanitized)

    def test_rejects_unrecognized_markdown_fence(self):
        sanitized, reasons = SemanticLayoutRenderer().sanitize_snippet("```css\n.x{}\n```")

        self.assertIn("markdown_fence", reasons)
        self.assertIn("```css", sanitized)

    def test_spacer_preserves_flex_geometry_without_atomic_generation(self):
        root = LayoutNode(
            0,
            (0, 0, 100, 50),
            "row",
            (
                LayoutNode(1, (0, 0, 25, 50), "spacer", portion=1),
                LayoutNode(2, (25, 0, 100, 50), "atomic", portion=3, element_ids=(0,)),
            ),
        )

        html = SemanticLayoutRenderer().render_final(root, {2: "<p>Content</p>"}).html

        self.assertIn("layout-spacer", html)
        self.assertEqual(html.count('data-atomic-root="true"'), 1)


if __name__ == "__main__":
    unittest.main()
