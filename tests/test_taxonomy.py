import json
import unittest

from agentmeter.taxonomy import CAPABILITY_REGISTRY, TAXONOMY_VERSION, Capability


class TaxonomyTests(unittest.TestCase):
    def test_taxonomy_version_constant(self) -> None:
        self.assertEqual(TAXONOMY_VERSION, "0.1.0")

    def test_from_string_exact_match(self) -> None:
        capability = Capability.from_string("extraction.invoice")
        self.assertEqual(capability, Capability.EXTRACTION_INVOICE)

    def test_from_string_unknown_fallback(self) -> None:
        capability = Capability.from_string("not.a.real.capability")
        self.assertEqual(capability, Capability.UNKNOWN)

    def test_from_string_never_raises_for_bad_input(self) -> None:
        capability = Capability.from_string(123)  # type: ignore[arg-type]
        self.assertEqual(capability, Capability.UNKNOWN)

    def test_category_property(self) -> None:
        self.assertEqual(Capability.EXTRACTION_CONTRACT.category, "extraction")
        self.assertEqual(Capability.UNKNOWN.category, "unknown")

    def test_registry_has_all_capabilities(self) -> None:
        self.assertEqual(set(CAPABILITY_REGISTRY.keys()), set(Capability))

    def test_registry_metadata_shape(self) -> None:
        for capability in Capability:
            metadata = CAPABILITY_REGISTRY[capability]
            self.assertIn("description", metadata)
            self.assertIn("typical_latency_ms", metadata)
            self.assertIn("typical_cost_usd", metadata)
            self.assertIsInstance(metadata["description"], str)
            self.assertIsInstance(metadata["typical_latency_ms"], int)
            self.assertIsInstance(metadata["typical_cost_usd"], float)

    def test_registry_is_json_serializable(self) -> None:
        payload = json.dumps(CAPABILITY_REGISTRY)
        self.assertIsInstance(payload, str)
        self.assertIn("extraction.invoice", payload)


if __name__ == "__main__":
    unittest.main()
