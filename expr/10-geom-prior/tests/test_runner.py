import unittest

import torch

from geom_prior.runner import compute_contrastive_text_subspace, parse_args


class ContrastiveTextSubspaceTests(unittest.TestCase):
    def test_subspace_uses_pairwise_activation_differences(self):
        targets = torch.tensor([[3.0, 1.0, 0.0], [1.0, 4.0, 0.0]])
        opposites = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])

        projection = compute_contrastive_text_subspace(targets, opposites, K=2)

        self.assertEqual(projection.shape, (3, 2))
        projected_irrelevant_axis = torch.tensor([0.0, 0.0, 1.0]) @ projection
        self.assertTrue(torch.allclose(projected_irrelevant_axis, torch.zeros(2)))
        self.assertTrue(
            torch.allclose(
                projection.T @ projection,
                torch.eye(2),
                atol=1e-6,
            )
        )

    def test_subspace_rejects_more_dimensions_than_pairs(self):
        targets = torch.ones(1, 3)
        opposites = torch.zeros(1, 3)

        with self.assertRaisesRegex(ValueError, "K must be between 1 and 1"):
            compute_contrastive_text_subspace(targets, opposites, K=2)

    def test_parser_accepts_paired_text_lists(self):
        args = parse_args(
            [
                "--steering-text",
                "precise",
                "factual",
                "--steering-opposite",
                "vague",
                "speculative",
            ]
        )

        self.assertEqual(args.steering_text, ["precise", "factual"])
        self.assertEqual(args.steering_opposite, ["vague", "speculative"])


if __name__ == "__main__":
    unittest.main()
