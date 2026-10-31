import unittest
import numpy as np
from landscape_audit import matched_contexts, comparison_effects, multilabel_metrics, paired_group_intervals


class LandscapeTests(unittest.TestCase):
    def test_unmatched_samples_do_not_define_a_coordinate_profile(self):
        self.assertEqual(matched_contexts([{'a': 0, 'b': 0}, {'a': 1, 'b': 1}], 'a'), [])

    def test_equal_score_can_hide_changed_labels(self):
        p = np.array([[[[1, 0]], [[0, 1]]]])
        scores = np.array([[[.5, .5, .5], [.5, .5, .5]]])
        r = comparison_effects(p, scores, [(0, 1)], 'policy')
        self.assertEqual(r['sameJointButChangedWholeTagVectors'], 1)
        self.assertIsNone(comparison_effects(p, scores, [], 'policy')['maxAbsoluteJointDifference'])

    def test_multilabel_exact_match_is_not_per_label_accuracy(self):
        y, p = [[1, 1], [1, 0]], [[1, 0], [1, 0]]
        r = multilabel_metrics(y, p, [1, 1], ['a', 'b'], ['x', 'y'])
        self.assertEqual(r['exactMatch'], .5)
        self.assertEqual(r['micro']['precision'], 1.)
        self.assertAlmostEqual(r['micro']['recall'], 2/3)
        self.assertEqual(r['perClass']['b']['positiveSongGroups'], 1)

    def test_paired_bootstrap_preserves_an_identical_control(self):
        y = np.array([[1, 0], [0, 1], [1, 1]])
        r = paired_group_intervals(y, {'a': y, 'b': y}, np.ones(3), ['s', 's', 't'], draws=20)
        self.assertTrue(all(v == [0., 0.] for v in r['delta95'].values()))

    def test_annotation_pairs_preserve_their_declared_controls(self):
        from make_annotation_pack import cases
        from collections import Counter
        c = cases()
        self.assertEqual(Counter(lane for _, lane, _ in c['A01']), Counter(lane for _, lane, _ in c['A02']))
        self.assertEqual([t for t, _, _ in c['A01']], [t for t, _, _ in c['A02']])
        self.assertEqual([(t, lane) for t, lane, _ in c['B01']], [(t, lane) for t, lane, _ in c['B02']])
        area = lambda rows: sum(end-t for t, _, end in rows if end is not None)
        self.assertEqual(area(c['B01']), area(c['B02']))
        for rows in c.values():
            self.assertTrue(all(0 <= lane < 7 and (end is None or end > t) for t, lane, end in rows))


if __name__ == '__main__':
    unittest.main()
