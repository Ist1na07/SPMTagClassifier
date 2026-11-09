import unittest
import numpy as np
from cut_consistency_model import World, Packed, TYPES, INDEX, OTHER, probability, frames, read


class CutConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = World()
        plan = read('research/rework/results/mil-stage1-r4/protocol.json')['plans'][0]['inner'][0]
        cls.train = [cls.world.main[i] for i in plan['train']]
        cls.manual = [cls.world.manual[i] for i in plan['strongTrain']]
        cls.packs = {k: Packed(cls.world, cls.train, cls.manual, kind=k) for k in ['bias', 'hand', 'rhythm', 'both']}
        cls.cfg = dict(localL2=.003, strongWeight=1., conceptWeight=1., effortPower=1., weakBalance=True, primaryPower=4.)

    def test_gradient_including_partial_not_cs_constraint(self):
        rng = np.random.default_rng(20261108)
        for p in self.packs.values():
            theta = rng.normal(0, .2, p.dimension); direction = rng.normal(size=p.dimension)
            _, gradient = p.objective(theta, self.cfg)
            for step in [1e-5, 3e-6]:
                numerical = (p.objective(theta + step * direction, self.cfg)[0] - p.objective(theta - step * direction, self.cfg)[0]) / (2 * step)
                self.assertAlmostEqual(numerical, gradient @ direction, places=6)

    def test_zero_exact_and_other_classes_and_cs_ratio_preserved(self):
        w = self.world; p = self.packs['both']
        for uid in ['concept/H01', 'concept/H02', 'local/F01', 'local/F02', self.train[0]['uid']]:
            zero = dict(kind='both', scaler=p.scaler, theta=np.zeros(p.dimension).tolist())
            np.testing.assert_array_equal(probability(w, uid, zero), w.teacher(uid))
            self.assertEqual(frames(w, uid, zero), w.inputs[uid]['frames'])
            model = {**zero, 'theta': np.linspace(-.4, .4, p.dimension).tolist()}
            result = probability(w, uid, model)
            np.testing.assert_array_equal(result[:, OTHER], w.teacher(uid)[:, OTHER])
            np.testing.assert_allclose(result[:, INDEX].sum(1), w.teacher(uid)[:, INDEX].sum(1), atol=1e-15)
            np.testing.assert_allclose(result[:, INDEX[0]] * w.teacher(uid)[:, INDEX[1]],
                result[:, INDEX[1]] * w.teacher(uid)[:, INDEX[0]], atol=1e-15)
            before, after = w.inputs[uid]['frames'], frames(w, uid, model)
            for a, b in zip(before, after):
                for key, value in a['scores'].items():
                    if key not in TYPES:
                        self.assertEqual(value, b['scores'][key])
                self.assertEqual(a['family'], b['family'])

    def test_not_cs_supervision_is_not_an_exhaustive_tech_label(self):
        p = self.packs['both']; cfg = {**self.cfg, 'strongWeight': 0., 'localL2': 0.}
        theta = np.zeros(p.dimension)
        loss_on = p.objective(theta, cfg)[0]
        loss_off = p.objective(theta, {**cfg, 'conceptWeight': 0.})[0]
        expected = 0.
        for index, weight in p.negative:
            not_cs = p.teacher[index].sum() - p.teacher[index, INDEX[:2]].sum()
            expected -= weight * np.log(not_cs)
        self.assertAlmostEqual(loss_on - loss_off, expected, places=12)
        self.assertEqual(p.concepts, [])
        self.assertEqual(p.ranks, [])

    def test_nested_train_embeddings_and_objectives(self):
        target = self.packs['both']
        for source in self.packs.values():
            def names(p):
                return ['intercept'] + [k for k, keep in zip(p.scaler['expandedKeys'], p.scaler['keep']) if keep]
            source_names, target_names = names(source), names(target)
            theta = np.linspace(-.1, .2, source.dimension); lifted = np.zeros(target.dimension)
            for i, key in enumerate(source_names):
                lifted[target_names.index(key)] = theta[i]
            np.testing.assert_allclose(source.x @ theta, target.x @ lifted, atol=1e-14)
            self.assertAlmostEqual(source.objective(theta, self.cfg)[0], target.objective(lifted, self.cfg)[0], places=12)


if __name__ == '__main__':
    unittest.main()
