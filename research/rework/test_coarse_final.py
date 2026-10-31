import tempfile,unittest
from unittest.mock import patch
from coarse_final import choose,np,Path,write
from coarse_model import FINE


class FinalSelectionTest(unittest.TestCase):
    def test_uses_all_repeated_inner_predictions_but_never_outer_winners(self):
        weak=[{'uid':f'{k}/{tag}','keys':k,'coarseTags':[tag],'recordWeight':1.}for k in [4,7]for tag in FINE]
        manual=[{'uid':f'manual/{i}'}for i in range(6)];plans=[]
        with tempfile.TemporaryDirectory(prefix='spm-final-selection-')as tmp:
            root=Path(tmp)
            for o in range(6):
                train=[i for i in range(10)if i%6!=o];strong=[i for i in range(6)if i!=o];inner=[]
                for j in range(3):
                    va=train[j::3];sv=strong[j::3];inner.append({'validation':va,'strongValidation':sv})
                    d=root/'fake'/'both'/str(o)/str(j)
                    write(d/'complete.json',{'mapUids':[weak[i]['uid']for i in va],'strongUids':[manual[i]['uid']for i in sv]})
                    y=[FINE.index(weak[i]['coarseTags'][0])for i in va]
                    first=[t if weak[i]['keys']==4 else 0 for i,t in zip(va,y)]
                    np.savez(d/'inner-predictions.npz',map=np.array([first,y]),segmentAccuracy=np.array([[0.]*len(sv),[1.]*len(sv)]))
                plans.append({'inner':inner})
            # Deliberately unreadable outer material cannot affect this choice.
            (root/'outer-predictions.json').write_text('INVALID OUTER DATA')
            with patch('coarse_final.configurations',return_value=[{},{}]):
                four,_,scope=choose(root,{'variant':'fake','mode':'both','criterion':'4k'},weak,manual,plans)
                shared,_,_=choose(root,{'variant':'fake','mode':'both','criterion':'shared'},weak,manual,plans)
            self.assertEqual(four['configId'],0);self.assertEqual(shared['configId'],1)
            self.assertEqual(scope['weakPredictionRows'],50);self.assertEqual(scope['heldOutInnerPredictionsPerRecord'],5)


if __name__=='__main__':unittest.main()
