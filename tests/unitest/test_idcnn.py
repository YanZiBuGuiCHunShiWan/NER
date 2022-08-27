import unittest

from ner.idcnn.idcnn_crf import IDCNNCRF
from ner.idcnn.idcnn_predictor import  IDCNNPredictor

class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.model_dir = "../../models/idcnn"

    def test_predictor(self):
        predictor = IDCNNPredictor(self.model_dir)
        texts = [
            ['你', '好', '呀'],
            ['一', '马', '当', '先', '就', '是', '好'],
        ]
        labels = predictor.predict(texts)
        print(labels)


if __name__ == '__main__':
    unittest.main()
