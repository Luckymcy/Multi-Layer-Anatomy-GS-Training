import tempfile
from pathlib import Path
import unittest
from appearance.views import split_view_indices, epoch_view_index


class ViewTests(unittest.TestCase):
    def test_split_uses_names_not_json_order_and_rejects_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            a,b=Path(directory)/'train.txt',Path(directory)/'test.txt'
            a.write_text('a.png\nc.png\n'); b.write_text('b.png\n')
            cameras=[{'img_name':n} for n in ['b','c','a']]
            self.assertEqual(split_view_indices(cameras,a,b),([1,2],[0]))
            b.write_text('a.png\n')
            with self.assertRaises(ValueError): split_view_indices(cameras,a,b)

    def test_epoch_covers_each_view_once_and_resume_uses_absolute_step(self):
        for epoch in range(3):
            selected=[epoch_view_index(epoch*165+i+1,165,42) for i in range(165)]
            self.assertEqual(sorted(selected),list(range(165)))
        continuous=[epoch_view_index(i,165,42) for i in range(1,601)]
        self.assertEqual(continuous[500:],[epoch_view_index(i,165,42) for i in range(501,601)])
