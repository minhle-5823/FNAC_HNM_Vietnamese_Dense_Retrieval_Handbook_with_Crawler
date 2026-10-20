"""Regression tests for Vietnamese compound preservation in lexical indexing."""
import unittest

from core.retrieval_lab import _lexical_token, tokenize_strategies


class FakeUnderthesea:
    backend = 'underthesea'

    def tokenize(self, text):
        return ['đại_học', 'khoa_học_tự_nhiên', ',', 'đăng_ký', 'và',
                'dịch_vụ', 'không', '5g', '_bạn', '_', 'xét_tuyển']


class VietnameseCompoundTests(unittest.TestCase):
    def test_preserves_compounds_across_all_strategies(self):
        groups = tokenize_strategies(FakeUnderthesea(), '', lambda w: w, lambda w: w)
        for name in ('baseline', 'no_stopwords', 'stemmed', 'lemmatized'):
            for term in ('đại_học', 'khoa_học_tự_nhiên', 'đăng_ký', 'dịch_vụ', 'xét_tuyển'):
                with self.subTest(strategy=name, term=term):
                    self.assertIn(term, groups[name])
            self.assertIn('không', groups[name])
            self.assertIn('5g', groups[name])
            self.assertNotIn(',', groups[name])
            self.assertNotIn('_', groups[name])
        self.assertIn('và', groups['baseline'])
        self.assertNotIn('và', groups['no_stopwords'])

    def test_lexical_token_filter(self):
        for bad in (',', '.', '/', '_', '__', 'word__broken', 'abc-xyz'):
            self.assertIsNone(_lexical_token(bad))
        self.assertEqual('đại_học', _lexical_token('đại_học'))
        self.assertEqual('khoa_học_tự_nhiên', _lexical_token('khoa_học_tự_nhiên'))
        self.assertEqual('bạn', _lexical_token('_bạn'))
        self.assertEqual('5g', _lexical_token('5g'))


if __name__ == '__main__':
    unittest.main()
