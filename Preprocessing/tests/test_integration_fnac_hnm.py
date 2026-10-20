import tempfile
import unittest
from pathlib import Path
import pandas as pd
from integration_fnac_hnm import load_auxiliary,extend_mining_corpus


class IntegrationBridgeTests(unittest.TestCase):
    def test_preserve_vire_doc_id_and_append_only_nonduplicates(self):
        base=pd.DataFrame({'doc_id':['d0','d1'],'context':['Đăng ký Internet', 'Thông báo học bổng']})
        aux=pd.DataFrame({'doc_id':['web_a','web_b'],'context':['  đăng ký  internet ','Cách đổi mật khẩu WiFi']})
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'aux.csv';aux.to_csv(f,index=False)
            merged=extend_mining_corpus(base,f)
        self.assertEqual(merged.doc_id.tolist(),['d0','d1','web_b'])
        self.assertEqual(merged.context.iloc[-1],'Cách đổi mật khẩu WiFi')

    def test_fail_on_colliding_ids(self):
        base=pd.DataFrame({'doc_id':['web_a'],'context':['A']})
        aux=pd.DataFrame({'doc_id':['web_a'],'context':['B']})
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'aux.csv';aux.to_csv(f,index=False)
            with self.assertRaises(ValueError):extend_mining_corpus(base,f)

    def test_887_real_auxiliary_documents(self):
        actual=load_auxiliary()
        self.assertEqual(len(actual),887)
        self.assertEqual(actual.columns.tolist(),['doc_id','context'])

if __name__=='__main__':unittest.main()
