from backend.train import read_pairs,ROOT
import json
def test_splits_count():
 m=json.loads((ROOT/'artifacts/metrics/results.json').read_text());assert sum(v['patients'] for v in m['split'].values())==m['source_patients']
 assert sum(v['pairs'] for v in m['split'].values())==m['usable_pairs']
def test_name_not_in_source():
 import pandas as pd
 assert 'name' not in pd.read_excel(ROOT/'data/source_anonymized.xlsx',nrows=0).columns
