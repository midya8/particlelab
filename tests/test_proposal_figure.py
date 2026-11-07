import copy
import html
import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
from examples.proposal_audit import audit
from examples.render_proposal import manifest,render


class ProposalFigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.audit=audit();cls.svg=render(cls.audit)
    def test_all_64_points_and_four_polylines(self):
        root=ET.fromstring(self.svg);ns={'s':'http://www.w3.org/2000/svg'};circles=root.findall('s:circle',ns);self.assertEqual(len(circles),64)
        expected=manifest(self.audit)
        for point in circles:
            i=int(point.attrib['data-panel']);seed=int(point.attrib['data-seed']);method=point.attrib['data-series'];v=expected['experiments'][i]['per_seed'][seed][method]['mean_rmse']
            self.assertEqual(float(point.attrib['data-value']),v)
            self.assertEqual(float(point.attrib['cx']),round([80,625][i]+seed/15*435,3))
            self.assertEqual(float(point.attrib['cy']),round(340-v/[.16,.1][i]*190,3))
        lines=root.findall('s:polyline',ns);self.assertEqual(len(lines),4)
        for line in lines:
            match=[c for c in circles if c.attrib['data-panel']==line.attrib['data-panel'] and c.attrib['data-series']==line.attrib['data-series']]
            self.assertEqual(line.attrib['points'],' '.join(c.attrib['cx']+','+c.attrib['cy'] for c in match))
    def test_metadata_and_reproducibility(self):
        root=ET.fromstring(self.svg);metadata=root.find('{http://www.w3.org/2000/svg}metadata')
        self.assertEqual(json.loads(metadata.text),manifest(self.audit));self.assertEqual(self.svg,render(audit()))
        self.assertEqual(self.svg,Path('assets/proposal-audit.svg').read_text())
    def test_rejects_changed_fixture_missing_seed_metric_and_domain(self):
        for change in ['noise','seed','average','domain','check']:
            v=copy.deepcopy(self.audit)
            if change=='noise':v['experiments'][0]['settings']['observation_std']=.1
            elif change=='seed':v['experiments'][0]['per_seed'].pop()
            elif change=='average':v['experiments'][0]['averages']['bootstrap']['mean_rmse']=0
            elif change=='domain':
                v['experiments'][0]['per_seed'][0]['bootstrap']['mean_rmse']=1
                v['experiments'][0]['averages']['bootstrap']['mean_rmse']=sum(r['bootstrap']['mean_rmse'] for r in v['experiments'][0]['per_seed'])/16
            else:v['checks']['finite_metrics']=False
            with self.assertRaises(ValueError):render(v)
    def test_render_does_not_mutate_input(self):
        original=copy.deepcopy(self.audit);render(self.audit);self.assertEqual(original,self.audit)
