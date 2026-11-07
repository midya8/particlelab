import copy
import importlib.util
import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET


def load(name):
    s=importlib.util.spec_from_file_location(name,'examples/'+name+'.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


class SmoothingFigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.audit=load('smoothing_audit').audit();cls.renderer=load('render_smoothing');cls.svg=cls.renderer.render(cls.audit)
    def test_all_points_coordinates_and_values(self):
        root=ET.fromstring(self.svg);m=self.renderer.manifest(self.audit);circles=root.findall('{*}circle');self.assertEqual(len(circles),292)
        for c in circles:
            name=c.attrib['data-series'];i=int(c.attrib['data-index']);v=m['series'][name][i]
            if name.endswith('mean'):x,n,lo,hi=65,12,-.6,.8
            elif name.endswith('variance'):x,n,lo,hi=430,64,0,.02
            else:x,n,lo,hi=795,64,0,32
            self.assertEqual(float(c.attrib['data-value']),v);self.assertAlmostEqual(float(c.attrib['cx']),x+i/(n-1)*280,places=3);self.assertAlmostEqual(float(c.attrib['cy']),380-(v-lo)/(hi-lo)*220,places=3)
    def test_metadata_and_regeneration(self):
        root=ET.fromstring(self.svg);self.assertEqual(json.loads(root.find('{*}metadata').text),self.renderer.manifest(self.audit));self.assertEqual(self.svg,Path('assets/smoothing-audit.svg').read_text())
        self.assertEqual(len(root.findall('{*}polyline')),7)
    def test_changed_or_incomplete_fixture_fails(self):
        for mutate in [lambda s:s['collapse']['settings']['data'].pop(),lambda s:s['linear']['all_seed_errors'].pop(),lambda s:s['collapse']['run']['smoothing_history'].pop(),lambda s:s['linear']['seed0']['smoothing_history'][0].update(mean=10),lambda s:s['collapse']['run']['filter']['settings'].update(seed=4)]:
            v=copy.deepcopy(self.audit);mutate(v)
            with self.assertRaises(ValueError):self.renderer.render(v)
    def test_no_alias_and_limits(self):
        before=copy.deepcopy(self.audit);m=self.renderer.manifest(self.audit);m['series']['genealogy_mean'][0]=999;self.assertEqual(before,self.audit)
        for text in ['not backward simulation','Fixed development fixtures','No independent-path','not to real-world truth']:self.assertIn(text,self.svg)
