import copy
import importlib.util
import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET


def load(name):
    spec=importlib.util.spec_from_file_location(name,'examples/'+name+'.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


class BackwardFigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.audit=load('backward_audit').audit();cls.renderer=load('render_backward');cls.svg=cls.renderer.render(cls.audit)
    def test_every_value_and_coordinate(self):
        root=ET.fromstring(self.svg);manifest=self.renderer.manifest(self.audit);points=root.findall('{*}circle');self.assertEqual(len(points),320)
        for p in points:
            name=p.attrib['data-series'];i=int(p.attrib['data-index']);v=manifest['series'][name][i]
            x,hi=(80,.03) if name.endswith('variance') else (625,32)
            self.assertEqual(float(p.attrib['data-value']),v);self.assertAlmostEqual(float(p.attrib['cx']),x+i/63*435,places=3);self.assertAlmostEqual(float(p.attrib['cy']),380-v/hi*230,places=3)
    def test_metadata_regeneration_and_qualification(self):
        root=ET.fromstring(self.svg);self.assertEqual(json.loads(root.find('{*}metadata').text),self.renderer.manifest(self.audit));self.assertEqual(self.svg,Path('assets/backward-audit.svg').read_text());self.assertEqual(len(root.findall('{*}polyline')),5)
        for text in ['not a policy or budget ranking','different sampling definitions','More diversity is not a guarantee','cannot create missing state support']:self.assertIn(text,self.svg)
    def test_changed_incomplete_or_outside_domain_rejected(self):
        for mutate in [lambda v:v['settings'].update(draw_seed=4),lambda v:v['backward']['smoothing_history'].pop(),lambda v:v['backward']['smoothing_history'][0].update(variance=.1),lambda v:v['backward']['filter']['settings'].update(seed=5)]:
            data=copy.deepcopy(self.audit);mutate(data)
            with self.assertRaises(ValueError):self.renderer.render(data)
    def test_audit_checks_and_no_mutation(self):
        before=copy.deepcopy(self.audit);self.renderer.render(self.audit);self.assertEqual(before,self.audit);self.assertTrue(all(self.audit['checks'].values()));self.assertEqual(self.audit['backward']['filter'],self.audit['genealogy']['filter'])
