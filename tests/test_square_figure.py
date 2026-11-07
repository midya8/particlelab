import copy
import importlib.util
import json
import math
import pathlib
import unittest
import xml.etree.ElementTree as ET

ROOT=pathlib.Path(__file__).resolve().parents[1]

def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'examples'/f'{name}.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

RENDER=load('render_square');AUDIT=load('square_audit');NS={'s':'http://www.w3.org/2000/svg'}


class SquareFigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.audit=AUDIT.audit();cls.svg=RENDER.render(cls.audit);cls.tree=ET.fromstring(cls.svg)
    def test_committed_regeneration_metadata_and_no_mutation(self):
        before=copy.deepcopy(self.audit);self.assertEqual(RENDER.render(self.audit),self.svg);self.assertEqual(self.audit,before)
        self.assertEqual(self.svg,(ROOT/'assets/square-symmetry.svg').read_text())
        data=json.loads(self.tree.find('s:metadata',NS).text)
        self.assertEqual(data['reference'],self.audit['reference']);self.assertEqual(data['runs'],self.audit['runs']);self.assertEqual(len(data['curve']),101)
        self.assertEqual(self.tree.attrib['aria-labelledby'],'title desc')
    def test_all_sign_points_and_coordinates(self):
        circles=self.tree.findall('s:circle',NS)
        for run,name in zip(self.audit['runs'],['eight-resampling','eight-none','4096-resampling']):
            points=[c for c in circles if c.attrib['data-series']==name];self.assertEqual(len(points),12)
            for i,(point,row) in enumerate(zip(points,run['points'])):
                p=row['filtered_sign_mass']['positive_probability'];self.assertEqual(float(point.attrib['data-value']),p)
                self.assertEqual(int(point.attrib['data-index']),i);self.assertAlmostEqual(float(point.attrib['cx']),625+i/11*435,3)
                self.assertAlmostEqual(float(point.attrib['cy']),390-p*230,3)
    def test_analytic_curve_and_mode_mean_marker_geometry(self):
        path=next(p for p in self.tree.findall('s:polyline',NS) if p.attrib['data-series']=='log-density-ratio')
        points=[tuple(map(float,p.split(','))) for p in path.attrib['points'].split()];self.assertEqual(len(points),101)
        m2=self.audit['reference']['history'][-1]['mode_squared']
        for i,(x,y) in enumerate(points):
            state=-2.5+i*.05;ratio=-12*(state*state-m2)**2/(2*.3**2)
            self.assertAlmostEqual(x,80+i/100*435,3);self.assertAlmostEqual(y,390-(ratio+1200)/1200*230,3)
        markers=[c for c in self.tree.findall('s:circle',NS) if c.attrib['data-series']=='exact-modes'];self.assertEqual(len(markers),2)
        for c,mode in zip(markers,self.audit['reference']['history'][-1]['modes']):
            self.assertEqual(float(c.attrib['data-value']),mode);self.assertAlmostEqual(float(c.attrib['cy']),160,3)
        mean=next(c for c in self.tree.findall('s:circle',NS) if c.attrib['data-series']=='exact-mean');self.assertEqual(float(mean.attrib['data-value']),0.);self.assertEqual(float(mean.attrib['cx']),297.5)
    def test_changed_incomplete_or_inconsistent_fixtures_fail_and_qualifications_visible(self):
        mutations=[lambda v:v['reference']['data'].pop(),lambda v:v['reference']['history'][-1].update(modes=[0.]),lambda v:v['runs'][0]['points'].pop(),lambda v:v['runs'][2]['settings'].update(seed=7),lambda v:v['runs'][0]['points'][-1]['filtered_sign_mass'].update(positive_probability=2.),lambda v:v['runs'][0]['points'][-1].update(unique_initial_ancestors=2)]
        for mutate in mutations:
            v=copy.deepcopy(self.audit);mutate(v)
            with self.assertRaises(ValueError):RENDER.render(v)
        text=' '.join(e.text or '' for e in self.tree.findall('s:text',NS))
        for phrase in ['not held-out','static only','not a typical state','not a budget rule','not event probability','No variance/evidence reference','mass 0']:
            self.assertIn(phrase,text)
