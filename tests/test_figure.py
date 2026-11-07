import copy
import importlib.util
import json
import math
import pathlib
import unittest
import xml.etree.ElementTree as ET
from particlelab.demo import demo

ROOT=pathlib.Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('render_demo',ROOT/'examples/render_demo.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
NS={'s':'http://www.w3.org/2000/svg'}


class FigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit=demo();cls.svg=module.render(cls.audit);cls.tree=ET.fromstring(cls.svg)
    def test_committed_regeneration_and_no_mutation(self):
        before=copy.deepcopy(self.audit);self.assertEqual(module.render(self.audit),self.svg);self.assertEqual(self.audit,before)
        self.assertEqual((ROOT/'assets/particle-audit.svg').read_text(),self.svg)
    def test_accessibility_and_metadata_all_rows(self):
        self.assertEqual(self.tree.attrib['aria-labelledby'],'title desc')
        data=json.loads(self.tree.find('s:metadata',NS).text)
        self.assertEqual(len(data['linear']),12);self.assertEqual(len(data['static']),20)
        self.assertEqual(data['seed_errors'],self.audit['linear']['per_seed_errors'])
        self.assertEqual(data['linear_settings'],self.audit['linear']['particle_run']['settings'])
        self.assertIn('No calibration',self.tree.find('s:desc',NS).text)
    def test_all_points_have_exact_values_and_positions(self):
        circles=self.tree.findall('s:circle',NS);data=module.manifest(self.audit)
        specifications=[('exact-mean','linear','exact_mean',75,155,435,230,-1,1.5),('particle-mean','linear','particle_mean',75,155,435,230,-1,1.5),('observations','linear','observation',75,155,435,230,-1,1.5),
          ('static-exact-var','static','exact_variance',625,155,435,230,0,.06),('static-particle-var','static','particle_variance',625,155,435,230,0,.06),('static-none-var','static','no_resampling_variance',625,155,435,230,0,.06),
          ('ess-before','static','ess_before',75,570,435,220,0,32),('ess-after','static','ess_after',75,570,435,220,0,32),('ess-none','static','ess_without_resampling',75,570,435,220,0,32),
          ('roots-after','static','roots_after',625,570,435,220,0,32),('roots-none','static','roots_without_resampling',625,570,435,220,0,32)]
        for name,group,key,x,y,w,h,lo,hi in specifications:
            items=[c for c in circles if c.attrib.get('data-series')==name];rows=data[group];self.assertEqual(len(items),len(rows))
            for i,(circle,row) in enumerate(zip(items,rows)):
                self.assertEqual(float(circle.attrib['data-value']),row[key]);self.assertEqual(int(circle.attrib['data-index']),i)
                self.assertAlmostEqual(float(circle.attrib['cx']),x+i/(len(rows)-1)*w,3)
                self.assertAlmostEqual(float(circle.attrib['cy']),y+h-(row[key]-lo)/(hi-lo)*h,3)
    def test_sd_polygons_use_every_mean_variance(self):
        polygons=self.tree.findall('s:polygon',NS);self.assertEqual(len(polygons),2)
        data=module.manifest(self.audit)['linear']
        for polygon,prefix in zip(polygons,['exact','particle']):
            points=[tuple(map(float,p.split(','))) for p in polygon.attrib['points'].split()];self.assertEqual(len(points),24)
            values=[r[prefix+'_mean']+math.sqrt(r[prefix+'_variance']) for r in data]+[r[prefix+'_mean']-math.sqrt(r[prefix+'_variance']) for r in data[::-1]]
            for point,value in zip(points,values):self.assertAlmostEqual(point[1],155+230-(value+1)/2.5*230,3)
    def test_bad_fixture_or_out_of_domain_fails(self):
        mutations=[lambda r:r['linear']['data'].pop(),lambda r:r['linear']['per_seed_errors'].pop(),lambda r:r['linear']['particle_run']['settings'].update(seed=7),
          lambda r:r['static_failure']['resampling']['history'][0].update(variance=.1),lambda r:r['static_failure']['without_resampling']['history'].pop(),
          lambda r:r['linear']['particle_run']['history'][0].update(mean=float('nan'))]
        for mutate in mutations:
            r=copy.deepcopy(self.audit);mutate(r)
            with self.assertRaises(ValueError):module.render(r)
    def test_qualifications_and_zero_variance_visible(self):
        text=' '.join(e.text or '' for e in self.tree.findall('s:text',NS))
        for phrase in ['not held-out performance','not coverage bands','not credible coverage validation','missing initial support','not zero posterior uncertainty','not effective present-state support','no real-data claim','All plotted values']:
            self.assertIn(phrase,text)
        data=module.manifest(self.audit);self.assertEqual(data['static'][-1]['particle_variance'],0.)
        self.assertGreater(data['static'][-1]['exact_variance'],0.)
