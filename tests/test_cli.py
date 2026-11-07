import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


class CliTests(unittest.TestCase):
    def run_cli(self,*args):return subprocess.run([sys.executable,'-m','particlelab',*args],capture_output=True,text=True)
    def test_filter_and_kalman(self):
        for args in [('filter','examples/sequence.json','--particles','17','--seed','19'),('kalman','examples/sequence.json')]:
            r=self.run_cli(*args);self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(len(json.loads(r.stdout)['history']),12)
    def test_help_and_invalid_arguments(self):
        for command in ['filter','kalman','square-reference']:
            p=self.run_cli(command,'--help');self.assertEqual(p.returncode,0)
            for field in ['data','a','process_std','observation_std','initial_mean','initial_std']:self.assertIn(field,p.stdout)
        for args in [(),('unknown',),('filter','examples/sequence.json','--particles','0'),('filter','examples/sequence.json','--observation','other')]:self.assertEqual(self.run_cli(*args).returncode,2)
    def test_strict_payload_missing_extra_wrong_type(self):
        payload=json.loads(pathlib.Path('examples/sequence.json').read_text())
        with tempfile.TemporaryDirectory() as folder:
            path=pathlib.Path(folder)/'data.json'
            for value in [None,[],dict(payload,extra=1),dict(payload,particles=32)]+[{k:v for k,v in payload.items() if k!=field} for field in payload]:
                path.write_text(json.dumps(value));p=self.run_cli('filter',str(path));self.assertEqual(p.returncode,2);self.assertNotIn('Traceback',p.stderr)
    def test_malformed_and_missing_file_error(self):
        for command in ['filter','kalman','square-reference']:
            p=self.run_cli(command,'README.md');self.assertEqual(p.returncode,2);self.assertIn('particlelab:',p.stderr);self.assertEqual(p.stdout,'')
    def test_demo_is_repeatable(self):
        a=self.run_cli('demo');b=self.run_cli('demo');self.assertEqual(a.returncode,0,a.stderr);self.assertEqual(a.stdout,b.stdout)
        self.assertTrue(all(json.loads(a.stdout)['checks'].values()))
