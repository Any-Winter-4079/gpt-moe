"""Check the two unapplied optimizer patches with stdlib only; never run training."""

import ast
import inspect
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = Path(__file__).resolve().parent


def extract(path, class_name, method_name=None):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    if method_name:
        node = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    return 'from __future__ import annotations\n' + ast.unparse(node)


def coefficients():
    tree = ast.parse((ROOT / 'optimizers/polar_express.py').read_text())
    return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'polar_express_coeffs' for t in n.targets))


class OptimizerStub:
    def __init__(self, params, defaults=None, **kwargs):
        params = list(params)
        self.param_groups = params if params and isinstance(params[0], dict) else [{'params': params}]
        self.kwargs = defaults if defaults is not None else kwargs


class AdamWStub(OptimizerStub):
    def __init__(self, params, lr, betas, eps, fused=False):
        super().__init__(params, lr=lr, betas=betas, eps=eps, fused=fused)


class ParameterStub:
    def __init__(self, *shape):
        self.shape = shape
        self.dtype = 'float32'
        self.requires_grad = True

    def dim(self):
        return len(self.shape)

    def size(self, dim):
        return self.shape[dim]

    def numel(self):
        return math.prod(self.shape)


def check_gram_constructor(scratch):
    module = ModuleType('gram_newton_schulz')
    module.GramNewtonSchulz = lambda **kwargs: SimpleNamespace(**kwargs)
    sys.modules['gram_newton_schulz'] = module
    env = {'torch': SimpleNamespace(optim=SimpleNamespace(Optimizer=OptimizerStub)),
           'polar_express_coeffs': coefficients()}
    exec(extract(scratch / 'optimizers/muon.py', 'Muon'), env)
    optimizer = env['Muon']([object()], backend='gram_newton_schulz', gram_reset_iterations=[1, 3])
    backend = optimizer.gram_backend
    assert backend.gram_newton_schulz_reset_iterations == [1, 3]
    assert not backend.ns_use_kernels
    assert backend.ns_epsilon == 1e-6 / 1.02
    assert backend.ns_coefficients[1:] == coefficients()[1:]
    expected = tuple(c / 1.02**power for c, power in zip(coefficients()[0], (1, 3, 5)))
    assert backend.ns_coefficients[0] == expected
    return backend.ns_coefficients


def check_dion_partition(scratch):
    module = ModuleType('dion')
    module.Dion3 = OptimizerStub
    sys.modules['dion'] = module
    env = {'inspect': inspect, 'torch': SimpleNamespace(optim=SimpleNamespace(AdamW=AdamWStub))}
    exec(extract(scratch / 'model/gpt.py', 'GPT', 'configure_optimizers'), env)
    params = [('transformer.wte.weight', ParameterStub(16, 4)),
              ('transformer.h.0.qkv.weight', ParameterStub(12, 4)),
              ('transformer.h.0.mlp.weight', ParameterStub(8, 4)),
              ('transformer.h.0.router.weight', ParameterStub(2, 4)),
              ('transformer.h.0.norm.weight', ParameterStub(4)),
              ('lm_head.weight', ParameterStub(16, 4))]
    options = dict(muon_lr_scale=.15, muon_momentum=.95, muon_use_nesterov=True,
                   muon_backend='polarexpress', muon_backend_steps=5,
                   dion3_lr_scale=.75, dion3_fraction=.25, dion3_mu=.95, dion3_beta2=.95,
                   dion3_weight_decay=0., dion3_eps=1e-6, dion3_adjust_lr='rms_norm',
                   dion3_use_triton=False, dion3_triton_post_ortho=False)
    for mode in ('dion3', 'muon', 'adamw'):
        # full DDP model, first PP stage, second PP stage, and empty AdamW groups
        for owned in (params, params[:-1], params[1:], params[1:4]):
            model = SimpleNamespace(optimizer_type=mode, named_parameters=lambda: iter(owned), **options)
            result = env['configure_optimizers'](model, .005, (.9, .95), 1e-8, .01, 'cuda',
                                                 muon_class=OptimizerStub, master_process=False, log_buffer=[])
            assigned = [p for opt in result.values() for group in opt.param_groups for p in group['params']]
            assert len(assigned) == len(set(assigned)) == len(owned)
            assert set(assigned) == {p for _, p in owned}
            if mode == 'dion3':
                assert set(result) == {'adamw', 'dion3'}
                dion = result['dion3']
                assert dion.kwargs['distributed_mesh'] is None
                assert dion.kwargs['lr'] == .005 * .75
                assert dion.kwargs['fraction'] == .25
                assert dion.kwargs['muon_beta2'] == .95
                assert dion.kwargs['epsilon'] == 1e-6
                matrix = [p for n, p in owned if p.dim() == 2 and 'wte' not in n and 'lm_head' not in n]
                assert [p for g in dion.param_groups for p in g['params']] == [p for p in matrix if p.size(0) != 3*p.size(1)] + [p for p in matrix if p.size(0) == 3*p.size(1)]
                for group in dion.param_groups:
                    assert bool(group.get('num_heads') == 3) == all(p.size(0) == 3*p.size(1) for p in group['params'])


def check_scalar_gram(transformed):
    # singular-value recurrence checks coefficient/normalization algebra in float64
    rng = random.Random(51)
    for scale in (0., 1e-10, 1e-4, 1., 1e5):
        values = [rng.random() * scale for _ in range(16)]
        norm = math.sqrt(sum(value * value for value in values))
        for value in values:
            standard = value / (norm * 1.02 + 1e-6)
            for a, b, c in coefficients():
                standard = a*standard + b*standard**3 + c*standard**5
            x = value / (norm + 1e-6 / 1.02)
            r, q = x*x, 1.
            for i, (a, b, c) in enumerate(transformed):
                if i == 2:
                    x *= q
                    r, q = x*x, 1.
                z = b*r + c*r*r
                q *= a + z
                if i < 4 and i + 1 != 2:
                    r *= (a + z)**2
            assert math.isclose(standard, q*x, rel_tol=1e-9, abs_tol=1e-11), (standard, q*x)


def main():
    for stem in ('05-gram-newton-schulz', '07-dion3'):
        patch = EXPERIMENTS / (stem + '.patch')
        names = [line[len('+++ b/'):] for line in patch.read_text().splitlines() if line.startswith('+++ b/')]
        subprocess.run(['git', 'apply', '--check', str(patch)], cwd=ROOT, check=True)
        with tempfile.TemporaryDirectory(prefix=stem + '-') as temp:
            scratch = Path(temp)
            # a real temporary repository keeps git apply inside this scratch directory
            subprocess.run(['git', 'init', '-q', str(scratch)], check=True)
            originals = {name: (ROOT / name).read_bytes() for name in names}
            for name, content in originals.items():
                target = scratch / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            subprocess.run(['git', 'apply', str(patch)], cwd=scratch, check=True)
            for name in names:
                if name.endswith('.py'):
                    ast.parse((scratch / name).read_text(), filename=name)
            if stem.startswith('05'):
                check_scalar_gram(check_gram_constructor(scratch))
            else:
                check_dion_partition(scratch)
            subprocess.run(['git', 'apply', '-R', '--check', str(patch)], cwd=scratch, check=True)
            subprocess.run(['git', 'apply', '-R', str(patch)], cwd=scratch, check=True)
            assert all((scratch / name).read_bytes() == content for name, content in originals.items())
        print(stem + ': apply/reverse, syntax, and local wiring/algebra checks passed')
    print('No PyTorch/CUDA kernels or optimizer steps were executed.')


if __name__ == '__main__':
    main()
