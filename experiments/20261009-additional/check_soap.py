"""Tensor-only SOAP checks after applying 09-soap.patch; no model or training run."""
import argparse
import copy
import importlib.util
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--device', default='cpu', choices=('cpu', 'cuda'))
    args = parser.parse_args()
    import torch

    spec = importlib.util.spec_from_file_location('soap_experiment', args.root / 'optimizers/soap.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(123)
    torch.set_float32_matmul_precision('highest')
    p = torch.nn.Parameter(torch.randn(3, 3, device=args.device))
    absent = torch.nn.Parameter(torch.ones(3, 3, device=args.device))
    initial = p.detach().clone()
    optimizer = module.SOAP([p, absent], lr=0.01, betas=(0.8, 0.9), weight_decay=0.2,
                            precondition_frequency=10, max_precond_dim=3)
    p.grad = torch.diag(torch.tensor([1., 2., 3.], device=args.device))
    optimizer.step()
    torch.testing.assert_close(p, initial, atol=0, rtol=0)
    assert optimizer.state[p]['step'] == 0 and absent not in optimizer.state
    basis = torch.eye(3, device=args.device).flip(1)
    for q in optimizer.state[p]['Q']:
        torch.testing.assert_close(q.abs(), basis, atol=0, rtol=0)
    ql, qr = optimizer.state[p]['Q']
    grad = torch.randn_like(p)
    projected = ql.T @ grad @ qr
    direction = ql @ ((0.2 * projected) / ((0.1 * projected.square()).sqrt() + 1e-8)) @ qr.T
    reference = initial - (0.01 * (0.1 ** 0.5) / 0.2) * direction
    reference.add_(reference, alpha=-0.01 * 0.2)
    p.grad = grad
    optimizer.step()
    torch.testing.assert_close(p, reference, atol=1e-6, rtol=1e-6)
    print('SOAP initial covariance-only step and explicit Adam-in-eigenbasis update passed')

    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        p = torch.nn.Parameter(torch.randn(3, 5, device=args.device, dtype=dtype))
        optimizer = module.SOAP([p], lr=0.01, precondition_frequency=2, max_precond_dim=3)
        for _ in range(5):
            p.grad = torch.randn_like(p)
            optimizer.step()
        assert optimizer.state[p]['GG'][1] == []
        saved = copy.deepcopy(optimizer.state_dict())
        resumed_p = torch.nn.Parameter(p.detach().clone())
        resumed = module.SOAP([resumed_p])
        resumed.load_state_dict(saved)
        for key in ('exp_avg', 'exp_avg_sq'):
            assert resumed.state[resumed_p][key].dtype == torch.float32
            torch.testing.assert_close(resumed.state[resumed_p][key], optimizer.state[p][key], atol=0, rtol=0)
        for key in ('GG', 'Q'):
            assert resumed.state[resumed_p][key][0].dtype == torch.float32
            torch.testing.assert_close(resumed.state[resumed_p][key][0], optimizer.state[p][key][0], atol=0, rtol=0)
        for _ in range(4):
            grad = torch.randn_like(p)
            p.grad, resumed_p.grad = grad, grad.clone()
            optimizer.step()
            resumed.step()
            torch.testing.assert_close(p, resumed_p, atol=0, rtol=0)
        # skip missing gradients without covariance, step-counter or decay changes
        before = copy.deepcopy(optimizer.state_dict())
        before_p = p.detach().clone()
        p.grad = None
        optimizer.step()
        torch.testing.assert_close(p, before_p, atol=0, rtol=0)
        assert optimizer.state[p]['step'] == before['state'][0]['step']
        for key in ('exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(optimizer.state[p][key], before['state'][0][key], atol=0, rtol=0)
        print(f'SOAP {dtype}: skipped oversized axis, FP32 state, QR refresh and exact checkpoint continuation passed')

    p = torch.nn.Parameter(torch.zeros(3, 3, device=args.device))
    for kwargs in ({'precondition_frequency': 0}, {'max_precond_dim': 0}, {'betas': (0.9, 1.0)}):
        try:
            module.SOAP([p], **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f'invalid SOAP configuration accepted: {kwargs}')
    print('SOAP invalid configuration checks passed')


if __name__ == '__main__':
    main()
