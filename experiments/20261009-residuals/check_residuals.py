from __future__ import annotations

import argparse
import copy
import sys
from contextlib import nullcontext
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config.model import GPTConfig
from config.training import TrainingConfig
from model.gpt import GPT
from runtime.pipeline import PipelineStageModel


def global_key(key, stage, split):
    if stage == 1 and key.startswith("transformer.h."):
        parts = key.split(".", 3)
        parts[2] = str(int(parts[2]) + split)
        return ".".join(parts)
    return key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    assert torch.cuda.is_available(), "run this check in the CUDA training environment"
    flags = [name for name in (
        "use_long_skips", "use_embedding_reinjection", "use_parallel_attn_mlp",
        "use_channelwise_path_scales", "use_depth_scaled_main_init",
    ) if hasattr(GPTConfig, name)]
    assert len(flags) == 1, "apply exactly one residual experiment before running this check"
    flag = flags[0]
    rtol, atol = (0.03, 0.01) if args.bf16 else (2e-4, 2e-5)
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if args.bf16 else nullcontext()
    torch.manual_seed(1337)
    tokens = torch.randint(0, 63, (2, 16), device="cuda")
    targets = torch.randint(0, 63, (2, 16), device="cuda")

    for enabled in (False, True):
        for moe in (False, True):
            for position in ("rope", "absolute"):
                config = GPTConfig(
                    n_layers=6, d_model=32, mlp_hidden_dim=64, vocab_size=64,
                    pad_token_id=63, eos_token_id=62, n_heads=2, n_kv_heads=2,
                    attn_dim=32, global_attn_dim=32, global_n_kv_heads=2,
                    full_attention_layers=[1, 3, 5], pos_encoding_type=position,
                    use_flex_attention=False, use_tied_embeddings=False,
                    use_moe=moe, num_experts=3, moe_top_k=2, use_grouped_moe=False,
                    weighted_main_path_init=0.2,
                )
                setattr(config, flag, enabled)
                training = TrainingConfig(use_liger_loss=False, use_bf16_autocast=args.bf16,
                                          use_activation_checkpointing=False, max_seq_len=16)
                training.swa_schedule_values = [16]
                training.swa_initial_window_size = 16
                full = GPT(config, training).cuda()
                # nonzero gates exercise both the extra forward path and its gradient
                with torch.no_grad():
                    for name, parameter in full.named_parameters():
                        if name.endswith(("long_skip_scale", "embedding_scale")):
                            parameter.fill_(0.15)
                stages = [GPT(config, training, stage_index=i).cuda() for i in (0, 1)]
                state = full.state_dict()
                for i, stage in enumerate(stages):
                    stage.load_state_dict({key: state[global_key(key, i, 3)] for key in stage.state_dict()})
                wrappers = [PipelineStageModel(stage) for stage in stages]

                for checkpoint in (False, True):
                    for model in [full, *stages]:
                        model.train()
                        model.use_activation_checkpointing = checkpoint
                        model.zero_grad(set_to_none=True)
                    with ctx:
                        reference = full(tokens, targets)
                        boundary = wrappers[0](tokens)
                        boundary = boundary if isinstance(boundary, tuple) else (boundary,)
                        actual = wrappers[1](*boundary, targets=targets)
                    for left, right in zip(reference, actual):
                        torch.testing.assert_close(left, right, rtol=rtol, atol=atol)
                    reference[0].backward()
                    actual[0].backward()
                    params = dict(full.named_parameters())
                    for i, stage in enumerate(stages):
                        for key, parameter in stage.named_parameters():
                            expected = params[global_key(key, i, 3)].grad
                            assert (expected is None) == (parameter.grad is None), key
                            if expected is not None:
                                assert torch.isfinite(parameter.grad).all(), key
                                torch.testing.assert_close(expected, parameter.grad, rtol=rtol, atol=atol)

                for model in [full, *stages]:
                    model.eval()
                with torch.no_grad(), ctx:
                    boundary = wrappers[0](tokens)
                    boundary = boundary if isinstance(boundary, tuple) else (boundary,)
                    torch.testing.assert_close(full(tokens), wrappers[1](*boundary), rtol=rtol, atol=atol)
                print(f"{flag}={enabled}, moe={moe}, position={position}: logits, losses and gradients agree with two-stage composition")

    if args.compile:
        config.use_moe = False
        config.pos_encoding_type = "rope"
        training.use_activation_checkpointing = True
        eager = GPT(config, training).cuda().train()
        candidate = copy.deepcopy(eager)
        with torch.no_grad():
            for model in (eager, candidate):
                for name, parameter in model.named_parameters():
                    if name.endswith(("long_skip_scale", "embedding_scale")):
                        parameter.fill_(0.15)
        compiled = torch.compile(candidate, fullgraph=True, dynamic=False)
        with ctx:
            expected = eager(tokens, targets)[0]
            actual = compiled(tokens, targets)[0]
        torch.testing.assert_close(expected, actual, rtol=rtol, atol=atol)
        expected.backward()
        actual.backward()
        for left, right in zip(eager.parameters(), candidate.parameters()):
            if left.grad is not None:
                torch.testing.assert_close(left.grad, right.grad, rtol=rtol, atol=atol)
        print("Compiled dense forward/backward agrees with eager with block checkpointing enabled")
    print("These are single-GPU component checks; multi-GPU scheduling, FlexAttention and training performance still need measurement")


if __name__ == "__main__":
    main()
