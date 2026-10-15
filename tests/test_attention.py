import io
import unittest
from dataclasses import asdict

import torch

from model import GPT, GPTConfig
from attention import CausalSelfAttention, SlidingWindowAttention


class AttentionTests(unittest.TestCase):

    def config(self, **kwargs):
        args = dict(vocab_size=29, block_size=12, n_layer=2, n_head=2,
                    n_embd=16, dropout=0.0)
        args.update(kwargs)
        return GPTConfig(**args)

    def test_window_boundaries(self):
        # Zero queries/keys make attention uniform over exactly the visible tokens.
        # Identity values/output make the expected result a local moving average.
        x = torch.arange(2 * 8 * 16, dtype=torch.float32).view(2, 8, 16) / 32
        for window in (1, 3, 8, 20):
            with self.subTest(window=window):
                attn = SlidingWindowAttention(self.config(attention_type='swa',
                                                         sliding_window=window, bias=False)).eval()
                with torch.no_grad():
                    attn.c_attn.weight.zero_()
                    attn.c_attn.weight[32:].copy_(torch.eye(16))
                    attn.c_proj.weight.copy_(torch.eye(16))
                expected = torch.stack([x[:, max(0, t - window + 1):t + 1].mean(1)
                                        for t in range(8)], dim=1)
                torch.testing.assert_close(attn(x), expected)
                attn.flash = False
                torch.testing.assert_close(attn(x), expected)

    def test_full_window_matches_mha(self):
        torch.manual_seed(42)
        config = self.config(attention_type='swa', sliding_window=12)
        mha = CausalSelfAttention(config).eval()
        swa = SlidingWindowAttention(config).eval()
        swa.load_state_dict(mha.state_dict())
        x = torch.randn(2, 9, 16)
        torch.testing.assert_close(swa(x), mha(x))

    def test_locality_and_causality(self):
        attn = SlidingWindowAttention(self.config(attention_type='swa', sliding_window=3)).eval()
        x = torch.randn(2, 8, 16)
        expected = attn(x)
        changed = x.clone()
        changed[:, :3] += 10 # Outside the window for query position 5.
        changed[:, 6:] -= 10 # Future tokens for query position 5.
        torch.testing.assert_close(attn(changed)[:, 5], expected[:, 5])

    def test_sdpa_manual_outputs_and_gradients(self):
        config = self.config(attention_type='swa', sliding_window=3)
        sdpa = SlidingWindowAttention(config)
        manual = SlidingWindowAttention(config)
        manual.load_state_dict(sdpa.state_dict())
        manual.flash = False
        x = torch.randn(2, 8, 16, requires_grad=True)
        other = x.detach().clone().requires_grad_(True)
        expected, actual = sdpa(x), manual(other)
        torch.testing.assert_close(actual, expected)
        expected.square().sum().backward()
        actual.square().sum().backward()
        torch.testing.assert_close(other.grad, x.grad)
        for parameter, reference in zip(manual.parameters(), sdpa.parameters()):
            torch.testing.assert_close(parameter.grad, reference.grad)

    def test_gpt_training_checkpoint_crop_and_generation(self):
        config = self.config(attention_type='swa', sliding_window=3, dropout=0.2, bias=False)
        model = GPT(config)
        tokens = torch.randint(0, 29, (2, 8))
        logits, loss = model(tokens, tokens)
        self.assertEqual(logits.shape, (2, 8, 29))
        loss.backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        model.eval()
        torch.testing.assert_close(model(tokens)[0], model(tokens)[0], atol=0, rtol=0)
        buffer = io.BytesIO()
        torch.save(dict(model_args=asdict(config), model=model.state_dict()), buffer)
        buffer.seek(0)
        checkpoint = torch.load(buffer, weights_only=True)
        restored = GPT(GPTConfig(**checkpoint['model_args'])).eval()
        restored.load_state_dict(checkpoint['model'])
        torch.testing.assert_close(restored(tokens)[0], model(tokens)[0])
        self.assertEqual(restored.config.sliding_window, 3)
        restored.crop_block_size(6)
        self.assertEqual(restored.transformer.wpe.weight.shape, (6, 16))
        self.assertEqual(restored.generate(tokens, 5).shape, (2, 13))

    def test_mla_cache_after_refactor(self):
        model = GPT(self.config(attention_type='mla')).eval()
        tokens = torch.randint(0, 29, (2, 8))
        expected, _ = model(tokens, tokens)
        prefix, cache = model.forward_cached(tokens[:, :3])
        suffix, cache = model.forward_cached(tokens[:, 3:], cache)
        torch.testing.assert_close(torch.cat((prefix, suffix), dim=1), expected,
                                   atol=2e-6, rtol=2e-5)
        self.assertEqual(cache.length, 8)
        self.assertNotIn('wpe', model.transformer)

    def test_invalid_window(self):
        for window in (0, -1, 1.5):
            with self.subTest(window=window), self.assertRaises(ValueError):
                self.config(attention_type='swa', sliding_window=window)


if __name__ == '__main__':
    unittest.main()
