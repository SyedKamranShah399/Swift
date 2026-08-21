"""
parity_check_decode.py -- proves the gated decode_v8 in detect_gated.py produces
output IDENTICAL to the original decode_v8 in detect.py, with no Hailo hardware.

It builds synthetic HEF-shaped outputs (random logits at the three YOLOv8 scales)
and runs both decoders over them for several seeds and conf thresholds, asserting
the boxes and scores match exactly (np.array_equal, not just allclose).

    python3 parity_check_decode.py
"""
import numpy as np

import detect            # original
import detect_gated      # gated copy


def fake_outputs(rng):
    """Random logits shaped like the raw HEF outputs: per scale a box(64) and
    cls(1) tensor, with a NHWC-style leading batch dim (squeezed inside decode)."""
    out = {}
    for i, size in enumerate((80, 40, 20)):
        # cls logits centred near the threshold so a realistic handful survive
        out[f"cls{i}"] = rng.normal(-2.0, 2.5, (1, size, size, 1)).astype(np.float32)
        out[f"box{i}"] = rng.normal(0.0, 1.0, (1, size, size, 64)).astype(np.float32)
    return out


def main():
    total = 0
    for seed in range(25):
        rng = np.random.default_rng(seed)
        outputs = fake_outputs(rng)
        for conf in (0.10, 0.25, 0.50, 0.75):
            b0, s0 = detect.decode_v8(outputs, conf)
            b1, s1 = detect_gated.decode_v8(outputs, conf)
            assert b0.shape == b1.shape, f"shape mismatch seed={seed} conf={conf}: {b0.shape} vs {b1.shape}"
            assert np.array_equal(b0, b1), f"BOX mismatch seed={seed} conf={conf}"
            assert np.array_equal(s0, s1), f"SCORE mismatch seed={seed} conf={conf}"
            total += len(b0)
    print(f"PARITY OK -- 25 seeds x 4 thresholds, identical boxes & scores "
          f"({total} detections compared, bit-for-bit equal)")


if __name__ == "__main__":
    main()
