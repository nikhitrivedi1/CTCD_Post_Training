"""Quick single-math-test across all bench models. Diagnoses KaTeX scoring."""

import os

from olmocr.bench.tests import load_single_test

TEST = {
    "id": "2503.04048_pg46_math_000",
    "type": "math",
    "max_diffs": 0,
    "math": r"{\mathcal{V}}(\psi_m)\rightarrow +\infty",
    "pdf": "arxiv_math/2503.04048_pg46.pdf",
    "page": 1,
}

test_obj = load_single_test(TEST)

bench_dir = "outputs/bench"
for model in sorted(os.listdir(bench_dir)):
    md_path = os.path.join(bench_dir, model, "markdowns", "arxiv_math_2503.04048_pg46_pg1.md")
    if not os.path.exists(md_path):
        print(f"{model:50s} NO MARKDOWN")
        continue
    with open(md_path) as f:
        md = f.read()
    passed, explanation = test_obj.run(md)
    status = "PASS" if passed else "FAIL"
    print(f"{model:50s} {status}  {(explanation or '')[:80]}")
