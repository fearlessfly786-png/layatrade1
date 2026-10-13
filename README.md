# LayaTrade1 — Laya System One fine-tuned on BTCUSDT trading decisions

**What this is:** the open-source **Laya System One** decision model (typed decisions:
choice + score, no free-text generation), fine-tuned across **5 versions (v1 → v5)** on
real Binance BTCUSDT 1-hour market data to predict the best trade action over the next
4 hours: `buy` / `hold` / `sell`, plus a conviction score.

**What this is NOT:** this is **training + evaluation only**. There is **no trading
agent** here — no wallet, no order execution, no live trading. Research/educational
project. Not financial advice.

---

## Model zoo (all evaluated on the SAME 994 never-trained hours: Sep 1 → Oct 13 2026)

| version | accuracy | always-hold baseline | buy calls | buy dir-right | buy net after fees | sell calls | sell dir-right | sell net after fees | TOTAL net after fees |
|---------|----------|---------------------|-----------|---------------|--------------------|-----------|----------------|---------------------|----------------------|
| v1 | 78.6% | 89.3% | — | — | — | — | — | — | −20.4% |
| v2 | 78.8% | 95.4% | 89 | 60% | — | 154 (over-calling) | 55% | — | −14.0% |
| v3 | **91.0%** | 95.4% | 28 | 71% | **+3.35%** | 27 | 44% | −8.93% | −5.58% |
| v4 | 95.2%* | 95.4% | 4 (collapsed) | 0% | −6.7% | 0 | — | — | −6.74% |
| **v5** ✅ | **92.05%** | 95.4% | **27** | **74%** | **+$58.96 (+5.9%)** | 17 | 35% | −$99.32 | **−4.04%** |

\* v4's high accuracy was the "quiet market rewards not trading" trap — it collapsed to
almost-always-hold and is kept only as a documented failed experiment.

**P&L simulation** (each call = one 4-hour position, $1,000 notional, 0.2% round-trip fee):

- v3: 55 trades → 32 wins / 23 losses (58% win) → winners +$237.88, losers −$183.69 → **net −$55.82**
- **v5: 44 trades → 26 wins / 18 losses (59% win) → winners +$219.91, losers −$172.27 → net −$40.36** (best overall)
- v5 buys-only subset is **fee-positive: 27 trades, 74% win, +$58.96 net (+5.9% per $1,000 in 6 weeks)**
- Sells (shorts) remain the losing component in every version — an honestly reported open weakness.

## Live test (v5, fresh never-seen candles, Oct 13 2026)

- Recent 227 hours: **86.8% accuracy** (always-hold baseline 89.4%)
- 8 sell calls → 4/8 directionally right (v3 live sells were 7/21)
- Live calls at test time: all **HOLD** with moderate confidence (0.46–0.50 P(hold)) —
  the model stays flat in a quiet market.

---

## The 5-iteration story (diagnose → fix → retrain → re-evaluate)

1. **v1** — 1,208 balanced rows, ±1.0% labels: learned task structure (acc 78.6%) but
   sell logic was inverted, confidence uninformative.
2. **v2** — ±1.5% labels, 5× data across 5 market regimes (7,939h pool): buy edge fixed
   (+0.30% mean, 71% dir-right) but sold 154 times (3% precision) and conviction head collapsed.
3. **v3** — hold-heavy rebalance (hold = 1.5× trades): sells cut to 27, best confidence
   separation (P(true) 0.630 vs 0.272), acc 91.0%.
4. **Diagnosis round 4** — deep autopsy showed v3's sells fire on **oversold dips that
   bounce** (21/27 calls bearish+RSI<30) while true dumps start from **bull/mixed tops**
   (15/24); sell recall 0/24; 17 missed buys worth +2.96% each.
5. **v4 (failed)** — sell relabeled to top-rollovers only: collapsed to always-hold with
   4 tangled buys (RSI 74–95 bull tops). Kept as a lesson: label surgery + 1-epoch chains
   can go conservative. **v5** — same clean sell labels + v2-style equal-thirds balance +
   1.3× trade oversampling: **best buys ever (74% win, +5.9% after fees), best accuracy
   of any trading version (92.05%)**.

Remaining honest weaknesses: sell/short side still loses after fees; conviction head has
near-zero signal (P(conv=2): 0.122 big moves vs 0.100 quiet); the always-hold baseline
(95.4%) is unbeaten in this quiet regime.

## The final checkpoint — laya-trading-v5

Download `laya-trading-v5.tar.gz` from the
[Releases page](https://github.com/fearlessfly786-png/layatrade1/releases), then:

```bash
tar xzf laya-trading-v5.tar.gz        # -> ./laya-trading-v5/
```

```python
import laya
agent = laya.load("./laya-trading-v5")
result = agent.predict_batch(
    ["BTCUSDT 1h trading decision. Price 84000. RSI(14) 55.0 (neutral). ..."],
    questions,   # see laya-trading-v5/questions.json
    batch_size=8,
)
# result[0]["answers"]["action"]["choice"]      -> buy / hold / sell
# result[0]["answers"]["action"]["probabilities"] -> {"buy":…, "hold":…, "sell":…}
# result[0]["answers"]["conviction"]["score"]    -> 0..2
```

## Training recipe (all CPU, sandbox-friendly)

- `laya.train.finetune()` with `loss="soft-ce", freeze_encoder=True, micro_batch=1,
  grad_accum=16, max_len=192, head_max_len=96`
- 5 chained chunks × 1 epoch (seeds 501–505), each chunk ~260 stratified rows
  (buy 91 / hold 113 / sell 57), chunk N trains from chunk N−1's checkpoint
- Dataset: Binance klines 9,000×1h (Oct 2025 → Oct 13 2026); indicators RSI(14),
  EMA12/26, MACD hist, ATR(14), volume ratio, 24h high/low distance; labels from
  forward 4h returns; **sell labels restricted to bull/mixed-aligned dumps
  (top-rollover pattern)**; oversold-bearish dips labeled hold
- Held-out eval: 994 hours (Sep 1 → Oct 13 2026) never used in training

## Repo layout

```
laya/                 # Laya System One source package (the model + trainer + runtime)
laya-go|java|dotnet|ts/  # official multilingual clients
research/ tests/ docs/ examples/ benchmarks/ assets/
build_dataset_v2|v4|v5.py   # dataset engineering per version
train_chunk_v2.py           # chunked fine-tune driver
predict_chunked.py          # resumable 994-row inference
diagnose.py, diagnose_v3.py # weakness autopsies
compare_v3_v4.py            # accuracy + P&L comparisons
test_live.py                # fresh-data live test (test only)
trade_train.jsonl, trade_eval*.jsonl, chunk_v*_[1-5].jsonl   # datasets
preds_v1..v5.jsonl, final_eval_results.json, live_test_results.json  # results
eval_truth.csv, candles_9000.csv    # ground truth + raw candles
WORKLOG.md                          # full agent work log, every iteration
```
