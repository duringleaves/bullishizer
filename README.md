This is a quick and dirty tool to implement /u/Region-Format's "Methodology 3" bullishness algorithm for any stock as shared in this thread https://old.reddit.com/r/Superstonk/comments/1kv3y3e/some_fun_for_the_long_weekend_methodology_3/

It calculates a bullishness status based on a number of weighted indicators:

[1] RSI_Score = 100 x PercentileRank (RSI, rolling 52wks)

[2] MACD_Slope_Score = 100 x PercentileRank (dMACD/dt, rolling 52wks)

[3] EMA Alignment_Score = 20 x # of bullish crosses (EMA20>50>100>200) +
(EMA20-EMA50) normalized to 0-20

[4] Volume_Score = 100 x PercentileRank (weekly Volume, 52wks)

[5] OBV_Score = 100 x PercentileRank (weekly \DeltaOBV, 52wks)

[6] VolOsc_Score = 100 x PercentileRank (VolOsc, 52wks)

[7] ADX_Score = 100 x PercentileRank (ADX, 52wks)

[8] Stoch_Score = 100 x PercentileRank (%K-%D, 52wks)

[9] BBands_Breakout_Score = 100 x PercentileRank((Close-UpperBand)/Width, 52wks)

[10] Fib_Confluence_Score = 100 x PercentileRank (distance to Fib levels, 52wks)
---

• MACD_Slope: 20%

• Volume Spike: 18%

• EMA Alignment & Slope: 14%

• OBV Momentum: 13%

• RSI: 9%

• Bollinger Breakout: 7%

• VolOsc: 6%

• Stochastic Cross: 5%

• ADX: 4%

• Fibonacci proximity: 4%

Composite_Score = Sigma (weightS_i$ x PercentileS_i$)

These are then combined to provide an initial Composite_Score

Finally, the model adds a "Cluster Score" when more than five Bullishness Scores exceed 80, up to a max 100, and then added to output a:
Total Bullishness Score

I wrote it to send me notifications via SimplePush.

---
Installation:

git clone https://github.com/duringleaves/bullishizer

cd bullishizer

python -m venv .venv

. .venv/bin/activate

pip install -r requirements.txt

mv .env_sample .env

(edit .env with your preferred username/password, if desired)

python main.py

(default authentication: u: admin p: password123)