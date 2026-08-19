import pandas as pd
df = pd.read_hdf("anipose/2026-07-27_13/pose-2d/01.h5")   # adjust name to what's in pose-2d/
scores = df.xs("likelihood", level="coords", axis=1)
print(scores.describe())
print("fraction above 0.6:", (scores > 0.6).mean().mean())