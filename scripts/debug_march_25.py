"""Debug March 25 trade miss."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
from src.data.loader import load_continuous_contract
from src.data.resample import resample_all_timeframes

NY_TZ = ZoneInfo('America/New_York')

# Load data
df = load_continuous_contract()
df['time_ny'] = df['time'].dt.tz_convert(NY_TZ)

march_25 = datetime(2025, 3, 25).date()
prev_day = march_25 - timedelta(days=1)

# Session levels
asian_start = datetime.combine(prev_day, time(19, 0)).replace(tzinfo=NY_TZ)
asian_end = datetime.combine(march_25, time(0, 0)).replace(tzinfo=NY_TZ)
london_start = datetime.combine(march_25, time(2, 0)).replace(tzinfo=NY_TZ)
london_end = datetime.combine(march_25, time(8, 30)).replace(tzinfo=NY_TZ)

asian_bars = df[(df['time_ny'] >= asian_start) & (df['time_ny'] < asian_end)]
london_bars = df[(df['time_ny'] >= london_start) & (df['time_ny'] < london_end)]

london_high = london_bars['high'].max() if not london_bars.empty else 0
london_low = london_bars['low'].min() if not london_bars.empty else 0

print(f'London High: {london_high:.2f}')
print(f'London Low: {london_low:.2f}')

# Trade window
trade_start = datetime.combine(march_25, time(9, 30)).replace(tzinfo=NY_TZ)
trade_end = datetime.combine(march_25, time(11, 30)).replace(tzinfo=NY_TZ)
trade_window = df[(df['time_ny'] >= trade_start) & (df['time_ny'] <= trade_end)]

print(f'\nTrade Window High: {trade_window["high"].max():.2f}')
print(f'Broke London High? {trade_window["high"].max() > london_high}')

# 5m data
htf = resample_all_timeframes(df)
df_5m = htf['5m']
df_5m['time_ny'] = df_5m['time'].dt.tz_convert(NY_TZ)

window_5m = df_5m[(df_5m['time_ny'] >= trade_start) & (df_5m['time_ny'] <= trade_end)]
print(f'\n5m bars:')
for _, row in window_5m.iterrows():
    print(f'{row["time_ny"].strftime("%H:%M")} O:{row["open"]:.2f} H:{row["high"]:.2f} L:{row["low"]:.2f} C:{row["close"]:.2f}')

# Check for bearish FVG after London High break (for SHORT setup)
print('\n=== FVG Analysis ===')
# After london high break, look for bearish FVG
break_mask = window_5m['high'] > london_high
if break_mask.any():
    break_idx = break_mask.idxmax()
    after_break = window_5m.loc[break_idx:]
    print(f'Break at: {window_5m.loc[break_idx, "time_ny"]}')
    print(f'Bars after break: {len(after_break)}')
    
    # Check for bearish FVG - show all gap calculations
    print('\nChecking for gaps:')
    found_fvg = False
    for i in range(1, len(after_break) - 1):
        prev = after_break.iloc[i-1]
        curr = after_break.iloc[i]
        next_ = after_break.iloc[i+1]
        
        # Bearish FVG: gap between prev.low and next.high
        gap = prev['low'] - next_['high']
        if gap > 0:  # Any gap
            print(f'  {curr["time_ny"].strftime("%H:%M")}: gap={gap:.2f} (prev.L={prev["low"]:.2f}, next.H={next_["high"]:.2f})')
            if gap > 0.5:
                print(f'    ^ VALID BEARISH FVG!')
                found_fvg = True
    
    if not found_fvg:
        print('  No bearish FVG found with gap > 0.5')
        print('\n  Checking for bullish FVG (gap between prev.high and next.low):')
        for i in range(1, len(after_break) - 1):
            prev = after_break.iloc[i-1]
            curr = after_break.iloc[i]
            next_ = after_break.iloc[i+1]
            
            gap = next_['low'] - prev['high']
            if gap > 0:
                print(f'    {curr["time_ny"].strftime("%H:%M")}: bullish gap={gap:.2f}')
else:
    print('No break found')
