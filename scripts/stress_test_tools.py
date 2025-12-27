
import sys
import os
import json
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.skills.indicator_skills import GetATRTool, GetVWAPTool, DetectSupportResistanceTool, GetVolumeProfileTool
from src.skills.data_skills import GetTimeOfDayStatsTool

def stress_test():
    print("🚀 Starting Tool Stress Test\n")
    
    # 1. Test Time-of-Day Stats
    tod_tool = GetTimeOfDayStatsTool()
    tod_results = tod_tool.execute(lookback_days=10)
    print(f"✅ Time-of-Day Stats: {len(tod_results['hourly_stats'])} hours analyzed")
    
    # 2. Test ATR
    atr_tool = GetATRTool()
    atr_results = atr_tool.execute(lookback_bars=50)
    print(f"✅ ATR: Current ATR is {atr_results['current_atr']:.2f}")
    
    # 3. Test VWAP
    vwap_tool = GetVWAPTool()
    vwap_results = vwap_tool.execute(lookback_bars=5)
    print(f"✅ VWAP: Current VWAP is {vwap_results['current_vwap']:.2f}")
    
    # 4. Test S&R Detection
    sr_tool = DetectSupportResistanceTool()
    sr_results = sr_tool.execute(lookback_bars=500)
    print(f"✅ Support & Resistance: {len(sr_results['levels'])} levels detected")
    for level in sr_results['levels'][:3]:
        print(f"   - {level['type']}: {level['price']} (Strength: {level['strength']})")
        
    # 5. Test Volume Profile
    vp_tool = GetVolumeProfileTool()
    vp_results = vp_tool.execute(lookback_bars=500)
    print(f"✅ Volume Profile: POC at {vp_results['poc_price']:.2f}")

    print("\n🎉 All Research Tools Verified!")

if __name__ == "__main__":
    stress_test()
