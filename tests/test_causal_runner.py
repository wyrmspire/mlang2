import pytest
import pandas as pd
import numpy as np
from src.sim.causal_runner import CausalExecutor, StepResult
from src.sim.stepper import MarketStepper
from src.sim.account_manager import AccountManager
from src.sim.oco_engine import OCOBracket, OCOStatus
from src.policy.scanners import Scanner, ScanResult
from src.data.loader import load_continuous_contract
from src.data.resample import resample_all_timeframes

class MockScanner(Scanner):
    def __init__(self, triggers_on_indices=[5]):
        self.triggers_on_indices = triggers_on_indices
        self.call_count = 0
        
    @property
    def scanner_id(self) -> str:
        return "MockScanner"
        
    def scan(self, market_state, features):
        self.call_count += 1
        # Trigger on specific call counts
        if self.call_count in self.triggers_on_indices:
            return ScanResult(
                scanner_id=self.scanner_id,
                triggered=True,
                context={'test': True, 'score': 0.9}
            )
        return ScanResult(scanner_id=self.scanner_id, triggered=False)

@pytest.fixture
def real_data():
    # Load real data (slice for speed)
    df = load_continuous_contract()
    df = df.head(500)
    # Resample for higher timeframes
    htf = resample_all_timeframes(df)
    return df, htf['5m'], htf['15m']

def test_causal_executor_real_data_flow(real_data):
    df_1m, df_5m, df_15m = real_data
    
    # Start deeper to allow features
    start_idx = 200
    stepper = MarketStepper(df_1m, start_idx=start_idx, end_idx=start_idx + 50)
    account_manager = AccountManager()
    
    # Trigger on the 5th step from start
    scanner = MockScanner(triggers_on_indices=[5])
    
    executor = CausalExecutor(
        df=df_1m,
        stepper=stepper,
        account_manager=account_manager,
        scanner=scanner,
        df_5m=df_5m,
        df_15m=df_15m
    )
    
    # Step through
    results = []
    triggered = False
    
    for _ in range(10):
        res = executor.step()
        if res:
            results.append(res)
            if res.scanner_triggers:
                triggered = True
                assert len(res.new_orders) == 1
                
    assert len(results) == 10
    assert triggered, "Scanner should have triggered on the 5th step"
    
    # Verify features were computed (check ATR)
    last_res = results[-1]
    assert last_res.atr > 0, "ATR should be computed correctly with real data"
    assert last_res.features is not None

def test_causal_executor_trade_lifecycle(real_data):
    df_1m, df_5m, df_15m = real_data
    
    start_idx = 200
    stepper = MarketStepper(df_1m, start_idx=start_idx, end_idx=start_idx + 100)
    account_manager = AccountManager()
    
    # Trigger immediately on 1st step
    scanner = MockScanner(triggers_on_indices=[1])
    
    executor = CausalExecutor(
        df=df_1m,
        stepper=stepper,
        account_manager=account_manager,
        scanner=scanner,
        df_5m=df_5m,
        df_15m=df_15m
    )
    
    # 1. Trigger Entry
    res1 = executor.step()
    assert res1.new_orders, "Should create order"
    bracket = res1.new_orders[0]
    
    # 2. Step forward until entry fill (usually immediate/next bar for Market)
    filled = False
    for _ in range(5):
        res = executor.step()
        if not res: break
        
        # Check for fills events in the result
        entry_fills = [e for b, e in res.fills if e == 'ENTRY']
        if entry_fills:
            filled = True
            break
            
    assert filled, "Order should fill within a few bars"
    
    # 3. Step forward until exit
    # This might take a while depending on price action.
    # We just want to ensure logic runs without error.
    for _ in range(50):
        res = executor.step()
        if not res: break
        
        exit_fills = [e for b, e in res.fills if e in ['STOP_LOSS', 'TAKE_PROFIT', 'TIMEOUT']]
        if exit_fills:
            # Trade completed
            assert bracket.status in [OCOStatus.CLOSED_SL, OCOStatus.CLOSED_TP, OCOStatus.CLOSED_TIMEOUT]
            return

    # If we get here, trade didn't close in 50 bars, which is fine, 
    # but let's assert the account position open
    acc = account_manager.get_account('default')
    assert len(acc.positions) > 0 or len(acc.closed_positions) > 0
