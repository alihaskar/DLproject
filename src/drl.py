import pandas as pd
from .regimes.market_regime_detector import MarketRegimeDetector
from .metalabel.triple_barrier import MetaLabeler
from .RL.rl_manager import RLManager

class DRL:
    def __init__(self, data_path: str):
        """
        Initialize the DRL system with all components
        
        Args:
            data_path: Path to the input data CSV file
        """
        self.data_path = data_path
        self._regime_detector = None
        self._metalabeler = None
        self.rl = RLManager(data_path)
        
    @property
    def regime_detector(self):
        if self._regime_detector is None:
            self._regime_detector = MarketRegimeDetector(self.data_path)
        return self._regime_detector
    
    @property
    def metalabeler(self):
        if self._metalabeler is None:
            self._metalabeler = MetaLabeler(self.data_path)
        return self._metalabeler
    
    def regimes(self) -> pd.DataFrame:
        """
        Detect market regimes using all available methods
        
        Returns:
            DataFrame containing detected regimes and data
        """
        return self.regime_detector.detect_all_regimes()
    
    def metalabel(self) -> pd.DataFrame:
        """
        Generate metalabels for the trading strategy
        
        Returns:
            DataFrame containing metalabels and features
        """
        return self.metalabeler.generate_labels() 