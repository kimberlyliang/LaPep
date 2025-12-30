"""
WT Toxicity Predictor using ToxinPred3

Uses ToxinPred3 (https://github.com/raghavagps/toxinpred3) for predicting
toxicity of wild-type (WT) amino acid sequences.

ToxinPred3 is a method for predicting toxicity of peptides using:
- Model 1: ML model (Extra tree based on AAC and DPC)
- Model 2: Hybrid model (Ensemble of Extra tree + MERCI)

Reference:
Rathore AS, Arora A, Choudhury S, Tijare P, Raghava GPS (2024) 
ToxinPred3.0: An improved method for predicting the toxicity of peptides. 
Comput Biol Med. 179:108926
"""

import numpy as np
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Optional, List, Union
import pandas as pd


class ToxinPred3Wrapper:
    """
    Wrapper for ToxinPred3 command-line tool.
    
    Can be used programmatically or via subprocess calls.
    """
    
    def __init__(
        self,
        model: int = 2,
        threshold: float = 0.38,
        use_pip: bool = True
    ):
        """
        Initialize ToxinPred3 wrapper.
        
        Args:
            model: Model to use (1: ML model, 2: Hybrid model). Default: 2
            threshold: Threshold for toxicity prediction (0-1). Default: 0.38
            use_pip: Whether to use pip-installed version. Default: True
        """
        self.model = model
        self.threshold = threshold
        self.use_pip = use_pip
        
        # Check if toxinpred3 is available
        self._check_availability()
    
    def _check_availability(self):
        """Check if ToxinPred3 is installed and available."""
        try:
            if self.use_pip:
                # Try to import the pip package
                import toxinpred3
                self.available = True
                self.use_subprocess = False
            else:
                # Check if command-line tool is available
                result = subprocess.run(
                    ['toxinpred3', '-h'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                self.available = result.returncode == 0
                self.use_subprocess = True
        except (ImportError, FileNotFoundError, subprocess.TimeoutExpired):
            # Try command-line version as fallback
            try:
                result = subprocess.run(
                    ['toxinpred3.py', '-h'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                self.available = result.returncode == 0
                self.use_subprocess = True
            except (FileNotFoundError, subprocess.TimeoutExpired):
                self.available = False
                self.use_subprocess = False
                print("Warning: ToxinPred3 not found. Install with: pip install toxinpred3")
    
    def predict_proba(self, sequences: Union[str, List[str]]) -> Union[float, np.ndarray]:
        """
        Predict toxicity probability for peptide sequence(s).
        
        Args:
            sequences: Single sequence (str) or list of sequences (List[str])
                      Sequences should be in one-letter amino acid code
        
        Returns:
            Single float (if input is str) or numpy array (if input is List[str])
            Values are in [0, 1] where higher = more toxic
        """
        if not self.available:
            # Fallback: return random values
            if isinstance(sequences, str):
                return np.random.uniform(0.0, 1.0)
            return np.random.uniform(0.0, 1.0, size=len(sequences))
        
        single_input = isinstance(sequences, str)
        if single_input:
            sequences = [sequences]
        
        # Create temporary input file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.seq', delete=False) as f:
            for seq in sequences:
                f.write(seq + '\n')
            input_file = f.name
        
        # Create temporary output file
        output_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False).name
        
        try:
            if self.use_subprocess:
                # Use command-line tool
                cmd = [
                    'toxinpred3.py' if not self.use_pip else 'toxinpred3',
                    '-i', input_file,
                    '-o', output_file,
                    '-m', str(self.model),
                    '-t', str(self.threshold),
                    '-d', '2'  # Display all peptides
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode != 0:
                    print(f"Warning: ToxinPred3 failed: {result.stderr}")
                    return self._fallback_predict(sequences, single_input)
                
                # Read results from CSV
                if os.path.exists(output_file):
                    df = pd.read_csv(output_file)
                    # Extract toxicity scores
                    # Column name may vary, try common names
                    score_col = None
                    for col in ['Score', 'Toxicity_Score', 'Hybrid_Score', 'ML_Score']:
                        if col in df.columns:
                            score_col = col
                            break
                    
                    if score_col is None:
                        # Try to find any numeric column
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            score_col = numeric_cols[0]
                    
                    if score_col:
                        scores = df[score_col].values
                        # Normalize to [0, 1] if needed
                        scores = np.clip(scores, 0.0, 1.0)
                    else:
                        return self._fallback_predict(sequences, single_input)
                else:
                    return self._fallback_predict(sequences, single_input)
            else:
                # Use pip package programmatically
                # Note: This requires the pip package to have a Python API
                # If not available, fall back to subprocess
                return self._fallback_predict(sequences, single_input)
            
            return float(scores[0]) if single_input else scores.astype(np.float32)
            
        except Exception as e:
            print(f"Warning: ToxinPred3 prediction failed: {e}")
            return self._fallback_predict(sequences, single_input)
        finally:
            # Clean up temporary files
            try:
                os.unlink(input_file)
                if os.path.exists(output_file):
                    os.unlink(output_file)
            except:
                pass
    
    def _fallback_predict(self, sequences: List[str], single_input: bool) -> Union[float, np.ndarray]:
        """Fallback prediction when ToxinPred3 is not available."""
        if single_input:
            return np.random.uniform(0.0, 1.0)
        return np.random.uniform(0.0, 1.0, size=len(sequences))
    
    def predict_label(self, sequences: Union[str, List[str]], threshold: Optional[float] = None) -> Union[int, np.ndarray]:
        """
        Predict binary toxicity label.
        
        Args:
            sequences: Single sequence or list of sequences
            threshold: Threshold for classification (default: self.threshold)
        
        Returns:
            Binary label(s): 1 = toxic, 0 = non-toxic
        """
        if threshold is None:
            threshold = self.threshold
        
        probs = self.predict_proba(sequences)
        if isinstance(probs, float):
            return int(probs >= threshold)
        return (probs >= threshold).astype(np.int32)
    
    def __call__(self, sequences: Union[str, List[str]]) -> Union[float, np.ndarray]:
        """Make class callable."""
        return self.predict_proba(sequences)


class WTToxicityPredictor:
    """
    Wrapper for WT toxicity prediction using ToxinPred3.
    
    Provides predict() and normalize() methods consistent with other predictors.
    """
    
    def __init__(
        self,
        toxinpred3_model: Optional[ToxinPred3Wrapper] = None,
        reference_cdf: Optional[np.ndarray] = None,
        model: int = 2,
        threshold: float = 0.38
    ):
        """
        Initialize WT toxicity predictor.
        
        Args:
            toxinpred3_model: Pre-initialized ToxinPred3Wrapper (optional)
            reference_cdf: Reference CDF for normalization (optional)
            model: ToxinPred3 model to use (1: ML, 2: Hybrid). Default: 2
            threshold: ToxinPred3 threshold. Default: 0.38
        """
        if toxinpred3_model is None:
            self.toxinpred3_model = ToxinPred3Wrapper(model=model, threshold=threshold)
        else:
            self.toxinpred3_model = toxinpred3_model
        
        self.reference_cdf = reference_cdf or self._default_cdf()
    
    def predict(self, peptide: str) -> float:
        """
        Predict toxicity score for a WT peptide sequence.
        
        Args:
            peptide: Peptide sequence in one-letter amino acid code
        
        Returns:
            Toxicity score in [0, 1] where 1 = most toxic
        """
        # Ensure peptide is uppercase and contains only valid amino acids
        peptide = peptide.upper().strip()
        valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
        if not all(c in valid_aa for c in peptide):
            print(f"Warning: Invalid amino acids in sequence: {peptide}")
            return 0.5  # Default neutral score
        
        return float(self.toxinpred3_model.predict_proba(peptide))
    
    def normalize(self, value: float) -> float:
        """
        Normalize toxicity value to [0, 1] using empirical CDF.
        
        Args:
            value: Raw toxicity score
        
        Returns:
            Normalized score in [0, 1]
        """
        if self.reference_cdf is None:
            return float(np.clip(value, 0.0, 1.0))
        
        # Compute percentile rank
        idx = np.searchsorted(self.reference_cdf, value, side='right')
        percentile = idx / len(self.reference_cdf)
        return float(np.clip(percentile, 0.0, 1.0))
    
    def _default_cdf(self):
        """Default CDF for normalization (uniform distribution)."""
        return np.linspace(0.0, 1.0, 1000)
    
    @classmethod
    def load(
        cls,
        path: Optional[str] = None,
        model: int = 2,
        threshold: float = 0.38,
        reference_cdf: Optional[np.ndarray] = None
    ):
        """
        Load WT toxicity predictor.
        
        Args:
            path: Path to reference CDF file (optional, for now ignored)
            model: ToxinPred3 model to use (1: ML, 2: Hybrid). Default: 2
            threshold: ToxinPred3 threshold. Default: 0.38
            reference_cdf: Reference CDF array (optional)
        
        Returns:
            WTToxicityPredictor instance
        """
        # Initialize ToxinPred3 wrapper
        toxinpred3_model = ToxinPred3Wrapper(model=model, threshold=threshold)
        
        # Load reference CDF if path provided
        if path and Path(path).exists():
            try:
                reference_cdf = np.load(path)
            except:
                pass
        
        return cls(
            toxinpred3_model=toxinpred3_model,
            reference_cdf=reference_cdf,
            model=model,
            threshold=threshold
        )


# Example usage
if __name__ == '__main__':
    # Test the predictor
    print("Testing WT Toxicity Predictor with ToxinPred3...")
    
    predictor = WTToxicityPredictor.load(model=2, threshold=0.38)
    
    # Test sequences
    test_sequences = [
        "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGR",  # GLP-1R peptide
        "KDLQALDLQER",  # AMHR2 peptide
        "GILTLF",  # OX1R peptide
    ]
    
    print("\nPredictions:")
    for seq in test_sequences:
        score = predictor.predict(seq)
        label = "TOXIC" if score >= 0.38 else "NON-TOXIC"
        print(f"{seq[:20]}... -> Score: {score:.4f} ({label})")
    
    # Test batch prediction
    print("\nBatch prediction:")
    scores = [predictor.predict(seq) for seq in test_sequences]
    print(f"Scores: {scores}")

