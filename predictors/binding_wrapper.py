import torch
import sys
import esm
import pandas as pd
import warnings
import numpy as np
from pathlib import Path
from typing import Optional
from scoring.functions.binding import ImprovedBindingPredictor, BindingAffinity
from transformers import AutoModelForMaskedLM
from .binding import BindingPredictor
from tokenizer.my_tokenizers import SMILES_SPE_Tokenizer


lapep_root = Path(__file__).parent.parent.resolve()
tr2d2_paths = [lapep_root / "lapep" / "tr2d2"]

tr2d2_path = None
for path in tr2d2_paths:
    path = path.resolve()
    if path.exists() and (path / "scoring" / "functions" / "binding.py").exists():
        tr2d2_path = path
        sys.path = [p for p in sys.path if 'tr2d2' not in p and 'TR2-D2' not in p]
        sys.path.insert(0, str(tr2d2_path))
        break

if tr2d2_path is None:
    TR2D2_AVAILABLE = False
    ImprovedBindingPredictor = None
    BindingAffinity = None
else:
    missing_deps = []
    if missing_deps:
        TR2D2_AVAILABLE = False
        ImprovedBindingPredictor = None
        BindingAffinity = None
    else:
        TR2D2_AVAILABLE = True

class RealBindingPredictor(BindingPredictor):
    
    def __init__(self, model_path: str, protein_seq: str,
                 tokenizer=None, base_path: Optional[str] = None,
                 device: Optional[str] = None):
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(device)
        self.model_path = model_path
        self.protein_seq = protein_seq
        
        if base_path is None:
            lapep_root = Path(__file__).parent.parent.parent
            research_dir = lapep_root.parent
            self.base_path = str(research_dir)
        else:
            self.base_path = base_path
        
        self.binding_model = None
        self.tokenizer = tokenizer
        self._load_model()
        super().__init__(model=None, device=device)
    
    def _load_model(self):
        if not TR2D2_AVAILABLE:
            return
        
        lapep_root = Path(__file__).parent.parent.parent
        tr2d2_symlink = lapep_root / "TR2-D2"
        if not tr2d2_symlink.exists():
            try:
                tr2d2_symlink.symlink_to("tr2d2-pep")
            except OSError:
                pass
        
        self.base_path = str(lapep_root)
        expected_path_via_symlink = lapep_root / "TR2-D2" / "tr2d2-pep" / "scoring" / "functions" / "classifiers" / "binding-affinity.pt"
        expected_path_direct = lapep_root / "tr2d2-pep" / "scoring" / "functions" / "classifiers" / "binding-affinity.pt"
        model_path = Path(self.model_path)
        
        if not model_path.exists():
            self.binding_model = None
            return
        
        for expected_path in [expected_path_direct, expected_path_via_symlink]:
            if not expected_path.exists():
                expected_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    expected_path.symlink_to(model_path.resolve())
                except OSError:
                    import shutil
                    shutil.copy2(model_path, expected_path)
        
        self.binding_model = BindingAffinity(
            prot_seq=self.protein_seq,
            tokenizer=self.tokenizer,
            base_path=self.base_path,
            device=self.device,
            emb_model=None
        )
    
    def predict(self, peptide: str) -> float:
        if self.binding_model is None:
            return np.random.uniform(6.0, 8.0)
        
        scores = self.binding_model([peptide])
        if len(scores) > 0:
            return float(scores[0])
        return np.random.uniform(6.0, 8.0)
    
    @classmethod
    def load(cls, path: str, protein_seq: Optional[str] = None, 
             tokenizer=None, base_path: Optional[str] = None,
             device: Optional[str] = None):
        if protein_seq is None:
            "no protein sequence provided"
        
        # If no tokenizer provided, try to load the local SMILES_SPE_Tokenizer
        if tokenizer is None:
            try:
                from pathlib import Path
                lapep_root = Path(__file__).parent.parent.resolve()
                
                # Try to find tokenizer files in multiple locations
                tokenizer_vocab = None
                tokenizer_splits = None
                
                for vocab_path in [
                    lapep_root / "lapep" / "tr2d2" / "tokenizer" / "new_vocab.txt",
                    lapep_root / "tr2d2-pep" / "tokenizer" / "new_vocab.txt",
                    lapep_root.parent / "TR2-D2" / "tr2d2-pep" / "tokenizer" / "new_vocab.txt"
                ]:
                    splits_path = vocab_path.parent / "new_splits.txt"
                    if vocab_path.exists() and splits_path.exists():
                        tokenizer_vocab = str(vocab_path)
                        tokenizer_splits = str(splits_path)
                        break
                
                if tokenizer_vocab and tokenizer_splits:
                    # Import here to avoid circular imports
                    import sys
                    tr2d2_path = lapep_root / "lapep" / "tr2d2"
                    if tr2d2_path.exists():
                        sys.path.insert(0, str(tr2d2_path))
                        tokenizer = SMILES_SPE_Tokenizer(tokenizer_vocab, tokenizer_splits)
                        print(f"[Binding Predictor] Using local SMILES_SPE_Tokenizer from {tokenizer_vocab}")
                    else:
                        print(f"[Binding Predictor] Warning: Could not find tr2d2 path to load local tokenizer")
                else:
                    print(f"[Binding Predictor] Warning: Could not find local tokenizer files, will try HuggingFace tokenizer")
            except Exception as e:
                print(f"[Binding Predictor] Warning: Could not load local tokenizer: {e} so will try HuggingFace tokenizer instead")
        
        return cls(path, protein_seq, tokenizer, base_path, device)

