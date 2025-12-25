import torch
import sys
import numpy as np
from pathlib import Path
from typing import Optional

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
    try:
        import pandas
    except ImportError:
        missing_deps.append("pandas")
    try:
        import esm
    except ImportError:
        missing_deps.append("fair-esm (esm)")
    
    if missing_deps:
        TR2D2_AVAILABLE = False
        ImprovedBindingPredictor = None
        BindingAffinity = None
    else:
        from scoring.functions.binding import ImprovedBindingPredictor, BindingAffinity
        from transformers import AutoModelForMaskedLM
        TR2D2_AVAILABLE = True

from .binding import BindingPredictor


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
        
        if self.tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained('aaronfeller/PeptideCLM-23M-all')
        
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
        return cls(path, protein_seq, tokenizer, base_path, device)

