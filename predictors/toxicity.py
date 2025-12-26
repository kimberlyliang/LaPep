import numpy as np
import torch
import torch.nn as nn
from typing import Optional
from pathlib import Path
from transformers import AutoModelForMaskedLM
import warnings
import os


# Matches your training definition :contentReference[oaicite:5]{index=5}
class TransformerClassifier(nn.Module):
    def __init__(self, d_model=256, nhead=8, layers=2, ff=512, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(768, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, X, M):
        # X: (B,L,768), M: (B,L) bool, True=valid token, False=pad/special
        pad_mask = ~M  # True = ignore
        Z = self.proj(X)
        Z = self.enc(Z, src_key_padding_mask=pad_mask)

        Mf = M.unsqueeze(-1).float()
        denom = Mf.sum(dim=1).clamp(min=1.0)
        pooled = (Z * Mf).sum(dim=1) / denom
        return self.head(pooled).squeeze(-1)  # logits


class Toxicity:
    """
    Usage:
        tox = Toxicity(
            ckpt_path="/path/to/best_model.pt",
            device="cuda:0",
        )
        p = tox("CC(=O)NCCC1=CNc2c1cc(OC)cc2")  # SMILES -> P(toxic)
        ps = tox(["CCO", "CC(=O)O"])           # batch -> np.ndarray of probs
        y = tox.predict_label("CCO", threshold=0.5)
    """

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda:0",
        max_length: int = 768,
        tokenizer_vocab: Optional[str] = None,
        tokenizer_splits: Optional[str] = None,
        embedding_model_name: str = "aaronfeller/PeptideCLM-23M-all",
    ):
        # device
        if "cuda" in device and torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")

        self.max_length = int(max_length)

        # --- Find tokenizer paths automatically if not provided
        if tokenizer_vocab is None or tokenizer_splits is None:
            tokenizer_vocab, tokenizer_splits = self._find_tokenizer_paths()
        
        # --- tokenizer + embedding model (same components you used) :contentReference[oaicite:6]{index=6}
        # Import tokenizer after adding path to sys.path
        import sys
        from pathlib import Path as PathLib
        lapep_root = PathLib(__file__).parent.parent.resolve()
        tr2d2_path = lapep_root / "lapep" / "tr2d2"
        if tr2d2_path.exists():
            sys.path.insert(0, str(tr2d2_path))
        from tokenizer.my_tokenizers import SMILES_SPE_Tokenizer
        self.tokenizer = SMILES_SPE_Tokenizer(tokenizer_vocab, tokenizer_splits)
        
        # Set up cache directory for HuggingFace models
        cache_dir = os.environ.get('HF_HOME') or os.environ.get('TRANSFORMERS_CACHE')
        if cache_dir is None and os.path.exists('/scratch'):
            import getpass
            username = getpass.getuser()
            possible_dirs = [
                f'/scratch/pranamlab/{username}/.cache/huggingface',
                f'/scratch/{username}/.cache/huggingface',
                os.path.join(os.getcwd(), '.cache', 'huggingface'),
            ]
            for possible_dir in possible_dirs:
                try:
                    os.makedirs(possible_dir, exist_ok=True)
                    test_file = os.path.join(possible_dir, '.test_write')
                    with open(test_file, 'w') as f:
                        f.write('test')
                    os.remove(test_file)
                    cache_dir = possible_dir
                    os.environ['HF_HOME'] = cache_dir
                    os.environ['TRANSFORMERS_CACHE'] = cache_dir
                    break
                except (OSError, PermissionError):
                    continue
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=FutureWarning)
            warnings.filterwarnings('ignore', message='.*GenerationMixin.*')
            warnings.filterwarnings('ignore', message='.*doesn\'t directly inherit.*')
            warnings.filterwarnings('ignore', message='.*RoFormerForMaskedLM.*')
            try:
                embedding_model = AutoModelForMaskedLM.from_pretrained(
                    embedding_model_name,
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                    use_safetensors=True
                )
            except Exception as e:
                print(f"Warning: Safetensors loading failed for PeptideCLM model: {e}. Trying without safetensors.")
                embedding_model = AutoModelForMaskedLM.from_pretrained(
                    embedding_model_name,
                    trust_remote_code=True,
                    cache_dir=cache_dir
                )
        self.embedding_model = embedding_model.roformer
        self.embedding_model.to(self.device).eval()

        # special token ids (same idea as your script) :contentReference[oaicite:7]{index=7}
        self.special_ids = self._get_special_ids(self.tokenizer)
        self.special_ids_t = (
            torch.tensor(self.special_ids, device=self.device, dtype=torch.long)
            if len(self.special_ids) > 0
            else None
        )

        # --- classifier checkpoint (same save format) :contentReference[oaicite:8]{index=8}
        map_location = "cuda" if "cuda" in str(self.device) else "cpu"
        ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        best_params = ckpt["best_params"]
        self.best_params = dict(best_params)

        self.classifier = TransformerClassifier(
            d_model=int(best_params["d_model"]),
            nhead=int(best_params["nhead"]),
            layers=int(best_params["layers"]),
            ff=int(best_params["ff"]),
            dropout=float(best_params.get("dropout", 0.1)),
        )
        self.classifier.load_state_dict(ckpt["state_dict"])
        self.classifier.to(self.device).eval()

    @staticmethod
    def _find_tokenizer_paths():
        """Find tokenizer vocab and splits files automatically."""
        from pathlib import Path
        lapep_root = Path(__file__).parent.parent.resolve()
        
        vocab_path = lapep_root / "lapep" / "tr2d2" / "tokenizer" / "new_vocab.txt"
        splits_path = vocab_path.parent / "new_splits.txt"
        if vocab_path.exists() and splits_path.exists():
            return str(vocab_path), str(splits_path)
        
        # Fallback to default paths if not found
        raise FileNotFoundError(
            f"Could not find tokenizer files. Please ensure new_vocab.txt and new_splits.txt exist "
            f"in one of the expected locations."
        )
    
    @staticmethod
    def _get_special_ids(tokenizer):
        cand = [
            getattr(tokenizer, "pad_token_id", None),
            getattr(tokenizer, "cls_token_id", None),
            getattr(tokenizer, "sep_token_id", None),
            getattr(tokenizer, "bos_token_id", None),
            getattr(tokenizer, "eos_token_id", None),
            getattr(tokenizer, "mask_token_id", None),
        ]
        return sorted({x for x in cand if x is not None})

    def _isin_fallback(self, input_ids: torch.Tensor, special_ids_t: torch.Tensor):
        # Returns (B,L) bool: True where input_ids is in special_ids_t
        # Works on old PyTorch that lacks torch.isin
        return (input_ids.unsqueeze(-1) == special_ids_t.view(1, 1, -1)).any(dim=-1)

    @torch.no_grad()
    def _embed_unpooled(self, smiles_list):
        """
        Returns:
            X: (B, L, 768) float32
            M: (B, L) bool (True=valid token; False=pad or special)
        """
        tok = self.tokenizer(
            smiles_list,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True,
        )
        input_ids = tok["input_ids"].to(self.device)                 # (B,L)
        attention_mask = tok["attention_mask"].to(self.device).bool()# (B,L)

        out = self.embedding_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state                           # (B,L,768)

        valid = attention_mask
        if self.special_ids_t is not None and self.special_ids_t.numel() > 0:
            if hasattr(torch, "isin"):
                is_special = torch.isin(input_ids, self.special_ids_t)
            else:
                is_special = self._isin_fallback(input_ids, self.special_ids_t)
            valid = valid & (~is_special)

        # Keep full (B,L,768) but mask out pads/specials in M
        X = last_hidden.float()   # float32 for safety on CPU/GPU
        M = valid
        return X, M
    
    @torch.no_grad()
    def predict_proba(self, smiles):
        single = isinstance(smiles, str)
        smiles_list = [smiles] if single else list(smiles)

        X, M = self._embed_unpooled(smiles_list)
        logits = self.classifier(X, M)            # (B,)
        probs = torch.sigmoid(logits).detach().cpu().numpy()

        return float(probs[0]) if single else probs
    
    def predict_label(self, smiles, threshold = 0.5):
        p = self.predict_proba(smiles)
        if isinstance(p, float):
            return int(p >= threshold)
        return (p >= threshold).astype(np.int64)

    def __call__(self, smiles):
        return self.predict_proba(smiles)


class ToxicityPredictor:
    """
    Wraps the Toxicity class to provide predict() and normalize() methods.
    """
    
    def __init__(self, toxicity_model: Optional[Toxicity] = None, reference_cdf: Optional[np.ndarray] = None):
        self.toxicity_model = toxicity_model
        self.reference_cdf = reference_cdf or self._default_cdf()
    
    def predict(self, peptide: str) -> float:
        """
        Predict toxicity score for a peptide.
        Returns P(toxic) in [0, 1] where 1 = most toxic.
        """
        if self.toxicity_model is None:
            return np.random.uniform(0.0, 1.0)
        return float(self.toxicity_model.predict_proba(peptide))
    
    def normalize(self, value: float) -> float:
        """
        Normalize toxicity value to [0, 1] range.
        For toxicity, lower is better, but we normalize the raw probability.
        """
        if self.reference_cdf is None:
            return float(np.clip(value, 0.0, 1.0))
        percentile = np.searchsorted(self.reference_cdf, value) / len(self.reference_cdf)
        return float(np.clip(percentile, 0.0, 1.0))
    
    def _default_cdf(self):
        """Default CDF for normalization (uniform distribution)."""
        return np.linspace(0.0, 1.0, 1000)
    
    @classmethod
    def load(cls, path: str, device: Optional[str] = None):
        """
        Load toxicity predictor from checkpoint file.
        
        Args:
            path: Path to the checkpoint file (.pt file)
            device: Device to load model on (e.g., 'cuda:0', 'cpu')
        
        Returns:
            ToxicityPredictor instance
        """
        from pathlib import Path as PathLib
        
        model_path = PathLib(path)
        if not model_path.exists():
            print(f"Warning: Toxicity predictor file not found: {path}")
            print("Toxicity predictor will return random scores.")
            return cls(toxicity_model=None, reference_cdf=None)
        
        # Determine device
        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        try:
            # Load the Toxicity model
            toxicity_model = Toxicity(
                ckpt_path=str(path),
                device=device
            )
            print(f"[Toxicity Predictor] Loaded successfully from {path}")
            print(f"[Toxicity Predictor] Using device: {toxicity_model.device}")
            
            return cls(toxicity_model=toxicity_model, reference_cdf=None)
        except Exception as e:
            print(f"Warning: Could not load toxicity predictor from {path}: {e}")
            print("Toxicity predictor will return random scores.")
            import traceback
            traceback.print_exc()
            return cls(toxicity_model=None, reference_cdf=None)


if __name__ == '__main__':
    tox = Toxicity(
        ckpt_path="/scratch/pranamlab/tong/PeptiVerse/src/toxicity/transformer_50/best_model.pt",
        device="cuda:0",
    )
    
    p = tox("CC(=O)NCCC1=CNc2c1cc(OC)cc2")  # SMILES -> P(toxic)
    ps = tox(["CCO", "CC(=O)O"])           # batch -> np.ndarray of probs
    y = tox.predict_label("CCO", threshold=0.5)
    print(p, ps, y)