import xgboost as xgb
import torch
import numpy as np
from transformers import AutoModelForMaskedLM
from tokenizer.my_tokenizers import SMILES_SPE_Tokenizer

class Halflife:
    def __init__(self, device=None, apply_log1p=True):
        self.apply_log1p = apply_log1p
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load xgboost model
        self.predictor = xgb.Booster(model_file="/scratch/pranamlab/tong/PeptiVerse/src/smiles_halflife/xgb/smiles_halflife_best_xgboost.json")

        # Load embedding model + tokenizer (match what you used in training)
        base = AutoModelForMaskedLM.from_pretrained("aaronfeller/PeptideCLM-23M-all")
        self.emb_model = base.roformer.to(self.device).eval()

        self.tokenizer = SMILES_SPE_Tokenizer(
            "/scratch/pranamlab/tong/PeptiVerse/functions/tokenizer/new_vocab.txt",
            "/scratch/pranamlab/tong/PeptiVerse/functions/tokenizer/new_splits.txt",
        )

    @torch.no_grad()
    def generate_embeddings(self, sequences):
        embs = []
        for s in sequences:
            toks = self.tokenizer(s, return_tensors="pt")
            toks = {k: v.to(self.device) for k, v in toks.items()}

            out = self.emb_model(**toks)
            emb = out.last_hidden_state.mean(dim=1).squeeze(0).detach().cpu().numpy().astype(np.float32)
            embs.append(emb)

        if len(embs) == 0:
            return np.zeros((0, 768), dtype=np.float32)
        return np.stack(embs, axis=0)

    def predict_log1p(self, input_seqs):
        X = self.generate_embeddings(input_seqs)
        if X.shape[0] == 0:
            return np.array([], dtype=np.float32)

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        dmat = xgb.DMatrix(X)
        pred = self.predictor.predict(dmat).astype(np.float32)  # regression output
        return pred

    def predict_hours(self, input_seqs):
        pred = self.predict_log1p(input_seqs)
        if self.apply_log1p:
            return np.expm1(pred)  # convert log1p(hours) -> hours
        return pred

    def __call__(self, input_seqs):
        return self.predict_hours(input_seqs)

def unittest():
    halflife = Halflife(apply_log1p=True)
    seq = ["[C@@H](CCC(=O)O)NC(=O)[C@@H](CC(C)C)NC(=O)[C@@H](C)NC(=O)CNC(=O)[C@@H]([C@H](C)CC)NC(=O)CNC(=O)[C@@H]([C@H](C)CC)NC(=O)[C@@H](CC(C)C)NC(=O)[C@@H]([C@@H](C)O)NC(=O)[C@@H](C(C)C)C(=O)O"]
    pred_hours = halflife(seq)
    print("pred_hours:", pred_hours)

if __name__ == "__main__":
    unittest()
