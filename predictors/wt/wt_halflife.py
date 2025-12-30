import xgboost as xgb
import torch
import numpy as np
from transformers import AutoModelForMaskedLM
from tokenizer.my_tokenizers import SMILES_SPE_Tokenizer

class Halflife:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load xgboost model
        self.predictor = xgb.Booster(model_file="/scratch/pranamlab/tong/PeptiVerse/src/halflife/finetune_stability_xgb_raw/best_model.json")

        # Load embedding model + tokenizer (match what you used in training) - might need to edit this!!
        base = AutoModelForMaskedLM.from_pretrained("aaronfeller/PeptideCLM-23M-all")
        self.emb_model = base.roformer.to(self.device).eval()

        # grab the tokenizer that was used in training 
        self.tokenizer = SMILES_SPE_Tokenizer(
            "/scratch/pranamlab/tong/PeptiVerse/functions/tokenizer/new_vocab.txt",
            "/scratch/pranamlab/tong/PeptiVerse/functions/tokenizer/new_splits.txt",
        )

    @torch.no_grad()
    def generate_embeddings(self, sequences):
        embs = []
        for s in sequences:
            toks = self.tokenizer(s, return_tensors="pt")
            # move all of the tokens to the training device 
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
        return pred

    def __call__(self, input_seqs):
        return self.predict_hours(input_seqs)

def unittest():
    halflife = Halflife()
    seq = ["ELAGIGILTV"]
    pred_hours = halflife(seq)
    print("pred_hours:", pred_hours)

if __name__ == "__main__":
    unittest()
