from tokenizer.my_tokenizers import SMILES_SPE_Tokenizer
from transformers import AutoModelForMaskedLM
import numpy as np
from scoring.functions.binding import BindingAffinity
from scoring.functions.permeability import Permeability
from scoring.functions.solubility import Solubility
from scoring.functions.hemolysis import Hemolysis
from scoring.functions.nonfouling import Nonfouling
from pathlib import Path

# Auto-detect base_path from current file location
# This file is at: lapep/tr2d2/scoring/scoring_functions.py
# base_path should point to the Research directory (parent of LaPep)
_current_file = Path(__file__).resolve()
# Go from lapep/tr2d2/scoring/scoring_functions.py -> Research directory
# scoring -> tr2d2 -> lapep -> LaPep -> Research
_lapep_root = _current_file.parent.parent.parent.parent  # LaPep project root
_research_dir = _lapep_root.parent  # Research directory
base_path = str(_research_dir)

class ScoringFunctions:
    def __init__(self, score_func_names=None, prot_seqs=None, device=None):
        """
        Class for generating score vectors given generated sequence

        Args:
            score_func_names: list of scoring function names to be evaluated
            score_weights: weights to scale scores (default: 1)
            target_protein: sequence of target protein binder
        """
        import warnings
        import os
        
        # Get cache directory - ensure it's never None and use scratch if available
        cache_dir = os.environ.get('HF_HOME') or os.environ.get('TRANSFORMERS_CACHE') or os.environ.get('HF_HUB_CACHE')
        
        # If no cache dir set and we're on a system with /scratch, use it
        if cache_dir is None:
            if os.path.exists('/scratch'):
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
        
        if cache_dir is None:
            cache_dir = os.path.expanduser('~/.cache/huggingface')
            os.makedirs(cache_dir, exist_ok=True)
            os.environ['HF_HOME'] = cache_dir
        
        # Suppress deprecation warnings about GenerationMixin
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=FutureWarning)
            warnings.filterwarnings('ignore', message='.*GenerationMixin.*')
            warnings.filterwarnings('ignore', message='.*doesn\'t directly inherit.*')
            warnings.filterwarnings('ignore', message='.*RoFormerForMaskedLM.*')
            
            # Try to use safetensors to avoid torch.load security issue
            try:
                emb_model = AutoModelForMaskedLM.from_pretrained(
                    'aaronfeller/PeptideCLM-23M-all',
                    cache_dir=cache_dir,
                    use_safetensors=True  # Prefer safetensors format
                ).roformer.to(device).eval()
            except Exception as e:
                # Fallback: try without safetensors (may fail if torch < 2.6)
                print(f"Warning: Could not load with safetensors: {e}")
                print("Attempting to load without safetensors (may require torch >= 2.6)...")
                emb_model = AutoModelForMaskedLM.from_pretrained(
                    'aaronfeller/PeptideCLM-23M-all',
                    cache_dir=cache_dir,
                    use_safetensors=False
                ).roformer.to(device).eval()
        
        # Try to find tokenizer files - check multiple locations
        tokenizer_vocab = None
        tokenizer_splits = None
        
        # Check in lapep/tr2d2/tokenizer/ first (local copy)
        local_vocab = _current_file.parent.parent / "tokenizer" / "new_vocab.txt"
        local_splits = _current_file.parent.parent / "tokenizer" / "new_splits.txt"
        
        if local_vocab.exists() and local_splits.exists():
            tokenizer_vocab = str(local_vocab)
            tokenizer_splits = str(local_splits)
        else:
            # Fallback to original TR2-D2 location
            tokenizer_vocab = f'{base_path}/TR2-D2/tr2d2-pep/tokenizer/new_vocab.txt'
            tokenizer_splits = f'{base_path}/TR2-D2/tr2d2-pep/tokenizer/new_splits.txt'
        
        tokenizer = SMILES_SPE_Tokenizer(tokenizer_vocab, tokenizer_splits)
        prot_seqs = prot_seqs if prot_seqs is not None else []
        
        if score_func_names is None:
            # just do unmasking based on validity of peptide bonds
            self.score_func_names = []
        else:
            self.score_func_names = score_func_names
                
        # self.weights = np.array([1] * len(self.score_func_names) if score_weights is None else score_weights)
        
        # binding affinities
        self.target_protein = prot_seqs
        print(len(prot_seqs))
        
        if ('binding_affinity1' in score_func_names) and (len(prot_seqs) == 1):
            binding_affinity1 = BindingAffinity(prot_seqs[0], tokenizer=tokenizer, base_path=base_path, device=device)
            binding_affinity2 = None
        elif ('binding_affinity1' in score_func_names) and ('binding_affinity2' in score_func_names) and (len(prot_seqs) == 2):
            binding_affinity1 = BindingAffinity(prot_seqs[0], tokenizer=tokenizer, base_path=base_path, device=device)
            binding_affinity2 = BindingAffinity(prot_seqs[1], tokenizer=tokenizer, base_path=base_path, device=device)
        else:
            print("here")
            binding_affinity1 = None
            binding_affinity2 = None

        permeability = Permeability(tokenizer=tokenizer, base_path=base_path, device=device, emb_model=emb_model)
        sol = Solubility(tokenizer=tokenizer, base_path=base_path, device=device, emb_model=emb_model)
        nonfouling = Nonfouling(tokenizer=tokenizer, base_path=base_path, device=device, emb_model=emb_model)
        hemo = Hemolysis(tokenizer=tokenizer, base_path=base_path, device=device, emb_model=emb_model)

        self.all_funcs = {'binding_affinity1': binding_affinity1,
                          'binding_affinity2': binding_affinity2,
                          'permeability': permeability,
                          'nonfouling': nonfouling, 
                          'solubility': sol, 
                          'hemolysis': hemo
                          } 
        
    def forward(self, input_seqs):
        scores = []
        
        for i, score_func in enumerate(self.score_func_names): 
            score = self.all_funcs[score_func](input_seqs = input_seqs)
        
            scores.append(score)
            
        # convert to numpy arrays with shape (num_sequences, num_functions)
        scores = np.float32(scores).T
        
        return scores
    
    def __call__(self, input_seqs: list):
        return self.forward(input_seqs)

