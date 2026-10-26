# Downloads meta-llama/Llama-3.1-8B (base) from Hugging Face into this kernel's output, so training / inference kernels can
# attach it (mounted path contains 'llama-3.1/8b'). Needs the Kaggle secret HF_TOKEN attached to this notebook
# (editor: Add-ons -> Secrets). CPU session, internet on; the token is read at run time and never printed.
from pathlib import Path
from kaggle_secrets import UserSecretsClient
from huggingface_hub import snapshot_download

out = Path('/kaggle/working/llama-3.1/8b')
snapshot_download('meta-llama/Llama-3.1-8B', local_dir=str(out), token=UserSecretsClient().get_secret('HF_TOKEN'),
                  allow_patterns=['config.json', 'generation_config.json', '*.safetensors', 'model.safetensors.index.json',
                                  'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json'])
files = sorted(p.name for p in out.iterdir() if p.is_file())
print(files, round(sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) / 1e9, 1), 'GB')
assert 'config.json' in files and any(f.endswith('.safetensors') for f in files)
