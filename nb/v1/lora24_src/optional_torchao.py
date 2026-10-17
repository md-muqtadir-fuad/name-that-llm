"""Remove only the incompatible, unused torchao package in a Kaggle smoke VM.

PEFT 0.20.0 probes this optional dispatcher even for ordinary FP16 Linear
layers. No quantization is used by this experiment. No packages are installed.
"""
import importlib.metadata as metadata
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from packaging.version import Version

assert Path('/kaggle/input').is_dir(), 'Kaggle-only environment preflight.'
assert not any(name.split('.')[0] in {'torch', 'transformers', 'peft'}
               for name in sys.modules), 'Run before ML library imports.'
out = Path('/kaggle/working/optional_torchao_audit.json')
core = ('torch', 'transformers', 'peft', 'accelerate', 'safetensors', 'numpy', 'pandas')
before = {name: metadata.version(name) for name in core}
report = dict(status='INITIALIZING', core_versions_before=before,
              network_install_attempted=False, packages_installed=[],
              packages_removed=[], ordinary_unquantized_base_required=True)

try:
    configs = []
    for path in Path('/kaggle/input').rglob('config.json'):
        try:
            config = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if (config.get('model_type') == 'qwen3' and config.get('hidden_size') == 4096
                and config.get('num_hidden_layers') == 36
                and list(path.parent.glob('*.safetensors'))):
            configs.append(config)
    assert len(configs) == 1, 'Require the sole official Qwen3-8B base.'
    assert not configs[0].get('quantization_config'), 'Never disable a required quantizer.'
    assert not list(Path('/kaggle/input').rglob('adapter_model.safetensors'))
    try:
        installed = metadata.version('torchao')
    except metadata.PackageNotFoundError:
        installed = None
    report['torchao_version_before'] = installed
    if installed is not None and Version(installed) < Version('0.16.0'):
        assert before['peft'] == '0.20.0', 'Review the dispatcher for another PEFT release.'
        process = subprocess.run(
            [sys.executable, '-m', 'pip', '--disable-pip-version-check',
             'uninstall', '--yes', 'torchao'],
            text=True, capture_output=True, timeout=120,
        )
        report['uninstall_returncode'] = process.returncode
        report['uninstall_stdout'] = process.stdout[-4000:]
        report['uninstall_stderr'] = process.stderr[-4000:]
        assert process.returncode == 0, 'Optional torchao uninstall failed.'
        try:
            metadata.version('torchao')
        except metadata.PackageNotFoundError:
            pass
        else:
            raise AssertionError('torchao distribution remains installed.')
        assert importlib.util.find_spec('torchao') is None, 'torchao import remains available.'
        report['packages_removed'] = ['torchao']
        report['status'] = 'REMOVED_INCOMPATIBLE_OPTIONAL_TORCHAO'
    else:
        report['status'] = 'NO_CHANGE_REQUIRED'
    after = {name: metadata.version(name) for name in core}
    report['core_versions_after'] = after
    assert before == after, 'Core dependency versions changed.'
    report['core_versions_unchanged'] = True
except Exception as error:
    report.update(status='ERROR', error=repr(error))
    raise
finally:
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
