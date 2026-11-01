# AfroXLMR-large on v8 data (free Colab T4). If Colab disconnects: reconnect and run this cell again; finished steps are skipped.
import os, sys, shutil, zipfile, json
from google.colab import drive
drive.mount('/content/drive')
D = '/content/drive/MyDrive/domain_classifier_colab'   # domain_v8_colab.zip goes here
W = '/content/work'
R = f'{D}/runs_large_v8'                              # results stay on Drive
os.makedirs(D, exist_ok=True)
if not os.path.exists(f'{D}/domain_v8_colab.zip'):   # not on Drive yet: upload the two parts here (each under 10 MB)
    from google.colab import files
    got = {}
    while len([k for k in got if '.part' in k]) < 2:
        print('Choose domain_v8_colab.zip.part1 / .part2'); got.update(files.upload())
    with open(f'{D}/domain_v8_colab.zip', 'wb') as f:
        for k in sorted(k for k in got if '.part' in k):
            f.write(got[k])
    print('saved', f'{D}/domain_v8_colab.zip', os.path.getsize(f'{D}/domain_v8_colab.zip'), 'bytes')
if not os.path.exists(f'{W}/train_afroxlmr_domain.py'):
    zipfile.ZipFile(f'{D}/domain_v8_colab.zip').extractall(W)
os.makedirs(R, exist_ok=True)
!nvidia-smi --query-gpu=name,memory.total --format=csv

LARGE = 'Davlan/afro-xlmr-large'
COMMON = ['--head', 'multi6', '--mix-rate', '0.1', '--batch', '8', '--grad-accum', '2', '--grad-checkpointing']
S1 = ['--stage', '1', '--model', LARGE, '--epochs', '2', '--lr', '1e-5', '--cap-per-cell', '8000', *COMMON]
S2 = ['--stage', '2', '--epochs', '4', '--lr', '1e-5', '--eval-every', '0.5', '--patience', '2', '--replay-per-cell', '200',
      '--gold-share', '0.6', '--warmup', '0.1', *COMMON]

def step(name, args, keep_model=False):
    out = f'{R}/{name}'
    if os.path.exists(f'{out}/summary.json'):
        print('already done:', name); return
    local = f'{W}/runs/{name}'
    cmd = ' '.join(['cd', W, '&&', sys.executable, 'train_afroxlmr_domain.py', '--data-dir', 'data_v8', '--out-dir', local, *args])
    print('>>', name); get_ipython().system(cmd)
    assert os.path.exists(f'{local}/summary.json'), f'{name} failed: see the log above'
    os.makedirs(out, exist_ok=True)
    if keep_model:
        shutil.copytree(f'{local}/model', f'{out}/model', dirs_exist_ok=True)
    for f in ['eval_probs.npy', 'eval_ids.json', 'train_log.json', 'summary.json']:   # summary.json last = step finished
        shutil.copy(f'{local}/{f}', out)
    shutil.rmtree(local, ignore_errors=True)
    print('saved to Drive:', out)

step('stage1', S1, keep_model=True)                        # about 2 hours on a T4
for k in [0, 1, 2, 3, 4, -1]:                              # about 25 minutes each
    step(f'fold{k}' if k >= 0 else 'final', S2 + ['--init-from', f'{R}/stage1/model', '--fold', str(k)], keep_model=(k == -1))
print('ALL DONE')

# Small results file (no model weights) for scoring: My Drive/domain_classifier_colab/runs_large_v8_results.zip
with zipfile.ZipFile(f'{D}/runs_large_v8_results.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(R):
        for f in files:
            if '/model' not in root.replace(os.sep, '/'):
                z.write(os.path.join(root, f), os.path.relpath(os.path.join(root, f), R))
print(open(f'{R}/final/summary.json').read() if os.path.exists(f'{R}/final/summary.json') else 'final not finished yet')