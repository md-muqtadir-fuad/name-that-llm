"""Own five-outer-fold plus final Qwen24 LoRA; fixed three-epoch recipe.

Runs only on Kaggle. Each process sees one independent GPU. Public attribution
adapters are forbidden inputs. Outer labels never select epochs or checkpoints.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse,gc,hashlib,json,math,random,time,traceback
from pathlib import Path
from importlib.metadata import version
assert Path('/kaggle/input').is_dir(),'Kaggle only.'
import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoConfig,AutoModel,AutoTokenizer
from peft import LoraConfig,get_peft_model
from safetensors.torch import save_file
from amp_update import successful_update

parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int,choices=[0,1],required=True)
args=parser.parse_args()
torch.set_num_threads(2)
assert torch.cuda.is_available() and torch.cuda.device_count()==1
assert torch.cuda.get_device_capability(0)==(7,5)
device=torch.device('cuda:0')
root=Path('/kaggle/working');inputs=Path('/kaggle/input')
spec=json.loads((root/'training_spec.json').read_text())
assert spec['authorized_diagnostic_smoke_required'] is True
plan=spec['plan'];labels=np.asarray(plan['labels'],dtype=str)
assert len(labels)==18 and spec['amp']==dict(initial_scale=256.,minimum_scale=1.,max_overflows_per_update=3,accumulation=6,growth_interval=2000)
data=[p for p in inputs.rglob('train.csv') if (p.parent/'test.csv').exists()]
assert len(data)==1 and hashlib.sha256(data[0].read_bytes()).hexdigest()==plan['train_sha256']
train=pd.read_csv(data[0],dtype=str)
assert len(train)==2699 and train.id.is_unique and train.notna().all().all()
assert not list(inputs.rglob('adapter_model.safetensors')),'Public attribution adapters forbidden.'
model_paths=[]
for path in inputs.rglob('config.json'):
    try:config_json=json.loads(path.read_text())
    except (OSError,ValueError):continue
    if (config_json.get('model_type')=='qwen3' and config_json.get('hidden_size')==4096
            and config_json.get('num_hidden_layers')==36 and config_json.get('intermediate_size')==12288
            and list(path.parent.glob('*.safetensors'))):model_paths.append(path.parent)
assert len(model_paths)==1
model_dir=model_paths[0]
assert hashlib.sha256((model_dir/'config.json').read_bytes()).hexdigest()==spec['smoke_evidence']['base_config_sha256']
tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
tokenizer.padding_side='right'
if tokenizer.pad_token is None:tokenizer.pad_token=tokenizer.eos_token
tokens=tokenizer(train.text.tolist(),add_special_tokens=False,truncation=True,max_length=447)['input_ids']
tokens=[ids+[tokenizer.eos_token_id] for ids in tokens]
assert min(map(len,tokens))>=2 and max(map(len,tokens))<=448
layers=[2,4,6,8,12,16,20,24]
target=np.searchsorted(labels,train.model.to_numpy())
groups=train.prompt_id.to_numpy(dtype=str)
all_jobs=list(spec['jobs'])
assigned=all_jobs[args.worker::2]
folder=root/'lora24';folder.mkdir(exist_ok=True)
worker_status=dict(worker=args.worker,jobs=assigned,completed=[],status='INITIALIZING',
                   dependencies={name:version(name) for name in ('torch','transformers','peft','accelerate','safetensors')},
                   cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),public_adapters_loaded=False)
status_path=folder/f'worker{args.worker}_status.json'
def write_status():status_path.write_text(json.dumps(worker_status,indent=2))
write_status()

def epoch_batches(fit,rng):
    batches=[];coverage=[]
    for group in np.unique(groups[fit]):
        members=rng.permutation(fit[groups[fit]==group])
        assert len(members)>=3
        for start in range(0,len(members),3):
            batch=members[start:start+3]
            if len(batch)<3:
                more=rng.choice(np.setdiff1d(members,batch),3-len(batch),replace=False)
                batch=np.concatenate([batch,more])
            assert len(batch)==3 and len(np.unique(batch))==3 and len(np.unique(groups[batch]))==1
            batches.append(batch);coverage.extend(batch.tolist())
    assert set(coverage)==set(fit),'Every fitting answer must occur in each epoch.'
    batches=[batches[i] for i in rng.permutation(len(batches))]
    original_count=len(batches)
    while len(batches)%6:batches.append(batches[int(rng.integers(original_count))].copy())
    return batches,dict(original_microbatches=original_count,padded_microbatches=len(batches)-original_count,
                        example_exposures=len(batches)*3,unique_fitting_examples=len(fit))

try:
    for name in assigned:
        fit=np.asarray(plan['jobs'][name]['fit'],int);valid=np.asarray(plan['jobs'][name]['valid'],int)
        assert not set(fit)&set(valid) and not set(groups[fit])&set(groups[valid])
        assert set(fit)|set(valid)==set(range(len(train)))
        assert np.array_equal(np.unique(target[fit]),np.arange(18))
        checkpoint=folder/name
        assert not checkpoint.exists(),'No implicit resume or reuse of a prior fitted checkpoint.'
        checkpoint.mkdir()
        worker_status.update(status='TRAINING',current_job=name);write_status()
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        rng=np.random.default_rng(42)
        planned=[epoch_batches(fit,rng) for _ in range(3)]
        total_updates=sum(len(batches)//6 for batches,_ in planned)
        assert total_updates<spec['amp']['growth_interval'],'Manual-scale growth-tracker caveat requires review.'
        config=AutoConfig.from_pretrained(model_dir,local_files_only=True)
        config.num_hidden_layers=24;config.use_cache=False
        if getattr(config,'layer_types',None) is not None:config.layer_types=config.layer_types[:24]
        load_started=time.monotonic()
        base=AutoModel.from_pretrained(model_dir,config=config,dtype=torch.float16,
                local_files_only=True,low_cpu_mem_usage=True,device_map={'':0},attn_implementation='sdpa')
        assert not hasattr(base,'lm_head') and all(p.device==device for p in base.parameters())
        base.requires_grad_(False)
        model=get_peft_model(base,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,bias='none',
                    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.enable_input_require_grads()
        for p in model.parameters():
            if p.requires_grad:p.data=p.data.float()
        norms=nn.ModuleList([nn.LayerNorm(4096) for _ in layers]).to(device).float()
        plain=nn.Linear(4096*len(layers),18).to(device).float()
        centered=nn.Linear(4096*len(layers),18).to(device).float()
        dropout=nn.Dropout(.1)
        adapter_parameters=[p for p in model.parameters() if p.requires_grad]
        head_parameters=list(norms.parameters())+list(plain.parameters())+list(centered.parameters())
        parameters=adapter_parameters+head_parameters
        assert all(p.dtype==torch.float32 for p in parameters)
        assert all('lora_' in key for key,p in model.named_parameters() if p.requires_grad)
        optimizer=torch.optim.AdamW([{'params':adapter_parameters,'lr':1e-4},
                                     {'params':head_parameters,'lr':1e-3}],weight_decay=.01)
        warmup=max(1,int(total_updates*.1))
        def schedule(step):
            if step<warmup:return (step+1)/warmup
            fraction=min(1.,(step-warmup)/max(1,total_updates-warmup))
            return .5*(1+math.cos(math.pi*fraction))
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
        scaler=torch.amp.GradScaler('cuda',init_scale=256.,growth_interval=2000)
        records=[];attempts=[];skipped=[]
        job_report=dict(job=name,train_sha256=plan['train_sha256'],classes=labels.tolist(),
                        fit_rows=len(fit),valid_rows=len(valid),fit_ids=train.id.iloc[fit].tolist(),
                        epochs=3,total_updates=total_updates,planned_epochs=[record for _,record in planned],
                        load_seconds=time.monotonic()-load_started,attempts=attempts,skipped_scales=skipped,
                        updates=records,public_adapters_loaded=False,pretrained_base_only=True,
                        config_sha256=hashlib.sha256((model_dir/'config.json').read_bytes()).hexdigest(),
                        training_spec_sha256=hashlib.sha256((root/'training_spec.json').read_bytes()).hexdigest())
        report_path=checkpoint/'training_report.json'
        def persist_attempt(event):
            attempts.append(event)
            if 'OVERFLOW' in event['status']:skipped.append(event.copy())
            report_path.write_text(json.dumps(job_report,indent=2))
            if event['status']!='SUCCESS':print(name,'AMP',json.dumps(event),flush=True)
        def representation(indices):
            batch=tokenizer.pad({'input_ids':[tokens[i] for i in indices]},return_tensors='pt').to(device)
            with torch.autocast('cuda',dtype=torch.float16):
                hidden=model(**batch,output_hidden_states=True,use_cache=False).hidden_states
            mask=batch.attention_mask.unsqueeze(-1).float();mask[:,0]=0
            with torch.autocast('cuda',enabled=False):
                z=torch.cat([norm((hidden[layer].float()*mask).sum(1)/mask.sum(1))
                             for layer,norm in zip(layers,norms)],-1)
            assert bool(torch.isfinite(z).all())
            return z,int(batch.attention_mask.sum()),int(batch.input_ids.numel())
        def forward_batch(indices):
            assert len(indices)==3 and len(np.unique(groups[indices]))==1
            z,n_tokens,n_padded=representation(indices)
            y=torch.as_tensor(target[indices],device=device,dtype=torch.long)
            with torch.autocast('cuda',enabled=False):
                loss_plain=nn.functional.cross_entropy(plain(dropout(z)),y)
                loss_center=nn.functional.cross_entropy(centered(dropout(z-z.mean(0,keepdim=True))),y)
            return loss_plain+loss_center,dict(plain_loss=float(loss_plain.detach()),
                        center_loss=float(loss_center.detach()),tokens=n_tokens,padded_tokens=n_padded)
        torch.cuda.reset_peak_memory_stats(device)
        model.train();norms.train();plain.train();centered.train()
        for epoch,(batches,_) in enumerate(planned):
            for start in range(0,len(batches),6):
                record=successful_update(optimizer,scaler,parameters,adapter_parameters,forward_batch,
                        batches[start:start+6],len(records)+1,spec['amp'],persist_attempt,device)
                record.update(epoch=epoch+1,learning_rates=[group['lr'] for group in optimizer.param_groups])
                records.append(record);scheduler.step()
                if len(records)%10==0:
                    report_path.write_text(json.dumps(job_report,indent=2))
                    print(json.dumps(dict(job=name,worker=args.worker,completed=len(records),
                                         total=total_updates,epoch=epoch+1,last=record)),flush=True)
        assert len(records)==total_updates
        # Fixed final epoch only; no outer-label checkpoint selection.
        model.eval();norms.eval();plain.eval();centered.eval()
        model.gradient_checkpointing_disable()
        def extract(rows):
            values=np.empty((len(rows),4096*len(layers)),np.float32)
            order=np.argsort([len(tokens[i]) for i in rows],kind='stable')
            with torch.inference_mode():
                for start in range(0,len(order),3):
                    positions=order[start:start+3]
                    z,_,_=representation(rows[positions])
                    values[positions]=z.cpu().numpy()
            assert np.isfinite(values).all()
            return values
        valid_z=extract(valid)
        # Stable portable affine heads; CPU matrix multiplication uses these
        # saved arrays later, without any original-label fitting.
        np.savez_compressed(checkpoint/'readouts.npz',classes=labels,
            plain_coef=plain.weight.detach().cpu().numpy(),plain_intercept=plain.bias.detach().cpu().numpy(),
            center_coef=centered.weight.detach().cpu().numpy(),center_intercept=centered.bias.detach().cpu().numpy(),
            train_sha256=plan['train_sha256'],pool_layers=np.array(layers))
        np.savez_compressed(checkpoint/'validation_features.npz',ids=train.id.iloc[valid].to_numpy(dtype=str),
                            classes=labels,features=valid_z,train_sha256=plan['train_sha256'])
        model.save_pretrained(checkpoint/'adapter',safe_serialization=True,save_embedding_layers=False)
        tensors={**{f'norms.{key}':value.detach().cpu().contiguous() for key,value in norms.state_dict().items()},
                 **{f'plain.{key}':value.detach().cpu().contiguous() for key,value in plain.state_dict().items()},
                 **{f'centered.{key}':value.detach().cpu().contiguous() for key,value in centered.state_dict().items()}}
        save_file(tensors,checkpoint/'heads.safetensors')
        if name=='final':
            anchors=np.arange(3)
            np.savez_compressed(checkpoint/'anchor_features.npz',ids=train.id.iloc[anchors].to_numpy(dtype=str),
                                classes=labels,features=extract(anchors),train_sha256=plan['train_sha256'])
            tokenizer.save_pretrained(checkpoint/'tokenizer');config.save_pretrained(checkpoint/'base_config')
        job_report.update(status='COMPLETE',successful_updates=len(records),
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                          successful_attempt_seconds=sum(row['seconds'] for row in records),
                          overflow_retry_seconds=sum(row['seconds'] for row in skipped),
                          checkpoints_selected_by='Fixed epoch3; no held-out labels')
        report_path.write_text(json.dumps(job_report,indent=2))
        (checkpoint/'COMPLETE.json').write_text(json.dumps(dict(job=name,train_sha256=plan['train_sha256'],
                files={str(path.relative_to(checkpoint)):hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in checkpoint.rglob('*') if path.is_file()}),indent=2))
        worker_status['completed'].append(name);write_status()
        print('JOB COMPLETE',name,flush=True)
        del base,model,norms,plain,centered,optimizer,scheduler,scaler,parameters,adapter_parameters,head_parameters,tensors,valid_z
        gc.collect();torch.cuda.empty_cache()
    worker_status.update(status='COMPLETE');write_status()
except Exception as error:
    worker_status.update(status='ERROR',error=repr(error),traceback=traceback.format_exc())
    write_status()
    raise
