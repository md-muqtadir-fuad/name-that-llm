"""Bounded AMP updates shared by the smoke and full training workers."""
import time
import torch


def successful_update(optimizer,scaler,parameters,adapter_parameters,forward_batch,
                      batches,number,configuration,on_attempt,device):
    """Repeat only scaled-gradient overflows; never step the optimizer on them.

    Batches and dropout RNG are replayed identically. Nonfinite forward values
    fail immediately. Every overflow/successful attempt is emitted; other
    forward/backward/clipping exceptions propagate to the caller's error report.
    """
    assert len(batches)==configuration['accumulation']==6
    max_overflows=configuration['max_overflows_per_update']
    minimum_scale=configuration['minimum_scale']
    cpu_rng=torch.get_rng_state()
    cuda_rng=torch.cuda.get_rng_state(device)
    total_started=time.monotonic()
    retry_seconds=0.;overflows=0;attempt=0
    while True:
        attempt+=1
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng,device)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        started=time.monotonic()
        scale_before=float(scaler.get_scale())
        losses=[];plain_losses=[];center_losses=[];tokens=0;padded_tokens=0
        for batch in batches:
            loss,metrics=forward_batch(batch)
            assert bool(torch.isfinite(loss)),'Nonfinite forward loss; no AMP retry.'
            losses.append(float(loss.detach()))
            plain_losses.append(metrics['plain_loss'])
            center_losses.append(metrics['center_loss'])
            tokens+=metrics['tokens'];padded_tokens+=metrics['padded_tokens']
            scaler.scale(loss/len(batches)).backward()
            del loss,metrics
        scaler.unscale_(optimizer)
        finite=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in parameters)
        if not finite:
            overflows+=1
            next_scale=scale_before*.5
            terminal=(overflows>max_overflows or next_scale<minimum_scale or scale_before<=minimum_scale)
            # Explicit public GradScaler.update(new_scale=...) resets its
            # optimizer stage without ever calling optimizer.step on overflow.
            if not terminal:scaler.update(new_scale=next_scale)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            elapsed=time.monotonic()-started
            retry_seconds+=elapsed
            event=dict(update=number,attempt=attempt,status='TERMINAL_OVERFLOW' if terminal else 'OVERFLOW_RETRY',
                       finite_gradients=False,optimizer_step_performed=False,seconds=elapsed,
                       overflow_count=overflows,scale_before=scale_before,
                       scale_after=float(scaler.get_scale()),minimum_scale=minimum_scale)
            on_attempt(event)
            if terminal:
                raise FloatingPointError(f'Update{number}: overflow limit/minimum scale reached: {event}')
            continue
        assert any(p.grad is not None and bool(p.grad.abs().max()>0) for p in adapter_parameters),'Adapter gradient missing.'
        gradient_norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True))
        scaler.step(optimizer);scaler.update()
        torch.cuda.synchronize(device)
        elapsed=time.monotonic()-started
        event=dict(update=number,attempt=attempt,status='SUCCESS',finite_gradients=True,
                   optimizer_step_performed=True,seconds=elapsed,overflow_count=overflows,
                   scale_before=scale_before,scale_after=float(scaler.get_scale()))
        on_attempt(event)
        return dict(update=number,seconds=elapsed,successful_attempt_seconds=elapsed,
                    retry_seconds=retry_seconds,total_update_wall_seconds=time.monotonic()-total_started,
                    attempts=attempt,skipped_overflows=overflows,loss=sum(losses)/len(losses),
                    plain_loss=sum(plain_losses)/len(plain_losses),center_loss=sum(center_losses)/len(center_losses),
                    gradient_norm=gradient_norm,finite_gradients=True,
                    grad_scaler_before=scale_before,grad_scaler_after=float(scaler.get_scale()),
                    tokens=tokens,padded_tokens=padded_tokens,tokens_per_second=tokens/elapsed,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
