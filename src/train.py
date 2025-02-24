import os
import functools

from fp16 import zero_grad
from .resample import UniformSampler, LossSecondMomentResampler

INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:

    def __init__(
        self,
        *,
        model,
        diffussion,
        data,
        batch_size,
        microbatch,
        lr,
        save_interval,
        use_fp16=False,
        schedule_sampler=None,
        lr_anneal_steps=0
    ) -> None:
        self.model = model
        self.diffussion = diffussion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch
        self.lr = lr
        self.save_interval = save_interval
        self.use_fp16 = use_fp16
        self.schedule_sampler = schedule_sampler or UniformSampler(diffussion)
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0

        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE

        self.model_params = list(self.model.parameters())

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch = next(self.data)
            self.run_step(batch)
            if self.step % self.log_interval == 0:
                self.save()

                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1

    def run_step(self, batch, cond=None):
        self.forward_backward(batch)
        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()

    def forward_backward(self, batch, cond=None):
        zero_grad(self.model_params)
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch]
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.scheduler_sampler.sample(micro.shape[0])

            compute_losses = functools.partial(
                self.diffussion.training_losses, self.model, micro, t
            )

            if last_batch:
                losses = compute_losses()

            if isinstance(self.schedule_sampler, LossSecondMomentResampler):
                self.schedule_sampler.update_with_all_losses(t, losses["loss"].detach())

            loss = (losses["loss"] * weights).mean()

            if self.use_fp16:
                loss_scale = 2**self.lg_loss_scale
                (loss * loss_scale).backward()
            else:
                loss.backward()
