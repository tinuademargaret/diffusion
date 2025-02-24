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
        use_fp16=False,
        schedule_sampler=None
    ) -> None:
        self.model = model
        self.diffussion = diffussion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch
        self.lr = lr
        self.schedule_sampler = schedule_sampler or UniformSampler(diffussion)

        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE

        self.model_params = list(self.model.parameters())

    def forward_backward(self, batch, cond):
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
