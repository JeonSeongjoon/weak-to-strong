import torch
import numpy as np
import ruptures as rpt


class LossFnBase:
    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        This function calculates the loss between logits and labels.
        """
        raise NotImplementedError


# Baseline
class xent_loss(LossFnBase):       
    def __init__(self):
        self.name = "xent"
        
    def __call__(self, 
        logits: torch.Tensor, 
        labels: torch.Tensor, 
        step_frac: float,
        diff: torch.Tensor = None
    ) -> torch.Tensor:
        """
        This function calculates the cross entropy loss between logits and labels.

        Parameters:
        logits: The predicted values.
        labels: The actual values.
        step_frac: The fraction of total training steps completed.

        Returns:
        The mean of the cross entropy loss.
        """
        logits = logits.float()
        labels = labels.float()

        loss = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
        return loss.mean()
    


class product_loss_fn(LossFnBase):
    """
    This class defines a custom loss function for product of predictions and labels.

    Attributes:
    alpha: A float indicating how much to weigh the weak model.
    beta: A float indicating how much to weigh the strong model.
    warmup_frac: A float indicating the fraction of total training steps for warmup.
    """

    def __init__(
        self,
        alpha: float = 1.0,  # how much to weigh the weak model
        beta: float = 1.0,  # how much to weigh the strong model
        warmup_frac: float = 0.1,  # in terms of fraction of total training steps
    ):
        self.name = "product"
        self.alpha = alpha
        self.beta = beta
        self.warmup_frac = warmup_frac

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        step_frac: float,
        diff: torch.Tensor = None
    ) -> torch.Tensor:
        preds = torch.softmax(logits, dim=-1)
        target = torch.pow(preds, self.beta) * torch.pow(labels, self.alpha)
        target /= target.sum(dim=-1, keepdim=True)
        target = target.detach()
        loss = torch.nn.functional.cross_entropy(logits, target, reduction="none")
        return loss.mean()


class logconf_loss_fn(LossFnBase):
    """
    This class defines a custom loss function for log confidence.

    Attributes:
    aux_coef: A float indicating the auxiliary coefficient.
    warmup_frac: A float indicating the fraction of total training steps for warmup.
    """

    def __init__(
        self,
        aux_coef: float = 0.5,
        warmup_frac: float = 0.1,  # in terms of fraction of total training steps
    ):
        self.name = "logconf"
        self.aux_coef = aux_coef
        self.warmup_frac = warmup_frac

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        step_frac: float,
        diff: torch.Tensor = None
    ) -> torch.Tensor:
        logits = logits.float()
        labels = labels.float()
        coef = 1.0 if step_frac > self.warmup_frac else step_frac
        coef = coef * self.aux_coef
        preds = torch.softmax(logits, dim=-1)
        mean_weak = torch.mean(labels, dim=0)
        assert mean_weak.shape == (2,)
        threshold = torch.quantile(preds[:, 0], mean_weak[1])
        strong_preds = torch.cat(
            [(preds[:, 0] >= threshold)[:, None], (preds[:, 0] < threshold)[:, None]],
            dim=1,
        )
        target = labels * (1 - coef) + strong_preds.detach() * coef

        loss = torch.nn.functional.cross_entropy(logits, target, reduction="none")
        return loss.mean()


class conf_induc_loss(LossFnBase):
    def __init__(
        self,
        warmup_frac: float = 0.2,
        update_every: int = 50,
        ema_alpha: float = 0.9,
    ):
        self.name = "conf_induc"
        self.warmup_frac = warmup_frac
        self.update_every = update_every
        self.ema_alpha = ema_alpha
        self.threshold = None     # EMA-smoothed threshold (실제 사용)
        self._step_count = 0
        self._conf_buffer = []
        self.easy = None
        self.conf = None

    def __call__(self, 
            logits: torch.Tensor, 
            labels: torch.Tensor, 
            step_frac,
            diff: torch.Tensor = None
        ):

        logits = logits.float()
        labels = labels.float()
        conf = labels.max(dim=-1).values

        if step_frac < self.warmup_frac:
            self.easy = [None] * len(labels)
            self.conf = conf
            return torch.nn.functional.cross_entropy(logits, labels, reduction='none').mean()

        # 매 step confidence buffer에 누적
        self._conf_buffer.append(conf.detach().cpu().numpy())
        self._step_count += 1

        # update 주기마다: 누적 데이터로 binseg → EMA로 smoothing 
        if self._step_count % self.update_every == 0:
            all_conf = np.concatenate(self._conf_buffer)
            sorted_scores = np.sort(all_conf)
            try:
                algo = rpt.Binseg(model="l2").fit(sorted_scores)
                breakpoint_idx = algo.predict(n_bkps=1)[0]
                threshold_now = float(sorted_scores[breakpoint_idx])
            except Exception:
                threshold_now = float(np.median(all_conf))

            # EMA 갱신
            if self.threshold is None:
                self.threshold = threshold_now      # 첫 갱신은 그대로
            else:
                self.threshold = (
                    self.ema_alpha * self.threshold + (1 - self.ema_alpha) * threshold_now
                )

            self._conf_buffer = []   # buffer 비우기

        # 첫 update 전이면 weak label만 사용
        if self.threshold is None:
            return torch.nn.functional.cross_entropy(logits, labels, reduction='none').mean()

        coef = (conf < self.threshold).float().unsqueeze(-1)
        strong_preds = torch.nn.functional.one_hot(
                    logits.argmax(dim=-1), num_classes=logits.size(-1)
        ).float().detach()
        
        target =  (1.0 - coef) * labels + coef * strong_preds
        loss = torch.nn.functional.cross_entropy(logits, target, reduction='none')

        self.easy = coef
        self.conf = conf
        
        return loss.mean()


class conf_induc_filt_loss(LossFnBase):
    def __init__(
        self,
        warmup_frac: float = 0.1,
        update_every: int = 50,
        ema_alpha: float = 0.9,
    ):
        self.name = "conf_induc_filt"
        self.warmup_frac = warmup_frac
        self.update_every = update_every
        self.ema_alpha = ema_alpha
        self.threshold = None     # EMA-smoothed threshold (실제 사용)
        self._step_count = 0
        self._conf_buffer = []
        self.easy = None
        self.conf = None

    def __call__(self, 
            logits: torch.Tensor, 
            labels: torch.Tensor, 
            step_frac,
            diff: torch.Tensor = None
        ):

        logits = logits.float()
        labels = labels.float()
        conf = labels.max(dim=-1).values

        if step_frac < self.warmup_frac:
            self.easy = [None] * len(labels)
            self.conf = conf
            return torch.nn.functional.cross_entropy(logits, labels, reduction='none').mean()

        # 매 step confidence buffer에 누적
        self._conf_buffer.append(conf.detach().cpu().numpy())
        self._step_count += 1

        # update 주기마다: 누적 데이터로 binseg → EMA로 smoothing 
        if self._step_count % self.update_every == 0:
            all_conf = np.concatenate(self._conf_buffer)
            sorted_scores = np.sort(all_conf)
            try:
                algo = rpt.Binseg(model="l2").fit(sorted_scores)
                breakpoint_idx = algo.predict(n_bkps=1)[0]
                threshold_now = float(sorted_scores[breakpoint_idx])
            except Exception:
                threshold_now = float(np.median(all_conf))

            # EMA 갱신
            if self.threshold is None:
                self.threshold = threshold_now      # 첫 갱신은 그대로
            else:
                self.threshold = (
                    self.ema_alpha * self.threshold + (1 - self.ema_alpha) * threshold_now
                )

            self._conf_buffer = []   # buffer 비우기

        # 첫 update 전이면 weak label만 사용
        if self.threshold is None:
            return torch.nn.functional.cross_entropy(logits, labels, reduction='none').mean()

        keep = (conf < self.threshold).float()    # easy=0, hard=1
        self.easy = keep.unsqueeze(-1)
        self.conf = conf

        mask = keep == 0                          # easy 샘플만 선택

        if mask.sum() == 0:
            return logits.sum() * 0.0

        loss = torch.nn.functional.cross_entropy(
            logits[mask], labels[mask], reduction='none'
        )

        return loss.mean()



class conf_induc_anc_loss(LossFnBase):

    def __init__(
        self,
        warmup_frac: float = 0.1,
    ):
        self.name = "conf_induc_anc"
        self.warmup_frac = warmup_frac  

    def __call__(self, 
            logits: torch.Tensor, 
            labels: torch.Tensor, 
            step_frac,
            diff: torch.Tensor = None,
        ):
        if diff is None:
            raise ValueError("diff should be not None")

        logits = logits.float()
        labels = labels.float()

        if step_frac < self.warmup_frac:
            return torch.nn.functional.cross_entropy(logits, labels, reduction='none').mean()

        coef = diff.float().unsqueeze(-1)

        # coef error protection
        if torch.any((coef < 0) | (coef > 1)):
            raise ValueError(f"diff must be in [0, 1], got range [{coef.min().item()}, {coef.max().item()}]")
        
        strong_preds = torch.nn.functional.one_hot(
                    logits.argmax(dim=-1), num_classes=logits.size(-1)
        ).float().detach()
        target =  (1.0 - coef) * labels + coef * strong_preds
        loss = torch.nn.functional.cross_entropy(logits, target, reduction='none')

        return loss.mean()


class conf_induc_anc_filt_loss(LossFnBase):

    def __init__(self, warmup_frac: float = 0.1):
        self.name = "conf_induc_anc_filt"
        self.warmup_frac = warmup_frac

    def __call__(self,
            logits: torch.Tensor,
            labels: torch.Tensor,
            step_frac,
            diff: torch.Tensor = None,
        ):
        if diff is None:
            raise ValueError("diff should be not None")

        logits = logits.float()
        labels = labels.float()

        if step_frac < self.warmup_frac:
            return torch.nn.functional.cross_entropy(logits, labels, reduction='none').mean()

        coef = diff.float()          

        # coef error protection
        if torch.any((coef < 0) | (coef > 1)):
            raise ValueError(f"diff must be in [0, 1], got range [{coef.min().item()}, {coef.max().item()}]")

        w = 1.0 - coef               

        if w.sum() == 0:             
            return logits.sum() * 0.0

        loss = torch.nn.functional.cross_entropy(logits, labels, reduction='none')

        return (w * loss).sum() / w.sum()