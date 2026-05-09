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


# Custom loss function
class xent_loss(LossFnBase):
    def __call__(
        self, logits: torch.Tensor, labels: torch.Tensor, step_frac: float
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
        loss = torch.nn.functional.cross_entropy(logits, labels)
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
        self.alpha = alpha
        self.beta = beta
        self.warmup_frac = warmup_frac

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        step_frac: float,
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
        self.aux_coef = aux_coef
        self.warmup_frac = warmup_frac

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        step_frac: float,
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


class reverse_kl_loss(LossFnBase):
    """
    Reverse KL: KL(p_strong ‖ p_weak) = Σ p_strong * log(p_strong / p_weak)
    
    기존 xent_loss(Forward CE)의 대체제.
    strong model의 확률이 weight가 되므로 weak label의 noisy한 영역을 무시하는 효과.
    """

    def __call__(
        self, logits: torch.Tensor, labels: torch.Tensor, step_frac: float
    ) -> torch.Tensor:
        logits = logits.float()
        labels = labels.float()  # weak soft labels, shape: [B, C]

        log_p_strong = torch.log_softmax(logits, dim=-1)   # [B, C]
        p_strong = log_p_strong.exp()                       # [B, C]
        log_p_weak = torch.log(labels.clamp(min=1e-8))     # [B, C]

        # KL(p_strong ‖ p_weak) = Σ p_strong * (log p_strong - log p_weak)
        loss = (p_strong * (log_p_strong - log_p_weak)).sum(dim=-1)
        return loss.mean()


class reverse_ce_loss(LossFnBase):
    """
    Reverse CE: -Σ p_strong * log(p_weak)
    
    Reverse KL에서 entropy term H(p_strong)을 뺀 것.
    (Reverse KL = Reverse CE - H(p_strong))
    논문에서 RKL과 비슷한 성능이지만 구현이 조금 더 단순.
    """

    def __call__(
        self, logits: torch.Tensor, labels: torch.Tensor, step_frac: float
    ) -> torch.Tensor:
        logits = logits.float()
        labels = labels.float()  # weak soft labels, shape: [B, C]

        p_strong = torch.softmax(logits, dim=-1)        # [B, C]
        log_p_weak = torch.log(labels.clamp(min=1e-8))  # [B, C]

        # Reverse CE = -Σ p_strong * log(p_weak)
        loss = -(p_strong * log_p_weak).sum(dim=-1)
        return loss.mean()


class reverse_logconf_loss_fn(LossFnBase):
    """
    Reverse CE + logconf regularization (논문 Figure 3의 'Reve. Conf. CE')
    
    기존 logconf_loss_fn에서 CE → Reverse CE로 교체한 버전.
    Burns et al.의 confidence regularization을 유지하면서
    reverse loss의 이점을 함께 누릴 수 있음.
    """

    def __init__(self, aux_coef: float = 0.5, warmup_frac: float = 0.1):
        self.aux_coef = aux_coef
        self.warmup_frac = warmup_frac

    def __call__(
        self, logits: torch.Tensor, labels: torch.Tensor, step_frac: float
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
        # logconf와 동일하게 target 구성 (weak + strong 혼합)
        target = labels * (1 - coef) + strong_preds.detach() * coef

        # CE → Reverse CE 로 교체: weight가 p_weak → p_strong
        log_target = torch.log(target.clamp(min=1e-8))
        loss = -(preds * log_target).sum(dim=-1)
        return loss.mean()
    


#New confidence induction loss prototype
class conf_induc_loss_proto(LossFnBase):
    '''
    L(x) = CE( f(x), (1-g(x)) * f_w(x) + g(x) * f_t(x) )

    [Question]
    Should we use each distribution as a hard-label form?

    '''
    def __init__(
        self,
        warmup_frac: float = 0.1,
    ):
        self.warmup_frac = warmup_frac

    def __call__(
        self,
        logits: torch.Tensor, # train되는 모델의 logit
        labels: torch.Tensor, # ground truth or weak label (soft label)
        step_frac: float
    ):
        logits = logits.float()
        labels = labels.float()
        
        if step_frac < self.warmup_frac:
            return torch.nn.functional.cross_entropy(logits, labels, reduction='none').mean()


        # single change point detection
        conf = labels.max(dim=-1).values
        conf_np = conf.detach().cpu().numpy()
        sorted_scores = np.sort(conf_np)

        try:
            algo = rpt.Binseg(model="l2").fit(sorted_scores)
            breakpoint_idx = algo.predict(n_bkps=1)[0]
            threshold = float(sorted_scores[breakpoint_idx])
            # single change point detection method 사용 (논문의 방식과 동일)
            # threshold를 기준으로 weak model의 confidence가 낮으면 hard 높으면 easy or overlap이다.
        except Exception:
            threshold = float(np.median(conf_np))


        coef = (conf >= threshold).float().unsqueeze(-1)
        # binary classification task이기 때문에 soft label의 형태는 [#sample, 2]이다. 

        strong_preds = torch.softmax(logits, dim=-1).detach()
        # hard-label ver.
        #
        #

        target = coef * labels + (1.0 - coef) * strong_preds
        loss = torch.nn.functional.cross_entropy(logits, target, reduction = 'none')
        return loss.mean()


# Final version
class conf_induc_loss(LossFnBase):
    def __init__(
        self,
        warmup_frac: float = 0.1,
        update_every: int = 50,
        ema_alpha: float = 0.9,
    ):
        self.warmup_frac = warmup_frac
        self.update_every = update_every
        self.ema_alpha = ema_alpha
        self.threshold = None     # EMA-smoothed threshold (실제 사용)
        self._step_count = 0
        self._conf_buffer = []

    def __call__(self, logits, labels, step_frac):
        logits = logits.float()
        labels = labels.float()

        if step_frac < self.warmup_frac:
            return torch.nn.functional.cross_entropy(
                logits, labels, reduction='none'
            ).mean()

        conf = labels.max(dim=-1).values

        # ── 매 step confidence buffer에 누적 ──
        self._conf_buffer.append(conf.detach().cpu().numpy())
        self._step_count += 1

        # ── update 주기마다: 누적 데이터로 binseg → EMA로 smoothing ──
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
                    self.ema_alpha * self.threshold
                    + (1 - self.ema_alpha) * threshold_now
                )

            self._conf_buffer = []   # buffer 비우기

        # 첫 update 전이면 weak label만 사용
        if self.threshold is None:
            return torch.nn.functional.cross_entropy(
                logits, labels, reduction='none'
            ).mean()

        coef = (conf >= self.threshold).float().unsqueeze(-1)
        strong_preds = torch.softmax(logits, dim=-1).detach()
        
        target = coef * labels + (1.0 - coef) * strong_preds
        loss = torch.nn.functional.cross_entropy(
            logits, target, reduction='none'
        )
        
        return loss.mean()
