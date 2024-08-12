## B-Learner Experiment

### Introduction

In this analysis of Meta-Learner methodologies, we'll explore the recently published B-Learner paper. B-Learner postulates the degree of hidden confounding factors and derives precise boundaries alongside CATE predictions. (The level of confounding factors requires supplementation with domain expertise.)

A precise boundary represents the minimal bound that accounts for the effects of observed data and hidden confounders (domain knowledge). A less stringent bound is termed a valid bound, which encompasses additional values beyond the true causal effect. Consequently, caution is necessary when making decisions based on valid bounds.

### Background and Setup

We'll employ the Neyman-Rubin outcome framework. The unobservable distribution $P_{full}$ contains data $(X, A, Y(1), Y(0), U)$. The treatment is denoted by $A \in \{0,1\}$, covariates by $X \in \mathbb{R}^d$, $Y(0)$ and $Y(1)$ represent potential outcomes without and with treatment respectively, and unobservable confounders are expressed as $U \in \mathbb{R}^k$.

Given these premises, we can calculate CATE (Conditional Average Treatment Effect) from the difference in outcomes:

$$\tau(x) = \mathbb{E}_{P _{full}} [Y(1) - Y(0) | X = x]$$

Furthermore, in the absence of confounding factors, we can derive CATE from the difference in expected values of latent outcomes:

$$\tau(x) = \mathbb{E}_P[Y | X = x, A = 1] - \mathbb{E}_P[Y | X = x, A = 0]$$

However, some confounding exists that cannot be fully explained by the observed covariates $$X$$ alone. Therefore, due to unobserved confounding, we consider the range of deviation between $P_{full}$ and $P$ using a peripheral sensitivity model. Using the propensity scores obtained from $e(x,u) = P_{full}(A=1|X=x,U=u)$ and $e(x) = P(A=1|X=x)$, we sandwich the treatment odds ratio with a sensitivity parameter:

$$\Lambda^{-1} \leq \frac{e(x,u)}{1-e(x,u)} / \frac{e(x)}{1-e(x)} \leq \Lambda$$

When $\Lambda = 1$, we consider there to be no influence from unobserved confounding factors. As $\Lambda$ deviates from 1, the boundaries of CATE will be estimated. B-Learner aims to characterise this boundary.

The sensitivity parameter $\Lambda$ is a hyperparameter determining the degree of allowed confounding. As $\Lambda$ increases, the interval widens, potentially overturning decision-making. The appropriate value of $\Lambda$ needs to be determined.

### Properties of Boundary Estimates

Let $Q$ be a set of data containing unobserved confounds $(X, A, Y(1), Y(0), U)$, and $e^*$ be the propensity score obtained from the observed data. We can then rewrite the previous definition using $Q$:

$$\Lambda^{-1 } \leq \frac{Q(A=1|X=x,U=u)}{Q(A=0|X=x,U=u)} / \frac{e^*(x)}{1-e^*(x)} \leq \Lambda$$

Define the upper limit of outcome and CATE in the situation where $Q$ is sandwiched between $\Lambda$. In this format, the upper limit of CATE $\tau^+(x)$ will depend only on the observed data and sensitivity parameters:

$$Y^+(x,a) \equiv \sup_{Q \in M(\Lambda)} \mathbb{E}_Q[Y(a)|X=x]$$

$$\tau^+(x) \equiv \sup_{Q \in M(\Lambda)} \mathbb{E}_Q[Y(1)-Y(0)|X=x]$$

The lower limit can be obtained by converting sup to inf. The superscript + denotes the upper limit, while - indicates the lower limit.

### Valid Estimates

Estimates satisfying $\bar{\tau}^+(x) \geq \tau^+(x)$ are termed valid estimates. However, in this case, we slightly relax this condition to $\hat{\tau}^+(x) \geq \tau^+(x) - o_p(1)$.

Conversely, $\hat{\tau}^+(x) < \tau^+(x) - o_p(1)$ represents a form that does not encompass the true intervention effect, which is an undesirable outcome.

### Sharp Estimates

Estimates satisfying $\hat{\tau}^+(x) = \tau^+(x) + o_p(1)$ are called sharp estimates. Sharp estimates possess stronger properties than valid estimates.

### Identification and Estimation of Sharp Bounds

We formalise the sharp bound of CATE by the observed data distribution $P$:

First, we introduce a pseudo outcome corresponding to CVaR (Conditional Value at Risk) and unobserved outcome boundaries. $H$ represents CVaR, $R$ denotes the unobserved outcome bound, and $\rho^{\pm*}$ corresponds to the sharp bound:

$$H^{\pm}(z,q) = q(x,\alpha) + \frac{1}{1-\alpha}[y-q(x,\alpha)]^{\pm}$$

$$R^{\pm}(z,q) = \Lambda^{-1}y + (1-\Lambda^{-1})H^{\pm}(z,q)$$

$$\rho^{\pm*}(x,q) = \mathbb{E}[R^{\pm}(z,q)|X=x,A=a]$$

Even if the effect of a measure is positive overall (ATE), it may be negative in terms of individual effects (ITE). CVaR formalises this negative impact as a risk.

If $Q$ and $P$ match, the conditional potential outcome $Y$ is the conditional outcome of the observed data. Using $\mu^*(x,a) = \mathbb{E}[Y|X=x,A=a]$, we can express:

$\mathbb{E}_Q[Y(a)|X=x] = P[A=a|X=x] \times \mu^*(x,a) + P[A=1-a|X=x] \times \mathbb{E}_Q[Y(1-a)|X=x,A=a]$

Here, if we use the sharp bound $\rho^{\pm*}$, we can represent the upper and lower limits of $\mathbb{E}_Q[Y(1-a)|X=x,A=a]$ as the sharp bound:

$$Y^+(x,1) = e^*(x)\mu^*(x,1) + (1-e^*(x))\rho^{+*}(x,1)$$

$$Y^-(x,0) = (1-e^*(x))\mu^*(x,0) + e^*(x)\rho^{-*}(x,0)$$

From the above, we can now express the upper limit of CATE's sharp bound as $\tau^+(x) = Y^+(x,1) - Y^-(x,0)$. Additionally, we could express the upper limit of the sharp bound as a convex combination of the conditional outcome that can be estimated from $P$ and CVaR.

However, the counterfactual $\mathbb{E}_Q[Y(1-a)|X=x,A=a]$ must be well characterised.

We've obtained a valid/sharp bound for $\tau^+$. However, since the sensitivity parameter is a hyperparameter, the true $q$ cannot be confirmed, so it can be said that it's still insufficient for quasi-oracle estimation.

### B-Learner: Pseudo-Outcome Regression for Doubly-Robust Sharp CATE Bounds

Up to this point, we've formalised CATE's sharp bound. From here, we'll propose B-Learner by further improving the accuracy of the sharp bound.

#### Pseudo-outcome Regression for Quasi-oracle Estimation

Plug-in estimation of $e$, $\mu$, $\rho$ within $\tau^+$ from observed data introduces excessive bias. Therefore, we derive a pseudo outcome with a valid bound based on the influence function and use it as a covariate. By regressing on $X$, we estimate a sharp bound CATE with more desirable characteristics than the previous result.

The influence function is a system of functions used to learn conditional DTE (CDTE), and has a form similar to double-robust:

$\psi(Z,e,\alpha,\nu) = \kappa_1(X) - \kappa_0(X) - \frac{A-e(X)}{e(X)(1-e(X))}\alpha_A(X)^T\rho(Y,\nu_A(X))$

Given estimated nuisance $\hat{\eta} = (\hat{e}, \hat{q}^-(\cdot,0), \hat{q}^+(\cdot,1), \hat{\rho}^-(\cdot,0), \hat{\rho}^+(\cdot,1)) \in \Xi$, define a pseudo outcome corresponding to $Y^+(x,1)$, $Y^-(x,0)$, $\tau^+(x)$:

$\phi_1^+(Z,\hat{\eta}) = AY + (1-A)\hat{\rho}^+(X,1) + \frac{(1-\hat{e}(X))A}{\hat{e}(X)} \cdot (R^+(Z,\hat{q}^+(X,1)) - \hat{\rho}^+(X,1))$

$\phi_0^-(Z,\hat{\eta}) = (1-A)Y + A\hat{\rho}^-(X,0) + \frac{\hat{e}(X)(1-A)}{1-\hat{e}(X)} \cdot (R^-(Z,\hat{q}^-(X,0)) - \hat{\rho}^-(X,0))$

$\phi_{\tau}^+(Z,\hat{\eta}) = \phi_1^+(Z,\hat{\eta}) - \phi_0^-(Z,\hat{\eta})$

The third term on the right side of $\phi_1^+(Z,\hat{\eta})$ serves to orthogonalise the prediction error of $\hat{\rho}^+$. This allowed us to formulate pseudo-outcome boundaries.

Pseudo outcome can be considered a statistical estimator of observed data distribution $P$. Particularly, if $A=1$ and there is no confounding, $\phi_{\tau}^+(Z,\hat{\eta})$ has a similar shape to double-robust's pseudo outcome.

#### Algorithm

B-Learner is a two-step estimation method:

1. In the first stage, we estimate the nuisances (outcome, propensity score, CVaR) with k-fold cross-fitting and construct a pseudo-outcome estimator.
2. In the second step, we use the estimated pseudo-outcome as a covariate. Regress on $X$ and obtain the CATE bound.

The propensity score $e^*(x)$ or quantile $q^{\pm*}$ is derived using standard classifiers or regression models.

Also, for the outcome $\rho^{\pm*}(x,a) = \Lambda^{-1}\mu^*(x,a) + (1-\Lambda)^{-1}CVaR^{\pm}(x,a)$, it's possible to derive this by separately predicting $\mu^*(x,a)$ and $CVaR^{\pm}(x,a)$.

In the first stage alone, the estimation error of the sharp bound bias will be $|e^* - \hat{e}||\rho^* - \hat{\rho}| + (q^* - \hat{q})^2$. It seems that by performing up to the second stage, a robust estimation can be made.

Also, if the quantile estimates are inconsistent, $(q^+ - \hat{q}^+)^2, (q^- - \hat{q}^-)^2$ cannot be cancelled. In that case, it seems that smoothing estimation can be used.

### Experiments

In the paper, three types of verification were performed: simulation data, IHDP, and 401(k) eligibility. Here, we'll only discuss the results for 401(k) eligibility.

#### Impact of 401(k) Eligibility on Wealth Distribution

401(k) Eligibility is a dataset about 401(k) eligibility and its impact on financial assets. This dataset is known to be unconfounded, but assuming there is confounding, we verify B-Learner by changing $\Lambda$.

Left figure: For $\log \Lambda = 0.2$, the true effect is contained between the lower and upper bounds.

Right figure: When changing $\log \Lambda$, it shows the percentage where $\hat{\tau}(x) < 0$. When $\Lambda$ is small, many are positive, but if it increases to $\log \Lambda = 0.6$, about half of it becomes negative. In other words, if there were hidden confounding factors that exerted this level of influence, it's possible that about half of the negative influence would have been due to unobserved factors. With B-Learner, we were able to quantify these risks.

### 401k Verification

Let us verify whether the lower and upper bounds can be estimated using simulation data.

The simulation data generation process follows this format ($\sigma_p$ is a sigmoid function):

$$X \sim \text{Unif}([-2,2]^5)$$

$$A|X \sim \text{Bern}(\sigma(0.75X_0 + 0.5))$$

$$Y \sim \mathcal{N}((2A-1)(X_0 + 1) - 2\sin((4A-2)X_0), 1)$$

```sh
make help
make all
make run
make clean
```

### Result

The analysis of the B-Learner simulation results reveals several notable patterns.

Firstly, the average true Conditional Average Treatment Effect (CATE) remains constant at 2.0661 across all levels of `log_gamma`, indicating a consistent positive effect of the treatment. As `log_gamma` increases from 0 to 1, a widening of the estimated bounds is observed, with both the average lower and upper bounds diverging symmetrically. This expansion of the bounds reflects increasing uncertainty about the treatment effect as the model accounts for potential unobserved confounding. The coverage, representing the proportion of true CATEs falling within the estimated bounds, steadily improves from 10.65% at `log_gamma = 0` to 34.43% at `log_gamma = 1`, suggesting more conservative, wider bounds are more likely to contain the true effect.

Interestingly, the percentage of negative lower bounds increases from 60.11% to 87.16% as `log_gamma` rises, indicating a growing possibility of negative treatment effects for some individuals despite the positive average effect. The initial negative lower bounds across all `log_gamma` values suggest the method cannot rule out the possibility of negative treatment effects when accounting for potential confounding.

Finally, the gradual improvement in coverage as `log_gamma` increases demonstrates the method's ability to capture the true effect more reliably as it allows for greater confounding, albeit at the cost of wider, less precise bounds.

### Implications of 401k Eligibility

The results of the 401k simulation data reveal significant implications for understanding the impact of 401k eligibility on financial assets. The consistent positive average true CATE of 2.0661 suggests a generally beneficial effect of 401k eligibility. However, the increasing percentage of negative lower bounds, reaching 87.16% at `log_gamma = 1`, indicates substantial heterogeneity in individual treatment effects. This heterogeneity implies 401k eligibility might adversely affect some individuals' financial assets, despite the overall positive average effect. The widening bounds and improving coverage as `log_gamma` increases reflect the model's growing uncertainty when accounting for potential unobserved confounders. These findings accentuate the complexity of 401k eligibility's impact on wealth distribution and highlight the importance of considering individual variations and potential hidden confounders when evaluating retirement savings policies.

### Reference

- [B-Learner: Quasi-Oracle Bounds on Heterogeneous Causal Effects Under Hidden Confounding](https://arxiv.org/abs/2304.10577)

### License

This project is licensed under the [GNU General Public License v3.0](./LICENSE).

### Citation

```tex
@misc{ble401kv2024,
  author       = {Oketunji, A.F.},
  title        = {B-Learner Experiment},
  year         = 2024,
  version      = {0.0.1},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.13294344},
  url          = {https://doi.org/10.5281/zenodo.13294344}
}
```

### Copyright

(c) 2024 [Finbarrs Oketunji](https://finbarrs.eu). All Rights Reserved.
