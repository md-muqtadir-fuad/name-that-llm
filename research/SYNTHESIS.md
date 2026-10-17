# Research synthesis: Name That LLM (as of 2026-10-17)

## 1. What we know so far (measured, not assumed)

| # | Finding | Evidence |
|---|---|---|
| F1 | Removing a passage's prompt (topic) component is worth about **+0.05** balanced accuracy, **but only with the true groups**. Test groups recovered with TF-IDF give about 0. | paper_tests: twin, true groups +0.053 vs recovered −0.005 at 1–3 passages/prompt |
| F2 | Any rule that enforces "one model per group" fails on the real test set | Hungarian assignment (leaderboard drop), push-apart within groups 0.544 → 0.536 |
| F3 | The test set probably has 1–3 passages per prompt, so **most passages are singletons** | leaderboard behaviour of the group methods; qasminumber's analysis |
| F4 | Frozen LLM layers 7+22 of Qwen3-8B are the best single readout; test-set standardisation adds +0.011 | paper_tests |
| F5 | Telling labs apart is easy (0.64–0.69); telling sibling models apart is hard (about half of all errors are within a lab) | lexical analysis, confusion matrices |
| F6 | Specialists, NAP projection, prototype heads, pseudo-labelling and an after-the-fact lab term all give ≈0 or negative on the blend | experiments/lexical, experiments/papers |
| F7 | Surface style (punctuation, word length) is lab-level and is already captured by the LLM features. The 4-view TF-IDF SVC is still the best text-only member (+0.029 on top of the LLM features). | experiments/style |

## 2. A working model of the data

Write the pooled feature vector of passage $i$ (model $y_i$, prompt $p_i$, length $\ell_i$) as

$$x_i = \underbrace{s_{y_i}}_{\text{model style}} + \underbrace{t_{p_i}}_{\text{prompt/topic}} + \underbrace{\beta\,\ell_i}_{\text{length}} + \underbrace{u_{y_i,p_i}}_{\text{model}\times\text{prompt}} + \varepsilon_i .$$

- A linear head trained on $x$ has to separate the $s_y$ terms despite large nuisance terms $t_p$, whose variance is much bigger than the gaps between sibling models' $s_y$. That's why centring by the true prompt helps so much (F1).
- At test time $p_i$ is unknown and groups are tiny (F3). The useful quantity is therefore an estimate $\hat t(x_i)$ **from the passage alone**.

## 3. New methods (ranked by expected value per unit cost)

### M1. Predicted-topic residualisation (group-free centring), new
Train a map from content to the prompt mean on the training prompts. The target is the average over the ~18 models, so it carries no style:
$$\hat f=\arg\min_f\sum_i\big\lVert \bar m_{p_i}-\mu-f(c_i)\big\rVert^2+\alpha\lVert f\rVert^2_{\mathcal H},\qquad \tilde x_i=x_i-\lambda\hat f(c_i).$$
- $c_i$ is the content-word TF-IDF (stop words removed), and the kernel is cosine, so $\hat f(c)=k(c)^\top (K+\alpha I)^{-1}(M-\mu)$.
- It is cross-fitted, so training rows also get out-of-prompt predictions, which keeps train and test distributions matched.
- It works on singletons, unlike cluster centring.
- It differs from NAP: NAP removes a fixed subspace for every passage, which also removes style. M1 subtracts an offset that is specific to each passage's predicted topic.

**Test:** `experiments/papers/residualize.py` (results in §4).

### M2. Length residualisation (within-model slope), new
The host draws target lengths at random, so length is a nuisance. Estimate the slope **within** each model so the model means are kept:
$$\beta=\frac{\sum_i(\ell_i-\bar\ell_{y_i})(x_i-\bar x_{y_i})}{\sum_i(\ell_i-\bar\ell_{y_i})^2},\qquad \tilde x=x-\beta(\ell-\bar\ell).$$

### M3. Soft kernel centring (replaces hard clusters)
Estimate each passage's topic mean from its neighbours, with weights that decay with content similarity, then shrink by the effective number of neighbours:
$$\hat t_i=\frac{\sum_{j\ne i}w_{ij}x_j}{\sum_{j\ne i}w_{ij}},\quad w_{ij}=e^{s_{ij}/\tau}\mathbb 1[s_{ij}>s_0],\quad n^{\text{eff}}_i=\frac{(\sum_j w_{ij})^2}{\sum_j w_{ij}^2},\quad \tilde x_i=x_i-\frac{n^{\text{eff}}_i}{n^{\text{eff}}_i+\kappa}\hat t_i .$$
- Singletons, with no neighbours, are left untouched automatically.
- There is no hard threshold, which makes it robust to F2-style failures. Train the twin head with the same soft centring on simulated pools.
- M1 and M3 combine as $\tilde x_i = x_i - \hat f(c_i) - \lambda_i(\hat t_i-\hat f)$, i.e. the prior from M1 corrected by evidence from neighbours (a partial-pooling / empirical-Bayes estimate of $t_{p_i}$).

### M4. Hierarchical and contrastive training objective (for LoRA)
$$\mathcal L=\mathrm{CE}(W_m z,y)+\gamma\,\mathrm{CE}(W_\ell z,\mathrm{lab}(y))+\eta\,\mathcal L_{\text{SupCon}},\quad
\mathcal L_{\text{SupCon}}=-\sum_i\frac{1}{|P_i|}\sum_{p\in P_i}\log\frac{e^{z_i\cdot z_p/\tau}}{\sum_{a\ne i}e^{z_i\cdot z_a/\tau}} .$$
- $P_i$ contains other passages by the same model.
- The batch is built from same-prompt groups, so the negatives are the hardest content-matched ones (Wegmann 2022), which pushes $z$ to ignore $t_p$.
- The Gemma run already uses $\gamma=0.5$. SupCon is test #9 in the plan.

### M5. Sibling-aware factorised decoding
$$\log p(y\mid x)=\log p(\mathrm{lab}\mid x)+\log p(y\mid \mathrm{lab},x).$$
- The second factor uses heads trained only within the lab, on residualised features (M1/M2), where the sibling differences are the signal.
- Post-hoc stylometric specialists failed (F6), but **LLM-feature** within-lab heads on topic-removed features have not been tried.

### M6. Product-of-experts blend with per-member temperature, stacking as an alternative
$$\log p(y\mid x)\propto\sum_k w_k\,\log\mathrm{softmax}(T_k\,\ell_k(x))_y+b_y,$$
- Fit $w,T,b$ on nested held-out predictions. The bias $b$ is constrained, or shrunk to 0, because the public and private class mixes differ.
- Stacking with a small meta-classifier over all members' logits is the alternative (plan test #15).

### M7. Distribution-matching transduction, not pseudo-labels
- Pseudo-labels reinforce errors at ~45% accuracy (F6).
- Instead, match only the first and second moments of the test features to the train ones: test standardisation, which already helps, or CORAL, $\tilde X_{te}=X_{te}\,C_{te}^{-1/2}C_{tr}^{1/2}$, with shrunk covariances.
- These use no labels, so they can't reinforce errors.

## 4. Trial results (filled in as tests finish)
All results are on the frozen Qwen3-8B L7+22 head, 5-fold GroupKFold by prompt, base 0.451.

| Method | Δ vs base | Folds positive | Verdict |
|---|---|---|---|
| **Lab-gated mixture of experts, M5** (idea: **Md Muqtadir Fuad**): $\log p(\mathrm{lab}\mid x)+\log p(y\mid\mathrm{lab},x)$, experts standardised within the lab, gate weight 1, C=0.0025 | **+0.0118** | **4/5** | ✅ first new method past the bar; on-blend check pending |
| M2 length residualisation | +0.0015 | 4/5 | real but below the bar |
| M1 predicted-topic residualisation (TF-IDF or LLM kernel, all settings) | −0.0015 to −0.0055 | 1–2/5 | ❌ topic map doesn't transfer from 150 prompts; retry with the 1,902 external prompts in month 3 |
| Residual MoE (flat + experts) | −0.003 to −0.020 | 0–1/5 | ❌ |

## 5. Decision rules
- Ship only methods with ≥ +0.005 on held-out prompts, positive in ≥ 4 of 5 folds, and robust under simulated 1–3 passages/prompt with unequal class mixes.
- Never add hard per-group constraints (F2).
- Any feature transform (M1–M3, M7) must be applied identically to train (cross-fitted) and test, and to every frozen member and the LoRA features.
