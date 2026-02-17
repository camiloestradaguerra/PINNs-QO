"""
================================================================
PINN HÍBRIDA — Oscilaciones de Rabi Amortiguadas
+ Hyperparameter Sweep con Weights & Biases (wandb)
================================================================
Instalación:
  pip install wandb torch scipy numpy matplotlib

Uso:
  # Autenticarse una sola vez
  wandb login

  # Modo 1 — Sweep automático (recomendado):
  python rabi_pinn_wandb.py --mode sweep --count 40

  # Modo 2 — Un solo run con config por defecto:
  python rabi_pinn_wandb.py --mode single

  # Modo 3 — Reentrenar con la mejor config del sweep:
  python rabi_pinn_wandb.py --mode best

Qué se registra en W&B por epoch:
  • loss/total, loss/physics, loss/ic, loss/constraint, loss/data
  • metrics/rmse_rho_ee, metrics/rmse_rho_gg
  • metrics/purity_avg, metrics/purity_max, metrics/trace_error
  • train/lr

Al finalizar cada run:
  • Métricas finales (MAE, error máx, pureza)
  • 3 figuras (comparación, errores, restricciones)
  • Modelo guardado como artifact
================================================================
"""

import argparse
import matplotlib
matplotlib.use("Agg")          # sin ventanas — compatibilidad con sweep
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from scipy.integrate import solve_ivp
from tqdm import tqdm
import wandb

# ── Reproducibilidad ──────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═════════════════════════════════════════════════════════════════════════════
# 1.  SOLUCIÓN DE REFERENCIA RK45
#     Se genera UNA sola vez; la PINN la usa como supervisión.
# ═════════════════════════════════════════════════════════════════════════════

def solve_rk45(Omega=1.0, Delta=0.0, gamma=0.1, T_max=20.0, N=1000):
    """
    Integra las Ecuaciones de Bloch con RK45 (alta precisión).
    Estado: y = [rho_ee, rho_gg, Re(rho_eg), Im(rho_eg)]
    """
    def odes(t, y):
        ree, rgg, re, im = y
        reg = re + 1j * im
        d_ree = -1j * (Omega / 2) * (reg.conjugate() - reg) - gamma * ree
        d_rgg = -1j * (Omega / 2) * (reg - reg.conjugate())  + gamma * ree
        d_reg = -1j * (Delta * reg + Omega / 2 * (rgg - ree)) - (gamma / 2) * reg
        return [d_ree.real, d_rgg.real, d_reg.real, d_reg.imag]

    t_eval = np.linspace(0, T_max, N)
    sol    = solve_ivp(odes, [0, T_max], [0., 1., 0., 0.],
                       t_eval=t_eval, method="RK45",
                       rtol=1e-10, atol=1e-12)

    ree = sol.y[0];  rgg = sol.y[1]
    re  = sol.y[2];  im  = sol.y[3]
    P   = ree**2 + rgg**2 + 2*(re**2 + im**2)

    return sol.t, ree, rgg, re, im, P


# ═════════════════════════════════════════════════════════════════════════════
# 2.  ARQUITECTURA DE LA RED NEURONAL
# ═════════════════════════════════════════════════════════════════════════════

class RabiPINN(nn.Module):
    """
    Red neuronal con restricciones físicas embebidas.

    Salida: [rho_ee, rho_gg, Re(rho_eg), Im(rho_eg)]

    Garantías:
      • Tr(rho) = 1   → normalización sigmoid exacta
      • P <= 1        → límite de Cauchy-Schwarz en coherencias
    """

    ACTIVATIONS = {
        "tanh"    : nn.Tanh,
        "silu"    : nn.SiLU,
        "gelu"    : nn.GELU,
        "elu"     : nn.ELU,
        "softplus": nn.Softplus,
    }

    def __init__(self, hidden_layers: int, neurons: int,
                 activation: str = "tanh", dropout: float = 0.0):
        super().__init__()

        act_cls = self.ACTIVATIONS.get(activation, nn.Tanh)

        dims   = [1] + [neurons] * hidden_layers
        layers = []
        for i in range(len(dims) - 1):
            lin = nn.Linear(dims[i], dims[i + 1])
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
            layers.append(lin)
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        self.trunk = nn.Sequential(*layers)

        self.head = nn.Linear(neurons, 4)
        nn.init.xavier_normal_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        self.eps = 1e-8

    def forward(self, t):
        raw = self.head(self.trunk(t))          # (N, 4)

        # Poblaciones: sigmoid + normalización → Tr(rho) = 1 exacto
        ree_u  = torch.sigmoid(raw[:, 0:1])
        rgg_u  = torch.sigmoid(raw[:, 1:2])
        denom  = ree_u + rgg_u + self.eps
        rho_ee = ree_u / denom
        rho_gg = rgg_u / denom

        # Coherencias: limite Cauchy-Schwarz → P <= 1
        max_coh   = torch.sqrt(rho_ee * rho_gg + self.eps) * 0.999
        rho_eg_re = torch.tanh(raw[:, 2:3]) * max_coh
        rho_eg_im = torch.tanh(raw[:, 3:4]) * max_coh

        return rho_ee, rho_gg, rho_eg_re, rho_eg_im


# ═════════════════════════════════════════════════════════════════════════════
# 3.  PINN HÍBRIDA
# ═════════════════════════════════════════════════════════════════════════════

class HybridRabiPINN:
    """
    PINN hibrida: fisica (Ecuaciones de Bloch) + supervision con datos RK45.

    Perdida total:
      L = lam_phys * L_Bloch
        + lam_ic   * L_IC
        + lam_con  * L_restricciones
        + lam_data * L_datos_RK45
    """

    def __init__(self, cfg: dict, t_data: np.ndarray, y_data: np.ndarray):
        self.cfg   = cfg
        self.Omega = cfg["Omega"]
        self.Delta = cfg["Delta"]
        self.gamma = cfg["gamma"]
        self.eps   = 1e-8

        # Datos RK45 como tensores
        self.t_data = torch.tensor(
            t_data.reshape(-1, 1), dtype=torch.float32, device=DEVICE)
        self.y_data = torch.tensor(
            y_data, dtype=torch.float32, device=DEVICE)

        # Red neuronal
        self.net = RabiPINN(
            hidden_layers = cfg["hidden_layers"],
            neurons       = cfg["neurons"],
            activation    = cfg["activation"],
            dropout       = cfg.get("dropout", 0.0),
        ).to(DEVICE)

        # Optimizador
        opt_cls = {"adam": optim.Adam, "adamw": optim.AdamW}.get(
            cfg.get("optimizer", "adam"), optim.Adam)
        self.optimizer = opt_cls(
            self.net.parameters(),
            lr           = cfg["lr"],
            weight_decay = cfg.get("weight_decay", 0.0),
        )

        # Scheduler
        epochs = cfg["epochs"]
        sched  = cfg.get("scheduler", "multistep")
        if sched == "multistep":
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=[int(epochs * f) for f in (0.4, 0.65, 0.85)],
                gamma=cfg.get("lr_decay", 0.5),
            )
        elif sched == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs)
        else:   # step
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=max(1, epochs // 4),
                gamma=cfg.get("lr_decay", 0.5),
            )

        self.history = []

    # ── Derivadas automáticas ────────────────────────────────────────────────

    def _forward_grad(self, t):
        t = t.clone().requires_grad_(True)
        ree, rgg, reg_re, reg_im = self.net(t)

        def _d(u):
            return grad(u, t,
                        grad_outputs=torch.ones_like(u),
                        create_graph=True, retain_graph=True)[0]

        return t, ree, rgg, reg_re, reg_im, _d(ree), _d(rgg), _d(reg_re), _d(reg_im)

    # ── Términos de perdida ──────────────────────────────────────────────────

    def loss_physics(self, t_col):
        """
        Residuos de las Ecuaciones de Bloch en componentes reales:
          drho_ee/dt = -Omega * Im(rho_eg) - gamma * rho_ee
          drho_gg/dt = +Omega * Im(rho_eg) + gamma * rho_ee
          dRe(rho_eg)/dt = +Delta * Im(rho_eg) - (gamma/2) * Re(rho_eg)
          dIm(rho_eg)/dt = -Delta * Re(rho_eg) - (Omega/2)*(rho_gg-rho_ee) - (gamma/2)*Im(rho_eg)
        """
        _, ree, rgg, reg_re, reg_im, \
        d_ree, d_rgg, d_reg_re, d_reg_im = self._forward_grad(t_col)

        W = self.Omega;  D = self.Delta;  G = self.gamma

        R_ee = d_ree    - (-W * reg_im - G * ree)
        R_gg = d_rgg    - ( W * reg_im + G * ree)
        R_re = d_reg_re - ( D * reg_im - (G / 2) * reg_re)
        R_im = d_reg_im - (-D * reg_re - (W / 2) * (rgg - ree) - (G / 2) * reg_im)

        return (torch.mean(R_ee**2) + torch.mean(R_gg**2) +
                torch.mean(R_re**2) + torch.mean(R_im**2))

    def loss_ic(self, ic: list):
        """Condición inicial: rho_gg(0)=1, resto=0."""
        t0   = torch.zeros(1, 1, dtype=torch.float32, device=DEVICE)
        ic_t = torch.tensor(ic, dtype=torch.float32, device=DEVICE)
        ree, rgg, reg_re, reg_im = self.net(t0)
        return ((ree - ic_t[0])**2 + (rgg - ic_t[1])**2 +
                (reg_re - ic_t[2])**2 + (reg_im - ic_t[3])**2).mean()

    def loss_constraints(self, t_col):
        """Tr(rho)=1  y  Tr(rho^2) <= 1."""
        ree, rgg, reg_re, reg_im = self.net(t_col)
        trace_loss = torch.mean((ree + rgg - 1.0)**2)
        P          = ree**2 + rgg**2 + 2*(reg_re**2 + reg_im**2)
        p_loss     = torch.mean(torch.clamp(P - 1.0, min=0.0)**2)
        return trace_loss + 10.0 * p_loss

    def loss_data(self):
        """MSE contra datos RK45 (mini-batch aleatorio)."""
        bs  = self.cfg.get("batch_size", 512)
        idx = torch.randint(0, self.t_data.shape[0], (bs,), device=DEVICE)
        t_b = self.t_data[idx]
        y_b = self.y_data[idx]
        ree, rgg, reg_re, reg_im = self.net(t_b)
        return (torch.mean((ree    - y_b[:, 0:1])**2) +
                torch.mean((rgg    - y_b[:, 1:2])**2) +
                torch.mean((reg_re - y_b[:, 2:3])**2) +
                torch.mean((reg_im - y_b[:, 3:4])**2))

    # ── Entrenamiento ────────────────────────────────────────────────────────

    def train(self, ic=None, use_wandb=True):
        if ic is None:
            ic = [0.0, 1.0, 0.0, 0.0]

        cfg       = self.cfg
        epochs    = cfg["epochs"]
        N_col     = cfg["n_collocation"]
        T_max     = cfg["T_max"]
        lam_phys  = cfg["lam_phys"]
        lam_ic    = cfg["lam_ic"]
        lam_con   = cfg["lam_con"]
        lam_data  = cfg["lam_data"]
        log_every = cfg.get("log_every", 50)
        grad_clip = cfg.get("grad_clip", 1.0)

        best_loss  = float("inf")
        best_state = None
        nan_streak = 0

        for epoch in tqdm(range(epochs), desc="Entrenando", ncols=90):

            t_col = torch.rand(N_col, 1, device=DEVICE) * T_max
            self.optimizer.zero_grad()

            L_phys = self.loss_physics(t_col)
            L_ic   = self.loss_ic(ic)
            L_con  = self.loss_constraints(t_col)
            L_data = self.loss_data()

            L_total = (lam_phys * L_phys + lam_ic * L_ic +
                       lam_con  * L_con  + lam_data * L_data)

            if torch.isnan(L_total) or torch.isinf(L_total):
                nan_streak += 1
                if nan_streak > 20:
                    print("\nDivergencia (NaN). Deteniendo run.")
                    break
                continue
            nan_streak = 0

            L_total.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), grad_clip)
            self.optimizer.step()
            self.scheduler.step()

            val = L_total.item()
            if val < best_loss:
                best_loss  = val
                best_state = {k: v.cpu().clone()
                              for k, v in self.net.state_dict().items()}

            self.history.append({
                "total"     : val,
                "physics"   : L_phys.item(),
                "ic"        : L_ic.item(),
                "constraint": L_con.item(),
                "data"      : L_data.item(),
            })

            # Log a W&B cada log_every epocas
            if use_wandb and epoch % log_every == 0:
                with torch.no_grad():
                    ree, rgg, r_re, r_im = self.net(self.t_data)
                    P_all   = ree**2 + rgg**2 + 2*(r_re**2 + r_im**2)
                    P_avg   = P_all.mean().item()
                    P_max   = P_all.max().item()
                    rmse_ee = (ree - self.y_data[:, 0:1]).pow(2).mean().sqrt().item()
                    rmse_gg = (rgg - self.y_data[:, 1:2]).pow(2).mean().sqrt().item()
                    trace_e = (ree + rgg - 1.0).abs().max().item()
                    lr_now  = self.optimizer.param_groups[0]["lr"]

                wandb.log({
                    "epoch"              : epoch,
                    "loss/total"         : val,
                    "loss/physics"       : L_phys.item(),
                    "loss/ic"            : L_ic.item(),
                    "loss/constraint"    : L_con.item(),
                    "loss/data"          : L_data.item(),
                    "metrics/purity_avg" : P_avg,
                    "metrics/purity_max" : P_max,
                    "metrics/rmse_rho_ee": rmse_ee,
                    "metrics/rmse_rho_gg": rmse_gg,
                    "metrics/trace_error": trace_e,
                    "metrics/best_loss"  : best_loss,
                    "train/lr"           : lr_now,
                })

        # Restaurar mejor estado
        if best_state is not None:
            self.net.load_state_dict(
                {k: v.to(DEVICE) for k, v in best_state.items()})

        return best_loss

    # ── Predicción ───────────────────────────────────────────────────────────

    def predict(self, t_array: np.ndarray) -> dict:
        self.net.eval()
        with torch.no_grad():
            t_t = torch.tensor(
                t_array.reshape(-1, 1), dtype=torch.float32, device=DEVICE)
            ree, rgg, reg_re, reg_im = self.net(t_t)
            P = ree**2 + rgg**2 + 2*(reg_re**2 + reg_im**2)
        self.net.train()
        return {
            "rho_ee": ree.cpu().numpy().flatten(),
            "rho_gg": rgg.cpu().numpy().flatten(),
            "reg_re": reg_re.cpu().numpy().flatten(),
            "reg_im": reg_im.cpu().numpy().flatten(),
            "purity": P.cpu().numpy().flatten(),
        }


# ═════════════════════════════════════════════════════════════════════════════
# 4.  MÉTRICAS
# ═════════════════════════════════════════════════════════════════════════════

def compute_metrics(t_rk, ree_rk, rgg_rk, P_rk, t_eval, pred) -> dict:
    ree_ref = np.interp(t_eval, t_rk, ree_rk)
    rgg_ref = np.interp(t_eval, t_rk, rgg_rk)
    P_ref   = np.interp(t_eval, t_rk, P_rk)
    trace   = pred["rho_ee"] + pred["rho_gg"]
    purity  = pred["purity"]
    return {
        "final/mae_rho_ee"       : float(np.mean(np.abs(pred["rho_ee"] - ree_ref))),
        "final/max_rho_ee"       : float(np.max( np.abs(pred["rho_ee"] - ree_ref))),
        "final/mae_rho_gg"       : float(np.mean(np.abs(pred["rho_gg"] - rgg_ref))),
        "final/max_rho_gg"       : float(np.max( np.abs(pred["rho_gg"] - rgg_ref))),
        "final/mae_purity"       : float(np.mean(np.abs(purity - P_ref))),
        "final/max_purity_err"   : float(np.max( np.abs(purity - P_ref))),
        "final/max_trace_err"    : float(np.max( np.abs(trace - 1.0))),
        "final/purity_max"       : float(np.max(purity)),
        "final/purity_violations": int(np.sum(purity > 1.0)),
        "final/valid"            : int(np.max(purity) <= 1.0),
    }


def print_metrics(metrics: dict):
    print("\n" + "=" * 55)
    print("  MÉTRICAS FINALES")
    print("=" * 55)
    for k, v in metrics.items():
        label = k.replace("final/", "")
        val   = f"{v:.2e}" if isinstance(v, float) else str(v)
        print(f"  {label:<30} {val:>12}")
    estado = "VALIDO" if metrics["final/valid"] else "INVALIDO"
    print(f"\n  Estado fisico: {estado}")
    print("=" * 55 + "\n")


# ═════════════════════════════════════════════════════════════════════════════
# 5.  FIGURAS
# ═════════════════════════════════════════════════════════════════════════════

def make_figures(t_rk, ree_rk, rgg_rk, re_rk, im_rk, P_rk,
                 t_eval, pred) -> dict:
    """Las mismas 3 figuras del script original."""

    kw_rk   = dict(color="black", lw=1.8, ls="--", label="RK45")
    kw_pinn = dict(lw=2.2, label="PINN híbrida")
    figs    = {}

    # Fig 1: Comparación
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("PINN Híbrida vs RK45 — Oscilaciones de Rabi Amortiguadas",
                 fontsize=13, fontweight="bold")

    axes[0, 0].plot(t_rk, ree_rk, **kw_rk)
    axes[0, 0].plot(t_eval, pred["rho_ee"], color="crimson", **kw_pinn)
    axes[0, 0].set_title("Población excitada rho_ee(t)")
    axes[0, 0].set_ylabel("Probabilidad")

    axes[0, 1].plot(t_rk, rgg_rk, **kw_rk)
    axes[0, 1].plot(t_eval, pred["rho_gg"], color="steelblue", **kw_pinn)
    axes[0, 1].set_title("Población fundamental rho_gg(t)")
    axes[0, 1].set_ylabel("Probabilidad")

    axes[1, 0].plot(t_rk, P_rk, **kw_rk)
    axes[1, 0].plot(t_eval, pred["purity"], color="darkgreen", **kw_pinn)
    axes[1, 0].axhline(1.0, color="red",    ls=":", lw=1.3, label="P=1")
    axes[1, 0].axhline(0.5, color="orange", ls=":", lw=1.3, label="P=0.5")
    axes[1, 0].set_title("Pureza Tr(rho^2)")
    axes[1, 0].set_ylim([0, 1.1])

    coh_rk   = np.sqrt(re_rk**2 + im_rk**2)
    coh_pinn = np.sqrt(pred["reg_re"]**2 + pred["reg_im"]**2)
    axes[1, 1].plot(t_rk, coh_rk, **kw_rk)
    axes[1, 1].plot(t_eval, coh_pinn, color="purple", **kw_pinn)
    axes[1, 1].set_title("Coherencias |rho_eg(t)|")

    for ax in axes.flat:
        ax.set_xlabel("Tiempo  (1/Omega)")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    figs["comparacion"] = fig

    # Fig 2: Errores
    ree_ref = np.interp(t_eval, t_rk, ree_rk)
    rgg_ref = np.interp(t_eval, t_rk, rgg_rk)
    P_ref   = np.interp(t_eval, t_rk, P_rk)

    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    fig2.suptitle("Error Absoluto: PINN Híbrida vs RK45",
                  fontsize=13, fontweight="bold")
    pairs = [
        (np.abs(pred["rho_ee"] - ree_ref), "|rho_ee PINN - rho_ee RK45|", "crimson"),
        (np.abs(pred["rho_gg"] - rgg_ref), "|rho_gg PINN - rho_gg RK45|", "steelblue"),
        (np.abs(pred["purity"] - P_ref),   "|P PINN - P RK45|",           "darkgreen"),
    ]
    for ax, (err, lab, col) in zip(axes2, pairs):
        ax.semilogy(t_eval, err + 1e-12, color=col, lw=1.8)
        ax.set_title(lab); ax.set_xlabel("Tiempo (1/Omega)")
        ax.set_ylabel("Error (log)"); ax.grid(alpha=0.3)
    fig2.tight_layout()
    figs["errores"] = fig2

    # Fig 3: Restricciones
    trace  = pred["rho_ee"] + pred["rho_gg"]
    purity = pred["purity"]

    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4))
    fig3.suptitle("Verificación de Restricciones Físicas",
                  fontsize=13, fontweight="bold")

    axes3[0].plot(t_eval, trace, color="navy", lw=2, label="Tr(rho)")
    axes3[0].axhline(1.0, color="red", ls="--", lw=1.5, label="= 1")
    axes3[0].fill_between(t_eval, trace, 1.0, alpha=0.2, color="red")
    axes3[0].set_ylim([0.98, 1.02]); axes3[0].set_title("Traza Tr(rho) = 1")

    axes3[1].plot(t_eval, purity, color="darkgreen", lw=2, label="P(t)")
    axes3[1].axhline(1.0, color="red",    ls="--", lw=1.5, label="P = 1")
    axes3[1].axhline(0.5, color="orange", ls=":",  lw=1.3, label="P = 0.5")
    axes3[1].fill_between(t_eval, 1.0, purity,
                          where=purity > 1.0,
                          color="red", alpha=0.4, label="violacion")
    axes3[1].set_ylim([0.0, 1.1]); axes3[1].set_title("Pureza Tr(rho^2) <= 1")

    for ax in axes3:
        ax.set_xlabel("Tiempo (1/Omega)")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig3.tight_layout()
    figs["restricciones"] = fig3

    return figs


# ═════════════════════════════════════════════════════════════════════════════
# 6.  FUNCIÓN PARA EL SWEEP  (wandb.agent la llama en cada trial)
# ═════════════════════════════════════════════════════════════════════════════

def run_sweep_trial():
    run = wandb.init()
    cfg = dict(wandb.config)

    print(f"\n── Trial ──  capas={cfg['hidden_layers']}  "
          f"neuronas={cfg['neurons']}  lr={cfg['lr']:.2e}  "
          f"act={cfg['activation']}  "
          f"lam_data={cfg['lam_data']}  lam_ic={cfg['lam_ic']}")

    t_rk, ree_rk, rgg_rk, re_rk, im_rk, P_rk = solve_rk45(
        Omega=cfg["Omega"], Delta=cfg["Delta"],
        gamma=cfg["gamma"], T_max=cfg["T_max"], N=1000)
    y_rk = np.stack([ree_rk, rgg_rk, re_rk, im_rk], axis=1)

    pinn      = HybridRabiPINN(cfg=cfg, t_data=t_rk, y_data=y_rk)
    best_loss = pinn.train(ic=[0., 1., 0., 0.], use_wandb=True)

    t_eval  = np.linspace(0, cfg["T_max"], 500)
    pred    = pinn.predict(t_eval)
    metrics = compute_metrics(t_rk, ree_rk, rgg_rk, P_rk, t_eval, pred)
    wandb.log({**metrics, "final/best_loss": best_loss})

    figs = make_figures(t_rk, ree_rk, rgg_rk, re_rk, im_rk, P_rk, t_eval, pred)
    for name, fig in figs.items():
        wandb.log({f"plots/{name}": wandb.Image(fig)})
        plt.close(fig)

    print_metrics(metrics)
    run.finish()


# ═════════════════════════════════════════════════════════════════════════════
# 7.  CONFIGURACIÓN DEL SWEEP
# ═════════════════════════════════════════════════════════════════════════════

SWEEP_CONFIG = {
    # bayes  = optimización bayesiana (recomendado)
    # random = muestreo aleatorio (rápido para explorar)
    # grid   = exhaustivo (solo con pocos parámetros)
    "method": "bayes",
    "metric": {
        "name": "final/mae_rho_ee",
        "goal": "minimize",
    },
    # Hyperband: cancela trials malos temprano para ahorrar tiempo
    "early_terminate": {
        "type"    : "hyperband",
        "min_iter": 2000,
        "eta"     : 2,
    },
    "parameters": {

        # Físicos (fijos)
        "Omega": {"value": 1.0},
        "Delta": {"value": 0.0},
        "gamma": {"value": 0.1},
        "T_max": {"value": 20.0},

        # Arquitectura
        "hidden_layers": {"values": [3, 4, 5, 6]},
        "neurons"      : {"values": [64, 128, 256]},
        "activation"   : {"values": ["tanh", "silu", "gelu", "elu"]},
        "dropout"      : {"values": [0.0, 0.05, 0.1]},

        # Optimización
        "optimizer"    : {"values": ["adam", "adamw"]},
        "lr"           : {
            "distribution": "log_uniform_values",
            "min": 1e-4,
            "max": 5e-3,
        },
        "weight_decay" : {"values": [0.0, 1e-5, 1e-4]},
        "scheduler"    : {"values": ["multistep", "cosine", "step"]},
        "lr_decay"     : {"values": [0.3, 0.5, 0.7]},
        "grad_clip"    : {"values": [0.5, 1.0, 2.0]},

        # Pesos de las pérdidas
        "lam_phys"     : {"values": [0.5, 1.0, 2.0, 5.0]},
        "lam_ic"       : {"values": [50.0, 100.0, 200.0, 500.0]},
        "lam_con"      : {"values": [5.0, 10.0, 20.0, 50.0]},
        "lam_data"     : {"values": [10.0, 50.0, 100.0, 200.0]},

        # Entrenamiento
        "epochs"       : {"value": 20000},
        "n_collocation": {"values": [1000, 2000, 3000]},
        "batch_size"   : {"values": [256, 512, 1024]},
        "log_every"    : {"value": 50},
    },
}


# ═════════════════════════════════════════════════════════════════════════════
# 8.  CONFIG POR DEFECTO
# ═════════════════════════════════════════════════════════════════════════════

DEFAULT_CFG = {
    "Omega": 1.0, "Delta": 0.0, "gamma": 0.1, "T_max": 20.0,
    "hidden_layers": 5, "neurons": 128,
    "activation": "tanh", "dropout": 0.0,
    "optimizer": "adam", "lr": 5e-4, "lr_decay": 0.5,
    "scheduler": "multistep", "weight_decay": 0.0, "grad_clip": 1.0,
    "lam_phys": 1.0, "lam_ic": 200.0, "lam_con": 20.0, "lam_data": 50.0,
    "epochs": 20000, "n_collocation": 2000,
    "batch_size": 512, "log_every": 50,
}


# ═════════════════════════════════════════════════════════════════════════════
# 9.  EJECUCIÓN PRINCIPAL
# ═════════════════════════════════════════════════════════════════════════════

def _run_single_or_best(cfg, project, run_name):
    """Lógica compartida entre modo single y modo best."""
    run = wandb.init(project=project, name=run_name, config=cfg)

    t_rk, ree_rk, rgg_rk, re_rk, im_rk, P_rk = solve_rk45(
        Omega=cfg["Omega"], Delta=cfg["Delta"],
        gamma=cfg["gamma"], T_max=cfg["T_max"], N=1000)
    y_rk = np.stack([ree_rk, rgg_rk, re_rk, im_rk], axis=1)

    pinn = HybridRabiPINN(cfg=cfg, t_data=t_rk, y_data=y_rk)
    pinn.train(ic=[0., 1., 0., 0.], use_wandb=True)

    t_eval  = np.linspace(0, cfg["T_max"], 500)
    pred    = pinn.predict(t_eval)
    metrics = compute_metrics(t_rk, ree_rk, rgg_rk, P_rk, t_eval, pred)
    wandb.log(metrics)
    print_metrics(metrics)

    figs = make_figures(t_rk, ree_rk, rgg_rk, re_rk, im_rk, P_rk, t_eval, pred)
    for name, fig in figs.items():
        path = f"{run_name}_{name}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        wandb.log({f"plots/{name}": wandb.Image(fig)})
        plt.close(fig)
        print(f"  Guardado: {path}")

    model_file = f"{run_name}_model.pth"
    torch.save(pinn.net.state_dict(), model_file)
    art = wandb.Artifact(run_name, type="model")
    art.add_file(model_file)
    run.log_artifact(art)
    print(f"  Artifact: {model_file}")

    run.finish()


def main():
    parser = argparse.ArgumentParser(
        description="PINN Híbrida Rabi + W&B Sweep")
    parser.add_argument(
        "--mode", choices=["sweep", "single", "best"],
        default="single",
        help=(
            "sweep  = lanza hyperparameter sweep\n"
            "single = un run con la config por defecto\n"
            "best   = reentrenar editando BEST_CFG en el codigo"
        ),
    )
    parser.add_argument("--project", default="rabi-pinn",
                        help="Nombre del proyecto en W&B")
    parser.add_argument("--count", type=int, default=40,
                        help="Trials del sweep (solo en modo sweep)")
    parser.add_argument("--sweep_id", default=None,
                        help="ID de sweep existente para continuar")
    args = parser.parse_args()

    print(f"Dispositivo: {DEVICE}\n")

    # ── MODO SWEEP ────────────────────────────────────────────────────────────
    if args.mode == "sweep":
        print("=" * 60)
        print(f"  SWEEP  |  proyecto: {args.project}  |  trials: {args.count}")
        print(f"  método: {SWEEP_CONFIG['method']}   "
              f"objetivo: {SWEEP_CONFIG['metric']['name']}")
        print("=" * 60 + "\n")

        if args.sweep_id:
            sweep_id = args.sweep_id
            print(f"  Continuando sweep: {sweep_id}\n")
        else:
            sweep_id = wandb.sweep(SWEEP_CONFIG, project=args.project)
            print(f"  Sweep creado: {sweep_id}\n")

        wandb.agent(sweep_id,
                    function=run_sweep_trial,
                    count=args.count,
                    project=args.project)

        print(f"\nSweep finalizado.")
        print(f"Ver en: https://wandb.ai/<usuario>/{args.project}/sweeps/{sweep_id}")

    # ── MODO SINGLE ───────────────────────────────────────────────────────────
    elif args.mode == "single":
        print("=" * 60)
        print(f"  RUN ÚNICO  |  proyecto: {args.project}")
        print("=" * 60 + "\n")
        _run_single_or_best(DEFAULT_CFG, args.project, "single_run")

    # ── MODO BEST ─────────────────────────────────────────────────────────────
    elif args.mode == "best":
        print("=" * 60)
        print(f"  MEJOR CONFIG  |  proyecto: {args.project}")
        print("=" * 60)
        print("""
  PASOS para usar este modo:
    1. Ejecutar el sweep:  python rabi_pinn_wandb.py --mode sweep
    2. Ir a wandb.ai -> tu proyecto -> Sweeps
    3. Ordenar runs por 'final/mae_rho_ee' (ascendente)
    4. Copiar la config del mejor run en BEST_CFG (abajo en el codigo)
    5. Ejecutar:  python rabi_pinn_wandb.py --mode best
        """)

        # ← EDITAR con los valores óptimos encontrados en el sweep
        BEST_CFG = {
            **DEFAULT_CFG,          # punto de partida
            # Ejemplo: descomentar y ajustar con los valores del sweep
            "hidden_layers": 5,
            "neurons"      : 128,
            "activation"   : "tanh",
            "lr"           : 0.0005, # 8e-4,
            "lam_data"     : 50.0,
            "lam_ic"       : 200.0,
            "epochs"       : 100000,   # más épocas para el run definitivo
        }

        _run_single_or_best(BEST_CFG, args.project, "best_run")


if __name__ == "__main__":
    main()
