"""
================================================================
PINN PURA — Oscilaciones de Rabi Amortiguadas
+ Hyperparameter Sweep con Weights & Biases (wandb)
================================================================
Adaptación directa de rabi_pinn.py con soporte completo de W&B.

DIFERENCIA CLAVE con rabi_pinn_wandb.py (versión híbrida):
  Esta PINN NO usa datos de RK45 para entrenar.
  Aprende únicamente de:
    1. Ecuaciones de Bloch    ->  loss_physics
    2. Condición inicial t=0  ->  loss_ic
    3. Restricciones físicas  ->  loss_constraint

  RK45 solo aparece AL FINAL para evaluar el error.
  Nunca entra en el entrenamiento.

Función de pérdida (3 términos — sin L_data):
  L = lam_phys * L_physics
    + lam_ic   * L_ic
    + lam_con  * L_constraint

Instalación:
  pip install wandb torch scipy numpy matplotlib

Uso:
  wandb login

  python rabi_pinn_pura_wandb.py --mode sweep --count 40
  python rabi_pinn_pura_wandb.py --mode single
  python rabi_pinn_pura_wandb.py --mode best

Qué se registra en W&B por época:
  loss/total  loss/physics  loss/ic  loss/constraint
  metrics/purity_avg  metrics/purity_max  metrics/trace_error
  train/lr

Al finalizar cada run:
  • Métricas finales (MAE, error máx, pureza, traza)
  • 3 figuras idénticas a las del script original
  • Modelo guardado como artifact
================================================================
"""

import argparse

import matplotlib
matplotlib.use("Agg")           # sin ventanas — imprescindible en sweep
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
#     Solo para comparación al final. NO entra en el entrenamiento.
# ═════════════════════════════════════════════════════════════════════════════

def solve_reference(Omega=1.0, Delta=0.0, gamma=0.1, T_max=20.0, N=500):
    """
    Integra las Ecuaciones de Bloch con RK45 (igual que en rabi_pinn.py).
    Se llama DESPUÉS del entrenamiento para medir el error de la PINN.
    """
    def odes(t, y):
        ree, rgg, re, im = y
        reg = re + 1j * im
        rge = reg.conjugate()

        d_ree = -1j * (Omega / 2 * (rge - reg)) - gamma * ree
        d_rgg = -1j * (Omega / 2 * (reg - rge)) + gamma * ree
        d_reg = -1j * (Delta * reg + Omega / 2 * (rgg - ree)) - (gamma / 2) * reg

        return [d_ree.real, d_rgg.real, d_reg.real, d_reg.imag]

    t_eval = np.linspace(0, T_max, N)
    sol    = solve_ivp(odes, [0, T_max], [0., 1., 0., 0.],
                       t_eval=t_eval, method="RK45",
                       rtol=1e-9, atol=1e-11)

    ree = sol.y[0];  rgg = sol.y[1]
    re  = sol.y[2];  im  = sol.y[3]
    P   = ree**2 + rgg**2 + 2*(re**2 + im**2)

    return sol.t, ree, rgg, re, im, P


# ═════════════════════════════════════════════════════════════════════════════
# 2.  ARQUITECTURA DE LA RED NEURONAL
#     Igual que RabiPINN en rabi_pinn.py, pero ahora parametrizable.
# ═════════════════════════════════════════════════════════════════════════════

class RabiPINN(nn.Module):
    """
    Red neuronal para aproximar [rho_ee, rho_gg, Re(rho_eg), Im(rho_eg)].

    Restricciones físicas en la arquitectura:
      • rho_ee, rho_gg in (0,1)       -> sigmoid
      • Tr(rho) = 1                   -> normalización exacta
      • |rho_eg|^2 <= rho_ee*rho_gg  -> límite Cauchy-Schwarz
    """

    ACTIVATIONS = {
        "tanh"    : nn.Tanh,
        "silu"    : nn.SiLU,
        "gelu"    : nn.GELU,
        "elu"     : nn.ELU,
        "softplus": nn.Softplus,
    }

    def __init__(self, hidden_layers: int = 4, neurons: int = 64,
                 activation: str = "tanh", dropout: float = 0.0):
        super().__init__()

        act_cls = self.ACTIVATIONS.get(activation, nn.Tanh)

        # Capas ocultas: idénticas al script original,
        # pero activación y dropout son hiperparámetros
        dims   = [1] + [neurons] * hidden_layers
        layers = []
        for i in range(len(dims) - 1):
            lin = nn.Linear(dims[i], dims[i + 1])
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
            layers.append(lin)
            layers.append(act_cls())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        self.trunk = nn.Sequential(*layers)

        # Cabeza: 4 valores crudos (sin restricciones todavía)
        self.head = nn.Linear(neurons, 4)
        nn.init.xavier_normal_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        self.eps = 1e-7     # igual que en el script original

    def forward(self, t):
        raw = self.head(self.trunk(t))          # (N, 4)

        # Poblaciones: sigmoid + normalización -> Tr(rho) = 1 exacto
        ree_u  = torch.sigmoid(raw[:, 0:1])
        rgg_u  = torch.sigmoid(raw[:, 1:2])
        denom  = ree_u + rgg_u + self.eps
        rho_ee = ree_u / denom
        rho_gg = rgg_u / denom

        # Coherencias: Cauchy-Schwarz -> P <= 1  (margen 1% igual al original)
        max_coh   = torch.sqrt(rho_ee * rho_gg + self.eps) * 0.99
        rho_eg_re = torch.tanh(raw[:, 2:3]) * max_coh
        rho_eg_im = torch.tanh(raw[:, 3:4]) * max_coh

        return rho_ee, rho_gg, rho_eg_re, rho_eg_im


# ═════════════════════════════════════════════════════════════════════════════
# 3.  CLASE PRINCIPAL — PINN PURA
#     Misma lógica que RabiOscillationPINN en rabi_pinn.py,
#     ahora recibe un dict cfg y registra métricas en W&B.
# ═════════════════════════════════════════════════════════════════════════════

class RabiOscillationPINN:
    """
    PINN pura para Oscilaciones de Rabi Amortiguadas.

    Pérdida total (3 términos — SIN datos externos):
      L = lam_phys * L_physics
        + lam_ic   * L_ic
        + lam_con  * L_constraint
    """

    def __init__(self, cfg: dict):
        self.cfg   = cfg
        self.Omega = cfg["Omega"]
        self.Delta = cfg["Delta"]
        self.gamma = cfg["gamma"]
        self.eps   = 1e-7

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

        # Scheduler (incluye StepLR, el del script original)
        epochs = cfg["epochs"]
        sched  = cfg.get("scheduler", "step")

        if sched == "step":
            # Igual al original: StepLR(step_size=5000, gamma=0.9)
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size = cfg.get("step_size", 5000),
                gamma     = cfg.get("lr_decay", 0.9),
            )
        elif sched == "multistep":
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones = [int(epochs * f) for f in (0.4, 0.65, 0.85)],
                gamma      = cfg.get("lr_decay", 0.5),
            )
        elif sched == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs)
        else:
            self.scheduler = None

        self.history = []   # historial local (igual que en el original)

        n_params = sum(p.numel() for p in self.net.parameters())
        print(f"  Parametros de la red : {n_params:,}")
        print(f"  Omega={self.Omega}  Delta={self.Delta}  gamma={self.gamma}")

    # ── Derivadas automáticas (idéntico al original) ─────────────────────────

    def _derivatives(self, t):
        """Devuelve las 4 componentes y sus derivadas respecto a t."""
        t.requires_grad_(True)
        ree, rgg, reg_re, reg_im = self.net(t)

        def _d(u):
            return grad(u, t,
                        grad_outputs=torch.ones_like(u),
                        create_graph=True,
                        retain_graph=True)[0]

        return (ree, rgg, reg_re, reg_im,
                _d(ree), _d(rgg), _d(reg_re), _d(reg_im))

    # ── Pérdidas (idénticas al original) ────────────────────────────────────

    def loss_physics(self, t):
        """
        Residuos de las Ecuaciones de Bloch en componentes reales:
          drho_ee/dt = -Omega*Im(rho_eg) - gamma*rho_ee
          drho_gg/dt = +Omega*Im(rho_eg) + gamma*rho_ee
          dRe(rho_eg)/dt = +Delta*Im(rho_eg) - (gamma/2)*Re(rho_eg)
          dIm(rho_eg)/dt = -(Delta*Re(rho_eg) + (Omega/2)*(rho_gg-rho_ee)) - (gamma/2)*Im(rho_eg)
        """
        ree, rgg, reg_re, reg_im, \
        d_ree, d_rgg, d_reg_re, d_reg_im = self._derivatives(t)

        W = self.Omega;  D = self.Delta;  G = self.gamma
        X = rgg - ree

        eq_ee    = d_ree    - (-W * reg_im - G * ree)
        eq_gg    = d_rgg    - ( W * reg_im + G * ree)
        eq_eg_re = d_reg_re - ( D * reg_im - (G / 2) * reg_re)
        eq_eg_im = d_reg_im - (-(D * reg_re + (W / 2) * X) - (G / 2) * reg_im)

        return (torch.mean(eq_ee**2)    +
                torch.mean(eq_gg**2)    +
                torch.mean(eq_eg_re**2) +
                torch.mean(eq_eg_im**2))

    def loss_ic(self, t0, ic):
        """
        Condición inicial en t = 0.
        ic = [rho_ee(0), rho_gg(0), Re(rho_eg)(0), Im(rho_eg)(0)]
        """
        ree, rgg, reg_re, reg_im = self.net(t0)
        return (torch.mean((ree    - ic[0])**2) +
                torch.mean((rgg    - ic[1])**2) +
                torch.mean((reg_re - ic[2])**2) +
                torch.mean((reg_im - ic[3])**2))

    def loss_constraint(self, t):
        """
        Restricciones físicas:
          1. Tr(rho) = rho_ee + rho_gg = 1
          2. Pureza P = rho_ee^2 + rho_gg^2 + 2|rho_eg|^2 <= 1
        """
        ree, rgg, reg_re, reg_im = self.net(t)

        trace_loss  = torch.mean((ree + rgg - 1.0)**2)

        P           = ree**2 + rgg**2 + 2*(reg_re**2 + reg_im**2)
        purity_viol = torch.clamp(P - 1.0, min=0.0)
        purity_loss = torch.mean(purity_viol**2)

        return trace_loss + 10.0 * purity_loss

    # ── Entrenamiento ────────────────────────────────────────────────────────

    def train(self, ic=None, use_wandb=True):
        """
        Entrenamiento PINN pura — sin datos externos.

        Sigue la misma lógica de rabi_pinn.py más:
          • gradient clipping configurable
          • guardado del mejor modelo
          • log a W&B cada cfg['log_every'] épocas
        """
        if ic is None:
            ic = [0.0, 1.0, 0.0, 0.0]      # atomo en estado fundamental

        cfg       = self.cfg
        epochs    = cfg["epochs"]
        N_col     = cfg["n_collocation"]
        T_max     = cfg["T_max"]
        lam_phys  = cfg["lam_phys"]
        lam_ic    = cfg["lam_ic"]
        lam_con   = cfg["lam_con"]
        log_every = cfg.get("log_every", 50)
        grad_clip = cfg.get("grad_clip", 1.0)

        # Tensores fijos para la condición inicial (igual que el original)
        ic_tensor = torch.tensor(ic, dtype=torch.float32, device=DEVICE)
        t0        = torch.tensor([[0.0]], dtype=torch.float32, device=DEVICE)

        best_loss  = float("inf")
        best_state = None
        nan_count  = 0

        pbar = tqdm(range(epochs), desc="Entrenando PINN pura", ncols=90)

        for epoch in pbar:

            # Puntos de colocación aleatorios en (0, T_max]
            t_col = (torch.rand(N_col, 1, device=DEVICE) * T_max
                     ).requires_grad_(True)

            self.optimizer.zero_grad()

            # ── Las mismas 3 pérdidas del script original ─────────────────────
            L_phys = self.loss_physics(t_col)
            L_ic   = self.loss_ic(t0, ic_tensor)
            L_con  = self.loss_constraint(t_col)

            loss = lam_phys * L_phys + lam_ic * L_ic + lam_con * L_con

            # Detectar NaN (mismo criterio que en el original)
            if torch.isnan(loss):
                nan_count += 1
                if nan_count > 10:
                    print("\nDemasiados NaN. Deteniendo.")
                    break
                continue

            nan_count = 0
            loss.backward()

            # Gradient clipping (max_norm=1.0 en el original)
            torch.nn.utils.clip_grad_norm_(self.net.parameters(),
                                           max_norm=grad_clip)
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            # Guardar mejor estado
            val = loss.item()
            if val < best_loss:
                best_loss  = val
                best_state = {k: v.cpu().clone()
                              for k, v in self.net.state_dict().items()}

            # Historial local (igual que en el original)
            self.history.append({
                "total"     : val,
                "physics"   : L_phys.item(),
                "ic"        : L_ic.item(),
                "constraint": L_con.item(),
            })

            # Actualizar tqdm cada 50 épocas (igual que en el original)
            if epoch % 50 == 0:
                with torch.no_grad():
                    ree, rgg, reg_re, reg_im = self.net(t_col)
                    P_avg = torch.mean(
                        ree**2 + rgg**2 + 2*(reg_re**2 + reg_im**2))
                pbar.set_postfix({
                    "Loss": f"{val:.4f}",
                    "Phys": f"{L_phys.item():.4f}",
                    "IC"  : f"{L_ic.item():.4f}",
                    "P"   : f"{P_avg.item():.4f}",
                })

            # ── Log a W&B (nuevo respecto al original) ────────────────────────
            if use_wandb and epoch % log_every == 0:
                with torch.no_grad():
                    t_mon   = torch.rand(500, 1, device=DEVICE) * T_max
                    ree, rgg, r_re, r_im = self.net(t_mon)
                    P_all   = ree**2 + rgg**2 + 2*(r_re**2 + r_im**2)
                    P_avg_w = P_all.mean().item()
                    P_max_w = P_all.max().item()
                    trace_e = (ree + rgg - 1.0).abs().max().item()
                    lr_now  = self.optimizer.param_groups[0]["lr"]

                wandb.log({
                    "epoch"               : epoch,
                    "loss/total"          : val,
                    "loss/physics"        : L_phys.item(),
                    "loss/ic"             : L_ic.item(),
                    "loss/constraint"     : L_con.item(),
                    "metrics/purity_avg"  : P_avg_w,
                    "metrics/purity_max"  : P_max_w,
                    "metrics/trace_error" : trace_e,
                    "metrics/best_loss"   : best_loss,
                    "train/lr"            : lr_now,
                })

        # Restaurar el mejor modelo
        if best_state is not None:
            self.net.load_state_dict(
                {k: v.to(DEVICE) for k, v in best_state.items()})

        return best_loss

    # ── Predicción (idéntica al original) ───────────────────────────────────

    def predict(self, t_array: np.ndarray) -> dict:
        """Evalúa la red en un array numpy de tiempos."""
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
# 4.  MÉTRICAS FINALES
#     Se calculan SOLO al terminar, comparando contra RK45.
# ═════════════════════════════════════════════════════════════════════════════

def compute_metrics(t_ref, ref, t_eval, pred) -> dict:
    """Métricas interpolando RK45 en los puntos de evaluación."""
    ree_ref = np.interp(t_eval, t_ref, ref["ree"])
    rgg_ref = np.interp(t_eval, t_ref, ref["rgg"])
    P_ref   = np.interp(t_eval, t_ref, ref["P"])
    trace   = pred["rho_ee"] + pred["rho_gg"]
    purity  = pred["purity"]
    return {
        "final/mae_rho_ee"       : float(np.mean(np.abs(pred["rho_ee"] - ree_ref))),
        "final/max_rho_ee"       : float(np.max( np.abs(pred["rho_ee"] - ree_ref))),
        "final/mae_rho_gg"       : float(np.mean(np.abs(pred["rho_gg"] - rgg_ref))),
        "final/max_rho_gg"       : float(np.max( np.abs(pred["rho_gg"] - rgg_ref))),
        "final/mae_purity"       : float(np.mean(np.abs(purity - P_ref))),
        "final/max_trace_err"    : float(np.max( np.abs(trace - 1.0))),
        "final/purity_max"       : float(np.max(purity)),
        "final/purity_violations": int(np.sum(purity > 1.0)),
        "final/valid"            : int(np.max(purity) <= 1.0),
    }


def print_metrics(metrics: dict):
    print("\n" + "=" * 65)
    print("  MÉTRICAS FINALES")
    print("=" * 65)
    for k, v in metrics.items():
        label = k.replace("final/", "")
        val   = f"{v:.6f}" if isinstance(v, float) else str(v)
        print(f"  {label:<30} {val:>14}")
    estado = "VALIDO" if metrics["final/valid"] else "INVALIDO"
    print(f"\n  Estado fisico: {estado}")
    print("=" * 65 + "\n")


# ═════════════════════════════════════════════════════════════════════════════
# 5.  VISUALIZACIÓN
#     Idéntica a plot_all() del script original, separada en 3 funciones
#     para poder subir cada figura individualmente a W&B.
# ═════════════════════════════════════════════════════════════════════════════

def fig_comparison(t_ref, ref, t_eval, pred):
    """Fig 1 del original: PINN vs RK45 (2x2)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("PINN Pura vs RK45: Oscilaciones de Rabi Amortiguadas",
                 fontsize=14, fontweight="bold")

    kw_ref  = dict(color="black",  lw=1.5, ls="--", label="RK45 (referencia)")
    kw_pinn = dict(color="crimson", lw=2.0, label="PINN")

    axes[0, 0].plot(t_ref, ref["ree"], **kw_ref)
    axes[0, 0].plot(t_eval, pred["rho_ee"], **kw_pinn)
    axes[0, 0].set_title("Poblacion Excitada  rho_ee")
    axes[0, 0].set_ylabel("Probabilidad")
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(t_ref, ref["rgg"], **kw_ref)
    axes[0, 1].plot(t_eval, pred["rho_gg"], color="steelblue", lw=2, label="PINN")
    axes[0, 1].set_title("Poblacion Fundamental  rho_gg")
    axes[0, 1].set_ylabel("Probabilidad")
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(t_ref, ref["P"], **kw_ref)
    axes[1, 0].plot(t_eval, pred["purity"], color="darkgreen", lw=2, label="PINN")
    axes[1, 0].axhline(1.0, color="gray",   ls=":", lw=1.2, label="P = 1")
    axes[1, 0].axhline(0.5, color="orange", ls=":", lw=1.2, label="P = 0.5")
    axes[1, 0].set_title("Pureza  Tr(rho^2)")
    axes[1, 0].set_ylabel("Pureza")
    axes[1, 0].set_ylim([0, 1.1])
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    coh_ref  = np.sqrt(ref["re"]**2  + ref["im"]**2)
    coh_pinn = np.sqrt(pred["reg_re"]**2 + pred["reg_im"]**2)
    axes[1, 1].plot(t_ref, coh_ref, **kw_ref)
    axes[1, 1].plot(t_eval, coh_pinn, color="purple", lw=2, label="PINN")
    axes[1, 1].set_title("Coherencias  |rho_eg|")
    axes[1, 1].set_ylabel("|rho_eg|")
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel("Tiempo (1/Omega)")
    fig.tight_layout()
    return fig


def fig_constraints(t_ref, ref, t_eval, pred):
    """Fig 2 del original: restricciones físicas (1x3)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Verificacion de Restricciones Fisicas",
                 fontsize=13, fontweight="bold")

    trace = pred["rho_ee"] + pred["rho_gg"]
    axes[0].plot(t_eval, trace, color="navy", lw=2)
    axes[0].axhline(1.0, color="red", ls="--", lw=1.5, label="= 1 (exacto)")
    axes[0].set_title("Tr(rho) = rho_ee + rho_gg")
    axes[0].set_ylim([0.95, 1.05])
    axes[0].set_xlabel("Tiempo (1/Omega)")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(t_eval, pred["purity"], color="darkgreen", lw=2, label="P(t)")
    axes[1].axhline(1.0, color="red", ls="--", lw=1.5, label="P <= 1")
    axes[1].fill_between(t_eval, 1.0, pred["purity"],
                         where=pred["purity"] > 1.0,
                         color="red", alpha=0.4, label="violacion")
    axes[1].set_title("Pureza  Tr(rho^2) <= 1")
    axes[1].set_ylim([0, 1.15])
    axes[1].set_xlabel("Tiempo (1/Omega)")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    ree_ref = np.interp(t_eval, t_ref, ref["ree"])
    err_ee  = np.abs(pred["rho_ee"] - ree_ref)
    axes[2].semilogy(t_eval, err_ee + 1e-12, color="crimson", lw=2)
    axes[2].set_title("Error absoluto  |rho_ee PINN - rho_ee RK45|")
    axes[2].set_xlabel("Tiempo (1/Omega)")
    axes[2].set_ylabel("Error (log)")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    return fig


def fig_loss_history(history: list):
    """Fig 3 del original: evolución de las 4 pérdidas (1x4)."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("Evolucion de las Perdidas durante el Entrenamiento",
                 fontsize=13, fontweight="bold")

    keys   = ["total", "physics", "ic", "constraint"]
    titles = ["Total", "Fisica (Bloch)", "C. Inicial", "Restricciones"]
    colors = ["black", "royalblue", "darkorange", "forestgreen"]
    epochs = np.arange(len(history))

    for ax, k, ti, c in zip(axes, keys, titles, colors):
        vals = [h[k] for h in history]
        ax.semilogy(epochs, vals, color=c, lw=1.5)
        ax.set_title(ti)
        ax.set_xlabel("Epoca")
        ax.set_ylabel("Perdida (log)")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 6.  FUNCIÓN PARA EL SWEEP  (wandb.agent la llama en cada trial)
# ═════════════════════════════════════════════════════════════════════════════

def run_sweep_trial():
    """Un trial del sweep. W&B inyecta los hiperparámetros via wandb.config."""
    run = wandb.init()
    cfg = dict(wandb.config)

    print(f"\n-- Trial --  capas={cfg['hidden_layers']}  "
          f"neuronas={cfg['neurons']}  lr={cfg['lr']:.2e}  "
          f"act={cfg['activation']}  sched={cfg['scheduler']}  "
          f"lam_phys={cfg['lam_phys']}  lam_ic={cfg['lam_ic']}")

    # Entrenar PINN pura (SIN datos RK45)
    pinn      = RabiOscillationPINN(cfg=cfg)
    best_loss = pinn.train(ic=[0., 1., 0., 0.], use_wandb=True)

    # Evaluar contra RK45 al final (solo para métricas)
    t_ref, ree_r, rgg_r, re_r, im_r, P_r = solve_reference(
        Omega=cfg["Omega"], Delta=cfg["Delta"],
        gamma=cfg["gamma"], T_max=cfg["T_max"], N=500)
    ref    = {"ree": ree_r, "rgg": rgg_r, "re": re_r, "im": im_r, "P": P_r}
    t_eval = np.linspace(0, cfg["T_max"], 500)
    pred   = pinn.predict(t_eval)

    metrics = compute_metrics(t_ref, ref, t_eval, pred)
    wandb.log({**metrics, "final/best_loss": best_loss})

    for name, fig in [
        ("comparacion",   fig_comparison(t_ref, ref, t_eval, pred)),
        ("restricciones", fig_constraints(t_ref, ref, t_eval, pred)),
        ("perdidas",      fig_loss_history(pinn.history)),
    ]:
        wandb.log({f"plots/{name}": wandb.Image(fig)})
        plt.close(fig)

    print_metrics(metrics)
    run.finish()


# ═════════════════════════════════════════════════════════════════════════════
# 7.  CONFIGURACIÓN DEL SWEEP
#     3 pérdidas solamente (sin lam_data).
#     Scheduler por defecto = "step" para respetar el script original.
# ═════════════════════════════════════════════════════════════════════════════

SWEEP_CONFIG = {
    "method": "bayes",              # optimización bayesiana (más eficiente)
    "metric": {
        "name": "final/mae_rho_ee",
        "goal": "minimize",
    },
    "early_terminate": {            # cancela trials malos temprano
        "type"    : "hyperband",
        "min_iter": 2000,
        "eta"     : 2,
    },
    "parameters": {

        # ── Físicos (fijos en todo el sweep) ─────────────────────────────────
        "Omega": {"value": 1.0},
        "Delta": {"value": 0.0},
        "gamma": {"value": 0.1},
        "T_max": {"value": 20.0},

        # ── Arquitectura ──────────────────────────────────────────────────────
        "hidden_layers": {"values": [3, 4, 5, 6]},
        "neurons"      : {"values": [64, 128, 256]},
        "activation"   : {"values": ["tanh", "silu", "gelu", "elu"]},
        "dropout"      : {"values": [0.0, 0.05, 0.1]},

        # ── Optimización ──────────────────────────────────────────────────────
        "optimizer"    : {"values": ["adam", "adamw"]},
        "lr"           : {
            "distribution": "log_uniform_values",
            "min": 1e-4,
            "max": 5e-3,
        },
        "weight_decay" : {"values": [0.0, 1e-5, 1e-4]},

        # Schedulers — incluye "step" (el del script original)
        "scheduler"    : {"values": ["step", "multistep", "cosine"]},
        "lr_decay"     : {"values": [0.7, 0.8, 0.9]},
        "step_size"    : {"values": [3000, 5000, 8000]},   # solo para step

        "grad_clip"    : {"values": [0.5, 1.0, 2.0]},

        # ── Pesos de las 3 pérdidas (SIN lam_data) ────────────────────────────
        "lam_phys"     : {"values": [0.5, 1.0, 2.0, 5.0]},
        "lam_ic"       : {"values": [50.0, 100.0, 200.0, 500.0]},
        "lam_con"      : {"values": [5.0, 10.0, 20.0, 50.0]},

        # ── Entrenamiento ─────────────────────────────────────────────────────
        "epochs"       : {"value": [30000, 50000]},
        "n_collocation": {"values": [1000, 2000, 3000]},
        "log_every"    : {"value": 50},
    },
}


# ═════════════════════════════════════════════════════════════════════════════
# 8.  CONFIG POR DEFECTO
#     Refleja exactamente los valores del script original rabi_pinn.py
# ═════════════════════════════════════════════════════════════════════════════

DEFAULT_CFG = {
    # Físicos (iguales al original)
    "Omega": 1.0, "Delta": 0.0, "gamma": 0.1, "T_max": 20.0,
    # Arquitectura (iguales al original: 4 capas, 64 neuronas, tanh)
    "hidden_layers": 4, "neurons": 64,
    "activation": "tanh", "dropout": 0.0,
    # Optimización (iguales al original: Adam lr=5e-4, StepLR step=5000 gamma=0.9)
    "optimizer": "adam", "lr": 5e-4,
    "scheduler": "step", "step_size": 5000, "lr_decay": 0.9,
    "weight_decay": 0.0, "grad_clip": 1.0,
    # Pérdidas (iguales al original)
    "lam_phys": 1.0, "lam_ic": 200.0, "lam_con": 20.0,
    # Entrenamiento (iguales al original)
    "epochs": 20000, "n_collocation": 2000,
    "log_every": 50,
}


# ═════════════════════════════════════════════════════════════════════════════
# 9.  EJECUCIÓN PRINCIPAL
# ═════════════════════════════════════════════════════════════════════════════

def _run(cfg: dict, project: str, run_name: str):
    """Lógica común a los modos single y best."""
    run = wandb.init(project=project, name=run_name, config=cfg)

    pinn = RabiOscillationPINN(cfg=cfg)
    pinn.train(ic=[0., 1., 0., 0.], use_wandb=True)

    t_ref, ree_r, rgg_r, re_r, im_r, P_r = solve_reference(
        Omega=cfg["Omega"], Delta=cfg["Delta"],
        gamma=cfg["gamma"], T_max=cfg["T_max"], N=500)
    ref    = {"ree": ree_r, "rgg": rgg_r, "re": re_r, "im": im_r, "P": P_r}
    t_eval = np.linspace(0, cfg["T_max"], 500)
    pred   = pinn.predict(t_eval)

    metrics = compute_metrics(t_ref, ref, t_eval, pred)
    wandb.log(metrics)
    print_metrics(metrics)

    for name, fig in [
        ("comparacion",   fig_comparison(t_ref, ref, t_eval, pred)),
        ("restricciones", fig_constraints(t_ref, ref, t_eval, pred)),
        ("perdidas",      fig_loss_history(pinn.history)),
    ]:
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
    print(f"  Artifact W&B: {model_file}")

    run.finish()


def main():
    parser = argparse.ArgumentParser(
        description="PINN Pura Rabi + W&B Sweep (sin datos de entrenamiento)")
    parser.add_argument(
        "--mode", choices=["sweep", "single", "best"],
        default="single",
        help=(
            "sweep  = lanza hyperparameter sweep\n"
            "single = un run con la config del script original\n"
            "best   = reentrenar con la mejor config del sweep"
        ),
    )
    parser.add_argument("--project", default="rabi-pinn-pura",
                        help="Nombre del proyecto en W&B")
    parser.add_argument("--count", type=int, default=40,
                        help="Numero de trials del sweep")
    parser.add_argument("--sweep_id", default=None,
                        help="ID de sweep existente para continuar")
    args = parser.parse_args()

    print(f"Dispositivo: {DEVICE}\n")

    # ── MODO SWEEP ────────────────────────────────────────────────────────────
    if args.mode == "sweep":
        print("=" * 65)
        print(f"  SWEEP PINN PURA  |  proyecto: {args.project}")
        print(f"  trials: {args.count}  |  metodo: {SWEEP_CONFIG['method']}")
        print(f"  objetivo: {SWEEP_CONFIG['metric']['name']}")
        print(f"  NOTA: La PINN NO usa datos de RK45 para entrenar")
        print("=" * 65 + "\n")

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
        print("=" * 65)
        print(f"  RUN UNICO  |  proyecto: {args.project}")
        print(f"  Config: identica al script original rabi_pinn.py")
        print(f"  NOTA: La PINN NO usa datos de RK45 para entrenar")
        print("=" * 65 + "\n")
        _run(DEFAULT_CFG, args.project, "single_run")

    # ── MODO BEST ─────────────────────────────────────────────────────────────
    elif args.mode == "best":
        print("=" * 65)
        print(f"  MEJOR CONFIG  |  proyecto: {args.project}")
        print("=" * 65)
        print("""
  PASOS:
    1. Lanzar el sweep:
         python rabi_pinn_pura_wandb.py --mode sweep --count 40

    2. Ir a wandb.ai -> tu proyecto -> Sweeps
       Ordenar runs por 'final/mae_rho_ee' (ascendente)

    3. Copiar la config del mejor run en BEST_CFG abajo

    4. Ejecutar:
         python rabi_pinn_pura_wandb.py --mode best
        """)

        # ← EDITAR con los valores del mejor trial encontrado en el sweep
        BEST_CFG = {
            **DEFAULT_CFG,          # punto de partida: config original
            # Ejemplo (pegar los valores reales del sweep):
            # "hidden_layers": 5,
            # "neurons"      : 128,
            # "activation"   : "silu",
            # "lr"           : 3e-4,
            # "scheduler"    : "cosine",
            # "lam_ic"       : 500.0,
            # "lam_phys"     : 2.0,
            # "n_collocation": 3000,
            "epochs": 50000,        # mas epocas para el run definitivo
        }

        _run(BEST_CFG, args.project, "best_run")


if __name__ == "__main__":
    main()
