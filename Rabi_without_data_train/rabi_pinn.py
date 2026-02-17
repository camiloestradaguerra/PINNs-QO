"""
================================================
PINN para Oscilaciones de Rabi Amortiguadas
================================================
Resuelve las Ecuaciones de Bloch Ópticas usando
Physics-Informed Neural Networks (PINN) con PyTorch.

Sistema físico:
  - Átomo de dos niveles
  - Campo láser resonante (Delta = 0)
  - Decaimiento espontáneo (gamma)
  - Frecuencia de Rabi (Omega)

Ecuaciones resueltas:
  d(rho_ee)/dt = -i*(Omega/2)*(rho_ge - rho_eg) - gamma*rho_ee
  d(rho_gg)/dt = -i*(Omega/2)*(rho_eg - rho_ge) + gamma*rho_ee
  d(rho_eg)/dt = -i*(Delta*rho_eg + Omega/2*(rho_gg - rho_ee)) - (gamma/2)*rho_eg

Restricciones físicas:
  1. Tr(rho) = rho_ee + rho_gg = 1
  2. Tr(rho^2) = rho_ee^2 + rho_gg^2 + 2*|rho_eg|^2 <= 1
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

# ─── Reproducibilidad ────────────────────────────────────────────────────────
np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  ARQUITECTURA DE LA RED NEURONAL
# ═══════════════════════════════════════════════════════════════════════════════

class RabiPINN(nn.Module):
    """
    Red neuronal que aproxima la solución de las Ecuaciones de Bloch.

    Entrada : t  ∈ [0, T_max]   (tiempo, escalar)
    Salida  : [rho_ee, rho_gg, Re(rho_eg), Im(rho_eg)]

    Restricciones físicas impuestas en la arquitectura:
      • rho_ee, rho_gg ∈ (0,1)  →  sigmoid + normalización
      • Tr(rho) = 1              →  normalización exacta
      • |rho_eg|² ≤ rho_ee*rho_gg  →  escala por desigualdad Cauchy-Schwarz
    """

    def __init__(self, hidden_layers=4, neurons=64):
        super().__init__()

        # Capas ocultas
        dims = [1] + [neurons] * hidden_layers
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            nn.init.xavier_normal_(layers[-1].weight)
            nn.init.zeros_(layers[-1].bias)
            layers.append(nn.Tanh())
        self.trunk = nn.Sequential(*layers)

        # Capa de salida: 4 valores crudos
        self.head = nn.Linear(neurons, 4)
        nn.init.xavier_normal_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        self.eps = 1e-7   # estabilidad numérica

    def forward(self, t):
        raw = self.head(self.trunk(t))          # (N, 4)  sin restricciones

        # ── Poblaciones ───────────────────────────────────────────────────────
        ree_u = torch.sigmoid(raw[:, 0:1])
        rgg_u = torch.sigmoid(raw[:, 1:2])
        denom = ree_u + rgg_u + self.eps        # garantiza Tr(rho)=1
        rho_ee = ree_u / denom
        rho_gg = rgg_u / denom

        # ── Coherencias con límite de Cauchy-Schwarz ──────────────────────────
        # |rho_eg|^2 ≤ rho_ee * rho_gg  →  |rho_eg| ≤ sqrt(rho_ee * rho_gg)
        max_coh = torch.sqrt(rho_ee * rho_gg + self.eps) * 0.99   # margen 1 %
        rho_eg_re = torch.tanh(raw[:, 2:3]) * max_coh
        rho_eg_im = torch.tanh(raw[:, 3:4]) * max_coh

        return rho_ee, rho_gg, rho_eg_re, rho_eg_im


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  CLASE PRINCIPAL PINN
# ═══════════════════════════════════════════════════════════════════════════════

class RabiOscillationPINN:
    """
    PINN para las Oscilaciones de Rabi Amortiguadas.

    Función de pérdida total:
      L = λ_phys * L_physics
        + λ_ic   * L_ic
        + λ_con  * L_constraint
    """

    def __init__(self, Omega=1.0, Delta=0.0, gamma=0.1, device="cpu"):
        self.Omega  = Omega
        self.Delta  = Delta
        self.gamma  = gamma
        self.device = torch.device(device)
        self.eps    = 1e-7

        # Red neuronal
        self.net = RabiPINN(hidden_layers=4, neurons=64).to(self.device)

        # Optimizador y scheduler
        self.optimizer = optim.Adam(self.net.parameters(), lr=5e-4)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=5000, gamma=0.9
        )

        self.history = []   # historial de pérdidas

        n_params = sum(p.numel() for p in self.net.parameters())
        print(f"Parámetros totales de la red: {n_params:,}")
        print(f"  Ω = {Omega},  Δ = {Delta},  γ = {gamma}\n")

    # ── Derivadas automáticas ────────────────────────────────────────────────

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

    # ── Pérdidas ─────────────────────────────────────────────────────────────

    def loss_physics(self, t):
        """
        Residuos de las Ecuaciones de Bloch:

          d(rho_ee)/dt = -i*(Ω/2)*(rho_ge - rho_eg) - γ*rho_ee
          d(rho_gg)/dt = -i*(Ω/2)*(rho_eg - rho_ge) + γ*rho_ee
          d(rho_eg)/dt = -i*(Δ*rho_eg + Ω/2*(rho_gg - rho_ee)) - (γ/2)*rho_eg
        """
        ree, rgg, reg_re, reg_im, \
        d_ree, d_rgg, d_reg_re, d_reg_im = self._derivatives(t)

        Omega = self.Omega
        Delta = self.Delta
        gamma = self.gamma

        # ρ_eg = reg_re + i*reg_im
        # ρ_ge = ρ_eg* = reg_re - i*reg_im

        # Ec. rho_ee: parte real de -i*(Ω/2)*(ρ_ge - ρ_eg) - γ*ρ_ee
        #   ρ_ge - ρ_eg = -2i * reg_im
        #   -i*(Ω/2)*(-2i*reg_im) = -Ω * reg_im
        rhs_ee = -Omega * reg_im - gamma * ree
        eq_ee  = d_ree - rhs_ee

        # Ec. rho_gg: parte real de -i*(Ω/2)*(ρ_eg - ρ_ge) + γ*ρ_ee
        #   ρ_eg - ρ_ge = +2i * reg_im
        #   -i*(Ω/2)*(2i*reg_im) = +Ω * reg_im
        rhs_gg = Omega * reg_im + gamma * ree
        eq_gg  = d_rgg - rhs_gg

        # Ec. rho_eg (parte real e imaginaria):
        #   d_rho_eg/dt = -i*(Δ*ρ_eg + Ω/2*(ρ_gg - ρ_ee)) - (γ/2)*ρ_eg
        # Separando Re e Im:
        #   Re: d_reg_re/dt = +Delta*reg_im - (gamma/2)*reg_re  +  Omega/2*(rgg-ree)*0
        #                                  ← parte Im del operador -i*(...)
        #   La parte -i*(...):
        #     -i*(Δ*(re+i*im) + (Ω/2)*X)
        #     = -i*Δ*re + Δ*im - i*(Ω/2)*X
        #   donde X = rgg - ree (real)
        #   Re: +Delta*reg_im - (Omega/2)*X   [de -i*(Ω/2)*X → Im parte es -(Ω/2)*X real]
        #   Wait — desarrollamos cuidadosamente:
        #
        #   Sea A = Δ*ρ_eg + (Ω/2)*(ρ_gg - ρ_ee)
        #         = [Δ*re + (Ω/2)*X] + i*[Δ*im]        (X = rgg-ree, real)
        #   -i*A = -i*[Δ*re + (Ω/2)*X] + [Δ*im]
        #   Re(-i*A) = +Δ*im
        #   Im(-i*A) = -(Δ*re + (Ω/2)*X)
        #
        #   Entonces:
        #   Re: d_reg_re/dt = Δ*reg_im - (γ/2)*reg_re
        #   Im: d_reg_im/dt = -(Δ*reg_re + (Ω/2)*(rgg - ree)) - (γ/2)*reg_im

        X = rgg - ree
        rhs_eg_re = Delta * reg_im - (gamma / 2) * reg_re
        rhs_eg_im = -(Delta * reg_re + (Omega / 2) * X) - (gamma / 2) * reg_im

        eq_eg_re = d_reg_re - rhs_eg_re
        eq_eg_im = d_reg_im - rhs_eg_im

        return (torch.mean(eq_ee**2) +
                torch.mean(eq_gg**2) +
                torch.mean(eq_eg_re**2) +
                torch.mean(eq_eg_im**2))

    def loss_ic(self, t0, ic):
        """
        Condición inicial en t=0:
          ic = [rho_ee(0), rho_gg(0), Re(rho_eg)(0), Im(rho_eg)(0)]
        """
        ree, rgg, reg_re, reg_im = self.net(t0)
        return (torch.mean((ree - ic[0])**2) +
                torch.mean((rgg - ic[1])**2) +
                torch.mean((reg_re - ic[2])**2) +
                torch.mean((reg_im - ic[3])**2))

    def loss_constraint(self, t):
        """
        Restricciones físicas:
          1. Tr(rho) = rho_ee + rho_gg = 1
          2. Pureza P = rho_ee² + rho_gg² + 2|rho_eg|² ≤ 1
        """
        ree, rgg, reg_re, reg_im = self.net(t)

        # 1. Traza
        trace_loss = torch.mean((ree + rgg - 1.0)**2)

        # 2. Pureza
        P = ree**2 + rgg**2 + 2*(reg_re**2 + reg_im**2)
        purity_viol = torch.clamp(P - 1.0, min=0.0)
        purity_loss = torch.mean(purity_viol**2)

        return trace_loss + 10.0 * purity_loss

    # ── Entrenamiento ────────────────────────────────────────────────────────

    def train(self, T_max=20.0, N_col=2000, epochs=20000,
              lam_phys=1.0, lam_ic=200.0, lam_con=20.0,
              ic=None):
        """
        Parámetros
        ----------
        T_max   : tiempo final
        N_col   : puntos de colocación por época
        epochs  : épocas de entrenamiento
        lam_*   : pesos de cada término de pérdida
        ic      : condición inicial [ree, rgg, re_eg, im_eg]
        """
        if ic is None:
            ic = [0.0, 1.0, 0.0, 0.0]   # átomo en estado fundamental

        ic_tensor = torch.tensor(ic, dtype=torch.float32, device=self.device)
        t0 = torch.tensor([[0.0]], dtype=torch.float32, device=self.device)

        nan_count = 0

        pbar = tqdm(range(epochs), desc="Entrenando PINN")
        for epoch in pbar:

            # Puntos de colocación aleatorios en (0, T_max]
            t_col = (torch.rand(N_col, 1, device=self.device) * T_max
                     ).requires_grad_(True)

            self.optimizer.zero_grad()

            # ── Pérdidas ─────────────────────────────────────────────────────
            L_phys = self.loss_physics(t_col)
            L_ic   = self.loss_ic(t0, ic_tensor)
            L_con  = self.loss_constraint(t_col)

            loss = lam_phys * L_phys + lam_ic * L_ic + lam_con * L_con

            # Detectar NaN
            if torch.isnan(loss):
                nan_count += 1
                if nan_count > 10:
                    print("\n⚠ Demasiados NaN. Deteniendo.")
                    break
                continue

            nan_count = 0
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.scheduler.step()

            self.history.append({
                "total": loss.item(),
                "physics": L_phys.item(),
                "ic": L_ic.item(),
                "constraint": L_con.item(),
            })

            # Actualizar barra cada 50 épocas
            if epoch % 50 == 0:
                with torch.no_grad():
                    ree, rgg, reg_re, reg_im = self.net(t_col)
                    P_avg = torch.mean(ree**2 + rgg**2 + 2*(reg_re**2 + reg_im**2))
                pbar.set_postfix({
                    "Loss": f"{loss.item():.4f}",
                    "Phys": f"{L_phys.item():.4f}",
                    "IC"  : f"{L_ic.item():.4f}",
                    "P"   : f"{P_avg.item():.4f}",
                })

    # ── Predicción ───────────────────────────────────────────────────────────

    def predict(self, t_array):
        """Evalúa la red en un array numpy de tiempos."""
        self.net.eval()
        with torch.no_grad():
            t_t = torch.tensor(
                t_array.reshape(-1, 1), dtype=torch.float32, device=self.device
            )
            ree, rgg, reg_re, reg_im = self.net(t_t)
            P = ree**2 + rgg**2 + 2*(reg_re**2 + reg_im**2)

        self.net.train()
        return {
            "rho_ee"  : ree.cpu().numpy().flatten(),
            "rho_gg"  : rgg.cpu().numpy().flatten(),
            "reg_re"  : reg_re.cpu().numpy().flatten(),
            "reg_im"  : reg_im.cpu().numpy().flatten(),
            "purity"  : P.cpu().numpy().flatten(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  SOLUCIÓN DE REFERENCIA CON SCIPY (RK45)
# ═══════════════════════════════════════════════════════════════════════════════

def solve_reference(Omega=1.0, Delta=0.0, gamma=0.1, T_max=20.0, N=500):
    """Integra las ecuaciones de Bloch con RK45 para comparación."""

    def odes(t, y):
        ree, rgg, re, im = y
        reg = re + 1j * im
        rge = reg.conjugate()

        d_ree = -1j * (Omega / 2 * (rge - reg)) - gamma * ree
        d_rgg = -1j * (Omega / 2 * (reg - rge)) + gamma * ree
        d_reg = -1j * (Delta * reg + Omega / 2 * (rgg - ree)) - (gamma / 2) * reg

        return [d_ree.real, d_rgg.real, d_reg.real, d_reg.imag]

    t_eval = np.linspace(0, T_max, N)
    sol = solve_ivp(odes, [0, T_max], [0., 1., 0., 0.],
                    t_eval=t_eval, method="RK45", rtol=1e-9, atol=1e-11)

    ree = sol.y[0]
    rgg = sol.y[1]
    re  = sol.y[2]
    im  = sol.y[3]
    P   = ree**2 + rgg**2 + 2*(re**2 + im**2)

    return sol.t, ree, rgg, re, im, P


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  VISUALIZACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

def plot_all(t_ref, ref, t_eval, pred, history):
    """
    Genera 3 figuras:
      Fig 1 – Comparación PINN vs RK45
      Fig 2 – Verificación de restricciones físicas
      Fig 3 – Evolución de las pérdidas durante el entrenamiento
    """

    # ── Figura 1: Comparación principal ─────────────────────────────────────
    fig1, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig1.suptitle("PINN vs RK45: Oscilaciones de Rabi Amortiguadas",
                  fontsize=14, fontweight="bold")

    kw_ref  = dict(color="black", lw=1.5, linestyle="--", label="RK45 (referencia)")
    kw_pinn = dict(color="crimson", lw=2,  label="PINN")

    # ρ_ee
    axes[0, 0].plot(t_ref, ref["ree"],  **kw_ref)
    axes[0, 0].plot(t_eval, pred["rho_ee"], **kw_pinn)
    axes[0, 0].set_title("Población Excitada  ρ_ee")
    axes[0, 0].set_ylabel("Probabilidad")
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    # ρ_gg
    axes[0, 1].plot(t_ref, ref["rgg"],  **kw_ref)
    axes[0, 1].plot(t_eval, pred["rho_gg"], color="steelblue", lw=2, label="PINN")
    axes[0, 1].plot(t_ref, ref["rgg"], **kw_ref)
    axes[0, 1].set_title("Población Fundamental  ρ_gg")
    axes[0, 1].set_ylabel("Probabilidad")
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    # Pureza
    axes[1, 0].plot(t_ref, ref["P"],  **kw_ref)
    axes[1, 0].plot(t_eval, pred["purity"], color="darkgreen", lw=2, label="PINN")
    axes[1, 0].axhline(1.0, color="gray", ls=":", lw=1.2, label="P = 1")
    axes[1, 0].axhline(0.5, color="orange", ls=":", lw=1.2, label="P = 0.5")
    axes[1, 0].set_title("Pureza  Tr(ρ²)")
    axes[1, 0].set_ylabel("Pureza")
    axes[1, 0].set_ylim([0, 1.1])
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    # |ρ_eg|  (coherencias)
    coh_ref  = np.sqrt(ref["re"]**2  + ref["im"]**2)
    coh_pinn = np.sqrt(pred["reg_re"]**2 + pred["reg_im"]**2)
    axes[1, 1].plot(t_ref, coh_ref, **kw_ref)
    axes[1, 1].plot(t_eval, coh_pinn, color="purple", lw=2, label="PINN")
    axes[1, 1].set_title("Coherencias  |ρ_eg|")
    axes[1, 1].set_ylabel("|ρ_eg|")
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel("Tiempo (1/Ω)")
    fig1.tight_layout()

    # ── Figura 2: Restricciones físicas ─────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    fig2.suptitle("Verificación de Restricciones Físicas",
                  fontsize=13, fontweight="bold")

    # Traza
    trace = pred["rho_ee"] + pred["rho_gg"]
    axes2[0].plot(t_eval, trace, color="navy", lw=2)
    axes2[0].axhline(1.0, color="red", ls="--", lw=1.5, label="= 1 (exacto)")
    axes2[0].set_title("Tr(ρ) = ρ_ee + ρ_gg")
    axes2[0].set_ylim([0.95, 1.05])
    axes2[0].set_xlabel("Tiempo (1/Ω)")
    axes2[0].legend(); axes2[0].grid(alpha=0.3)

    # Pureza
    axes2[1].plot(t_eval, pred["purity"], color="darkgreen", lw=2, label="P(t)")
    axes2[1].axhline(1.0, color="red", ls="--", lw=1.5, label="P ≤ 1")
    axes2[1].fill_between(t_eval, 1.0, pred["purity"],
                          where=pred["purity"] > 1.0,
                          color="red", alpha=0.4, label="violación")
    axes2[1].set_title("Pureza  Tr(ρ²) ≤ 1")
    axes2[1].set_ylim([0, 1.15])
    axes2[1].set_xlabel("Tiempo (1/Ω)")
    axes2[1].legend(); axes2[1].grid(alpha=0.3)

    # Error punto a punto vs RK45
    t_common = np.linspace(0, 20.0, 500)
    err_ee = np.abs(pred["rho_ee"] - ref["ree"])
    axes2[2].semilogy(t_eval, err_ee + 1e-12, color="crimson", lw=2)
    axes2[2].set_title("Error absoluto  |ρ_ee^PINN − ρ_ee^RK45|")
    axes2[2].set_xlabel("Tiempo (1/Ω)")
    axes2[2].set_ylabel("Error (log)")
    axes2[2].grid(alpha=0.3)

    fig2.tight_layout()

    # ── Figura 3: Historial de pérdidas ─────────────────────────────────────
    fig3, axes3 = plt.subplots(1, 4, figsize=(16, 4))
    fig3.suptitle("Evolución de las Pérdidas durante el Entrenamiento",
                  fontsize=13, fontweight="bold")

    epochs = np.arange(len(history))
    keys   = ["total", "physics", "ic", "constraint"]
    titles = ["Total", "Física (Lindblad)", "C. Inicial", "Restricciones"]
    colors = ["black", "royalblue", "darkorange", "forestgreen"]

    for ax, k, ti, c in zip(axes3, keys, titles, colors):
        vals = [h[k] for h in history]
        ax.semilogy(epochs, vals, color=c, lw=1.5)
        ax.set_title(ti)
        ax.set_xlabel("Época")
        ax.set_ylabel("Pérdida (log)")
        ax.grid(alpha=0.3)

    fig3.tight_layout()

    return fig1, fig2, fig3


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  EJECUCIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Parámetros físicos ───────────────────────────────────────────────────
    Omega  = 1.0    # Frecuencia de Rabi
    Delta  = 0.0    # Desintonía
    gamma  = 0.1    # Decaimiento espontáneo
    T_max  = 20.0   # Tiempo final

    print("=" * 65)
    print("  PINN — Oscilaciones de Rabi Amortiguadas")
    print("=" * 65)
    print(f"  Ω = {Omega}   Δ = {Delta}   γ = {gamma}   T = {T_max}")
    print("=" * 65 + "\n")

    # ── Solución de referencia (RK45) ────────────────────────────────────────
    print("Calculando solución de referencia con RK45 …")
    t_ref, ree_r, rgg_r, re_r, im_r, P_r = solve_reference(
        Omega=Omega, Delta=Delta, gamma=gamma, T_max=T_max, N=500
    )
    ref = {"ree": ree_r, "rgg": rgg_r, "re": re_r, "im": im_r, "P": P_r}
    print("  ✓ RK45 listo\n")

    # ── Crear y entrenar PINN ────────────────────────────────────────────────
    pinn = RabiOscillationPINN(Omega=Omega, Delta=Delta, gamma=gamma,
                               device=str(device))

    pinn.train(
        T_max   = T_max,
        N_col   = 2000,       # puntos de colocación por época
        epochs  = 200000,      # épocas de entrenamiento
        lam_phys= 1.0,        # peso ecuaciones de Bloch
        lam_ic  = 200.0,      # peso condición inicial  (alto: crítico)
        lam_con = 20.0,       # peso restricciones físicas
        ic      = [0., 1., 0., 0.],   # ρ_gg(0) = 1
    )

    # ── Predicciones PINN ────────────────────────────────────────────────────
    print("\nGenerando predicciones …")
    t_eval = np.linspace(0, T_max, 500)
    pred   = pinn.predict(t_eval)

    # ── Métricas ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  MÉTRICAS FINALES")
    print("=" * 65)

    max_P   = np.max(pred["purity"])
    mean_P  = np.mean(pred["purity"])
    n_viol  = np.sum(pred["purity"] > 1.0)
    err_ee  = np.abs(pred["rho_ee"] - ree_r)
    trace   = pred["rho_ee"] + pred["rho_gg"]

    print(f"  Pureza máxima            : {max_P:.6f}")
    print(f"  Pureza promedio          : {mean_P:.6f}")
    print(f"  Violaciones P > 1        : {n_viol} / {len(pred['purity'])}")
    print(f"  Error MAE en ρ_ee        : {np.mean(err_ee):.6f}")
    print(f"  Error máx en ρ_ee        : {np.max(err_ee):.6f}")
    print(f"  Traza máx  |Tr-1|        : {np.max(np.abs(trace - 1)):.2e}")
    print(f"  Estado        : {'✓ VÁLIDO' if max_P <= 1.0 and n_viol == 0 else '✗ INVÁLIDO'}")
    print("=" * 65)

    # ── Gráficas ─────────────────────────────────────────────────────────────
    print("\nGenerando gráficas …")
    fig1, fig2, fig3 = plot_all(t_ref, ref, t_eval, pred, pinn.history)

    fig1.savefig("rabi_pinn_comparison.png",   dpi=300, bbox_inches="tight")
    fig2.savefig("rabi_pinn_constraints.png",  dpi=300, bbox_inches="tight")
    fig3.savefig("rabi_pinn_loss_history.png", dpi=300, bbox_inches="tight")

    print("  ✓ rabi_pinn_comparison.png")
    print("  ✓ rabi_pinn_constraints.png")
    print("  ✓ rabi_pinn_loss_history.png")

    plt.show()
    print("\n¡Listo!")
