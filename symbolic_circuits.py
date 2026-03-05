"""
symbolic_circuits.py

Prints the symbolic Hamiltonian (E_C, E_L, E_J, E_P as sympy symbols)
followed by the numerical Hamiltonian for each circuit family.

Run:
    python symbolic_circuits.py              # todos los circuitos
    python symbolic_circuits.py lc           # solo LC oscilador
    python symbolic_circuits.py transmon     # solo transmon
    ...

Claves disponibles:
  lc, two_lc, transmon, fluxonium,
  dual_transmon, dual_fluxonium,
  jj_qps_parallel, jj_qps_chain,
  two_jj_series, two_jj_one_qps, two_qps_one_jj,
  triangle, two_caps_parallel, two_caps_series,
  lc_coupled, star
"""

import sys
sys.path.insert(0, '.')

from QuantumSCC import Circuit, Capacitor, Inductor, Junction, PhaseSlip

# ── helpers ───────────────────────────────────────────────────────────────────

SEP = "═" * 70

def header(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

def run(title, edges):
    header(title)
    Circuit(edges).symbolic_hamiltonian_expression()


# ── 1. LC oscilador ──────────────────────────────────────────────────────────
# Circuito más simple: 1 modo armónico.
# H = 2·E_C·n² + 2·E_L·φ²   con ω = √(4·E_C·E_L)

def lc():
    run("LC oscilador  (1 modo armónico)",
        [(0, 1, Capacitor(1, 'pF')),
         (0, 1, Inductor(1, 'nH'))])


# ── 2. Dos LC independientes ─────────────────────────────────────────────────
# Dos osciladores sin acoplamiento, 2 modos.

def two_lc():
    run("2 LC independientes  (2 modos desacoplados)",
        [(0, 1, Capacitor(1, 'pF')), (0, 1, Inductor(1, 'nH')),
         (0, 2, Capacitor(1, 'pF')), (0, 2, Inductor(1, 'nH'))])


# ── 3. Transmon ──────────────────────────────────────────────────────────────
# JJ ∥ C.  Variable compacta de flujo φ_c.
# H = 4·E_C·n_c² − E_J·cos(φ_c)

def transmon():
    run("Transmon  (JJ ∥ C  →  4·E_C·n² − E_J·cos φ)",
        [(0, 1, Junction(1, 'GHz', cap=Capacitor(1, 'pF')))])


# ── 4. Fluxonium (JJ ∥ L) ───────────────────────────────────────────────────
# El inductor extiende el flujo compacto del JJ → 1 modo armónico.
# H = 2·E_C·n² + 2·E_L·φ² − E_J·cos(φ)

def fluxonium():
    run("Fluxonium  (JJ ∥ L  →  2·E_C·n² + 2·E_L·φ² − E_J·cos φ)",
        [(0, 1, Junction(1, 'GHz', cap=Capacitor(1, 'pF'))),
         (0, 1, Inductor(1, 'nH'))])


# ── 5. Dual-transmon (QPS) ───────────────────────────────────────────────────
# Dual exacto del transmon: carga compacta q_c.
# H = 2·E_L·ψ_c² − E_P·cos(q_c)

def dual_transmon():
    run("Dual-transmon  (QPS ∥ L  →  2·E_L·ψ² − E_P·cos q)",
        [(0, 1, PhaseSlip(1, 'GHz', ind=Inductor(1, 'nH')))])


# ── 6. Dual-fluxonium (QPS ∥ C) ─────────────────────────────────────────────
# El capacitor extiende la carga compacta del QPS → 1 modo armónico.
# Dual del fluxonium: los roles de L y C están intercambiados.

def dual_fluxonium():
    run("Dual-fluxonium  (QPS ∥ C  →  dual del fluxonium)",
        [(0, 1, PhaseSlip(1, 'GHz', ind=Inductor(1, 'nH'))),
         (0, 1, Capacitor(1, 'pF'))])


# ── 7. JJ ∥ QPS (mismos nodos) ───────────────────────────────────────────────
# B2 regression: JJ y QPS en paralelo entre los mismos nodos.
# TEF_22 es singular → requiere pseudoinversa (fix del Schur).

def jj_qps_parallel():
    run("JJ ∥ QPS  (mismos nodos — B2 regression)",
        [(0, 1, Junction(1, 'GHz', cap=Capacitor(1, 'pF'))),
         (0, 1, PhaseSlip(1, 'GHz', ind=Inductor(1, 'nH')))])


# ── 8. JJ-QPS en cadena (nodos distintos) ────────────────────────────────────
# B3 regression: JJ en (0→1) y QPS en (1→2), variable compacta de flujo
# del JJ se acopla con la variable compacta de carga del QPS.

def jj_qps_chain():
    run("JJ-QPS cadena  (nodos distintos — B3 regression)",
        [(0, 1, Junction(1, 'GHz', cap=Capacitor(1, 'pF'))),
         (1, 2, PhaseSlip(1, 'GHz', ind=Inductor(1, 'nH')))])


# ── 9. Dos JJ en serie ───────────────────────────────────────────────────────
# Dos JJ en serie con cap de cierre: 2 variables compactas de flujo.
# La ecuación cuadrática resulta en términos cruzados entre E_C1,E_C2,E_C3
# porque el Schur complement de los 3 capacitores en el sector no-dinámico
# produce una forma cuadrática racional en los E_C.

def two_jj_series():
    run("2 JJ en serie  (2 modos compactos de flujo)",
        [(0, 1, Junction(1, 'GHz', cap=Capacitor(1, 'pF'))),
         (1, 2, Junction(1, 'GHz', cap=Capacitor(1, 'pF'))),
         (0, 2, Capacitor(1, 'pF'))])


# ── 10. Dos JJ + 1 QPS ──────────────────────────────────────────────────────
# Anillo con 2 JJ y 1 QPS.
# 1 modo compacto de flujo (JJ) + 1 modo compacto de carga (QPS).
# La topología de anillo mezcla las variables vía la topología.

def two_jj_one_qps():
    run("2 JJ + 1 QPS  (anillo mixto JJ-QPS)",
        [(0, 1, Junction(1, 'GHz', cap=Capacitor(1, 'pF'))),
         (1, 2, Junction(1, 'GHz', cap=Capacitor(1, 'pF'))),
         (2, 0, PhaseSlip(1, 'GHz', ind=Inductor(1, 'nH')))])


# ── 11. Dos QPS + 1 JJ (anillo) ─────────────────────────────────────────────
# Dual del caso anterior: 2 QPS y 1 JJ en anillo.
# 2 modos compactos de carga + 1 modo compacto de flujo.

def two_qps_one_jj():
    run("2 QPS + 1 JJ  (anillo mixto QPS-JJ)",
        [(0, 1, PhaseSlip(1, 'GHz', ind=Inductor(1, 'nH'))),
         (1, 2, PhaseSlip(1, 'GHz', ind=Inductor(1, 'nH'))),
         (2, 0, Junction(1, 'GHz', cap=Capacitor(1, 'pF')))])


# ── 12. Triángulo (C + 2L) ───────────────────────────────────────────────────
# C en rama (0-2), L en (0-1) y L en (1-2).
# El Schur del inductor no-dinámico produce un inductor efectivo
# L_eff = L1·L2/(L1+L2) (inductores en paralelo).

def triangle():
    run("Triángulo  (C en 0-2, L en 0-1, L en 1-2)",
        [(0, 2, Capacitor(1, 'GHz')),
         (0, 1, Inductor(1, 'GHz')),
         (1, 2, Inductor(1, 'GHz'))])


# ── 13. Dos capacitores en paralelo + 1 inductor ─────────────────────────────
# C1 ∥ C2 ∥ L: el Schur de los dos capacitores en el sector no-dinámico
# produce un único C_eff = C1·C2/(C1+C2) → E_C_eff = E_C1+E_C2 (en paralelo
# los capacitores suman).

def two_caps_parallel():
    run("2 caps en paralelo + 1 inductor  (C_eff = C1+C2)",
        [(0, 1, Capacitor(1, 'pF')),
         (0, 1, Capacitor(1, 'pF')),
         (0, 1, Inductor(1, 'nH'))])


# ── 14. Dos capacitores en serie + 1 inductor ────────────────────────────────
# C1 — C2 — L: capacitores en serie → C_eff = C1+C2 (en serie las E_C suman).

def two_caps_series():
    run("2 caps en serie + 1 inductor  (E_C_eff = E_C1+E_C2)",
        [(0, 1, Capacitor(1, 'pF')),
         (1, 2, Capacitor(1, 'pF')),
         (2, 0, Inductor(1, 'nH'))])


# ── 15. LC acoplados mediante capacitor de coupling ───────────────────────────
# L1 en (0-1), L2 en (2-0), C1 en (0-1), Cg en (1-2), C2 en (2-0).
# El Schur de los 3 capacitores en el sector no-dinámico introduce
# términos RACIONALES en los E_C (ej: −4E_C1²/(2E_C1+2E_C2+2E_Cg)).
# Esto es esperable: la inversión de la matriz de capacitancia produce
# cocientes de energías → la expresión simbólica NO es lineal en E_C.

def lc_coupled():
    run("LC acoplados vía Cg  (términos racionales en E_C por el Schur)",
        [(0, 1, Inductor(1, 'nH')),
         (1, 2, Capacitor(2, 'pF')),
         (2, 0, Inductor(1, 'nH')),
         (0, 1, Capacitor(1, 'pF')),
         (2, 0, Capacitor(1, 'pF'))])


# ── 16. Estrella simétrica (3C + 3L) ─────────────────────────────────────────
# 3 capacitores en triángulo (0-1, 1-2, 2-0) + 3 inductores al nodo central (3).
# 2 modos degenerados. El Schur introduce términos cruzados
# 4·E_L2·φ_e1·φ_e2 entre los dos modos.

def star():
    run("Estrella simétrica  (3C + 3L — 2 modos degenerados)",
        [(0, 1, Capacitor(1, 'pF')), (1, 2, Capacitor(1, 'pF')), (2, 0, Capacitor(1, 'pF')),
         (0, 3, Inductor(1, 'nH')), (1, 3, Inductor(1, 'nH')), (2, 3, Inductor(1, 'nH'))])


# ── dispatcher ────────────────────────────────────────────────────────────────

CIRCUITS = {
    'lc':              lc,
    'two_lc':          two_lc,
    'transmon':        transmon,
    'fluxonium':       fluxonium,
    'dual_transmon':   dual_transmon,
    'dual_fluxonium':  dual_fluxonium,
    'jj_qps_parallel': jj_qps_parallel,
    'jj_qps_chain':    jj_qps_chain,
    'two_jj_series':   two_jj_series,
    'two_jj_one_qps':  two_jj_one_qps,
    'two_qps_one_jj':  two_qps_one_jj,
    'triangle':        triangle,
    'two_caps_parallel': two_caps_parallel,
    'two_caps_series': two_caps_series,
    'lc_coupled':      lc_coupled,
    'star':            star,
}

if __name__ == '__main__':
    keys = sys.argv[1:] or list(CIRCUITS.keys())
    for k in keys:
        if k not in CIRCUITS:
            print(f"Clave desconocida: '{k}'. Disponibles: {list(CIRCUITS)}")
        else:
            CIRCUITS[k]()
