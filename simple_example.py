"""
example_debug.py

Collection of usage examples of QuantumSCC with DEBUG mode parameter.
Allows step-by-step visualization of Hamiltonian construction.
"""

from QuantumSCC import Circuit, Capacitor, Inductor, Junction


def run_lc_oscillator():
    """
    Example 1: LC Oscillator (Linear Circuit)
    Section IV.A of the Master's Thesis.
    """
    print("\n" + "="*60)
    print("RUNNING: LC OSCILLATOR")
    print("="*60)

    # 1. Define elements
    C = Capacitor(value=0.1, unit='nF')
    L = Inductor(value=1, unit='nH')

    # 2. Define topology: L and C in parallel (simple loop)
    topology = [
        (0, 1, L),
        (0, 1, C)
    ]

    # 3. Instantiate with debug=True
    circuit = Circuit(topology, debug=True)

    # 4. Result (Diagonalized Harmonic Hamiltonian)
    circuit.diagonal_harmonic_Hamiltonian_expression()


def run_coupled_lc():
    """
    Example 2: Coupled LC Oscillators (Linear Circuit)
    Section IV.B of the Master's Thesis.
    """
    print("\n" + "="*60)
    print("RUNNING: COUPLED LC OSCILLATORS")
    print("="*60)

    # 1. Define elements
    C = Capacitor(value=0.1, unit='nF')
    Cg = Capacitor(value=0.2, unit='nF')  # Coupling capacitor
    L = Inductor(value=1, unit='nH')

    # 2. Define topology (Two LC loops coupled by Cg)
    # Node 0: Common ground (or reference)
    # Node 1: Active node of oscillator 1
    # Node 2: Active node of oscillator 2
    topology = [
        (0, 1, L),   # Inductor of oscillator 1
        (0, 1, C),   # Capacitor of oscillator 1
        (1, 2, Cg),  # Coupling between nodes 1 and 2
        (2, 0, L),   # Inductor of oscillator 2
        (2, 0, C)    # Capacitor of oscillator 2
    ]

    # 3. Instantiate with debug=True
    circuit = Circuit(topology, debug=True)

    # 4. Result
    circuit.diagonal_harmonic_Hamiltonian_expression()


def run_fluxonium():
    """
    Example 3: Fluxonium (Nonlinear Circuit)
    Section V.A of the Master's Thesis.
    """
    print("\n" + "="*60)
    print("RUNNING: FLUXONIUM")
    print("="*60)

    # 1. Define elements
    # In QuantumSCC, the Josephson junction requires an explicit parallel capacitor
    C_J = Capacitor(value=0.1, unit='nF')
    J = Junction(value=1, unit='GHz', cap=C_J)
    L = Inductor(value=1, unit='nH')

    # 2. Define topology (L in parallel with J)
    topology = [
        (0, 1, J),
        (0, 1, L)
    ]

    # 3. Instantiate with debug=True
    circuit = Circuit(topology, debug=True)

    # 4. Result (Full Nonlinear Hamiltonian)
    circuit.Hamiltonian_expression()


def run_singular_circuit():
    """
    Example 4: Singular Circuit (Nonlinear Circuit)
    Section V.B of the Master's Thesis (Fig. 5).
    This circuit is critical for testing topological robustness.
    """
    print("\n" + "="*60)
    print("RUNNING: SINGULAR CIRCUIT (Fig. 5)")
    print("="*60)

    # 1. Define elements
    C = Capacitor(value=0.1, unit='nF')
    C_J = Capacitor(value=0.1, unit='nF')
    J = Junction(value=1, unit='GHz', cap=C_J)
    L = Inductor(value=1, unit='nH')

    # 2. Define topology
    # Inductor in series with C, closing the loop with the junction J
    topology = [
        (0, 1, J),  # Branch 1: Junction (with implicit C_J)
        (1, 2, L),  # Branch 2: Inductor
        (2, 0, C)   # Branch 3: Series capacitor
    ]

    # 3. Instantiate with debug=True
    circuit = Circuit(topology, debug=True)

    # 4. Result
    circuit.Hamiltonian_expression()


if __name__ == "__main__":
    """
    Uncomment the function you want to run.
    You may run several in sequence if you want to see all logs.
    """

    # --- LINEAR EXAMPLES ---
    run_lc_oscillator()
    run_coupled_lc()

    # --- NONLINEAR EXAMPLES ---
    # run_fluxonium()
    # run_singular_circuit()
