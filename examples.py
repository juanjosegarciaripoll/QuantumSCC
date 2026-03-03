"""
QuantumSCC Debug Utilities (Terminal Optimized)

This module contains functions to inspect all intermediate steps 
of the Hamiltonian derivation for superconducting circuits.

It converts LaTeX equations into pretty Unicode text for terminal display.
"""

from QuantumSCC import *
import numpy as np

def prettify_latex(latex_str):
    """
    Converts basic LaTeX commands to Unicode characters for cleaner 
    terminal output.
    """
    replacements = {
        # Operators and symbols
        r"\hat{H}": "Ĥ",
        r"\hat{\xi}": "ξ̂",
        r"\hat{\Psi}": "Ψ̂",
        r"\hat{A}": "Â",
        r"\hat{a}": "â",
        r"\hat{\varphi}": "φ̂",
        r"\hat{Q}": "Q̂",
        r"\hat{\Phi}": "Φ̂",
        r"\tilde{H}": "H̃",
        r"\dot{R}": "Ṙ",
        r"\dot{Z}": "Ż",
        r"\dagger": "†",
        r"\sum": "∑",
        r"\int": "∫",
        r"\wedge": "∧",
        r"\infty": "∞",
        r"\in": "∈",
        r"\quad": "   ",
        r"\approx": "≈",
        
        # Greek
        r"\omega": "ω",
        r"\Omega": "Ω",
        r"\phi": "φ",
        r"\Phi": "Φ",
        r"\xi": "ξ",
        r"\hbar": "ħ",
        r"\pi": "π",
        
        # Structure and Sub/Superscripts
        r"\frac{1}{2}": "½",
        r"\frac": "", 
        r"^{T}": "ᵀ",
        r"^T": "ᵀ",
        r"^{-1}": "⁻¹",
        r"_{2B}": "₂ʙ",
        r"_{cut}": "꜀ᵤₜ",
        r"_{loop}": "ₗₒₒₚ",
        r"\mathbb{I}": "𝟙",
        r"\mathbb{R}": "ℝ",
        r"\mathcal{N}": "N",
        r"\mathcal{P}": "P",
        
        # Cleanup
        "{": "",
        "}": "",
        r"\left(": "(",
        r"\right)": ")",
        r"\beginpmatrix": "[",
        r"\endpmatrix": "]",
        r"\\": "\n      "
    }
    
    clean_str = latex_str
    for latex, unicode_char in replacements.items():
        clean_str = clean_str.replace(latex, unicode_char)
    
    return clean_str

def print_fancy_header(title, equations):
    """
    Prints a header with a box and formatted equations.
    Calculates padding dynamically to ensure the box closes correctly.
    """
    BOX_WIDTH = 75
    
    # Draw Top
    print("\n" + "╔" + "═"*(BOX_WIDTH-2) + "╗")
    
    # Draw Title
    title_text = title.center(BOX_WIDTH-4)
    print(f"║ {title_text} ║")
    
    # Draw Separator
    print("╠" + "═"*(BOX_WIDTH-2) + "╣")
    
    # Draw Equations
    for label, latex in equations:
        pretty_eq = prettify_latex(latex)
        
        # Line 1: Label
        label_line = f" • {label}:"
        padding_label = " " * (BOX_WIDTH - 2 - len(label_line) - 1)
        print(f"║{label_line}{padding_label}║")
        
        # Line 2: Equation (indented)
        # Note: We need to handle potential multiline equations manually if they exist,
        # but here we assume mostly single line.
        eq_content = f"     {pretty_eq}"
        
        # Calculate padding carefully. 
        # len() counts characters. If Unicode chars render wider, visual alignment might vary slightly,
        # but this logic ensures the character count is correct for the box border.
        pad_len = BOX_WIDTH - 2 - len(eq_content) - 1
        
        if pad_len > 0:
            print(f"║{eq_content}{' ' * pad_len}║")
        else:
            # Fallback if equation is too long
            print(f"║{eq_content}║")
            
        # Empty Spacer Line
        print(f"║{' ' * (BOX_WIDTH-2)}║")
        
    # Draw Bottom
    print("╚" + "═"*(BOX_WIDTH-2) + "╝")

def introspect_circuit_class():
    """
    Function to explore the internal structure of the Circuit class.
    Useful for understanding the API before refactoring.
    """
    print("="*60)
    print("CIRCUIT CLASS INTROSPECTION")
    print("="*60)
    
    # Create a simple circuit to explore
    C = Capacitor(value=0.1, unit='nF')
    L = Inductor(value=1, unit='nH')
    circuit = Circuit([(0,1,L), (0,1,C)])
    
    # Explore all attributes and methods
    print("\nPUBLIC METHODS:")
    public_methods = [method for method in dir(circuit) if not method.startswith('_') and callable(getattr(circuit, method))]
    for method in sorted(public_methods):
        print(f"  {method}()")
    
    print("\nPRIVATE/INTERNAL METHODS:")
    private_methods = [method for method in dir(circuit) if method.startswith('_') and callable(getattr(circuit, method)) and not method.startswith('__')]
    for method in sorted(private_methods):
        print(f"  {method}()")
        
    print("\nPUBLIC ATTRIBUTES:")
    public_attrs = [attr for attr in dir(circuit) if not attr.startswith('_') and not callable(getattr(circuit, attr))]
    for attr in sorted(public_attrs):
        try:
            value = getattr(circuit, attr)
            if hasattr(value, 'shape'):
                print(f"  {attr}: {type(value).__name__} shape {value.shape}")
            elif isinstance(value, (int, float, complex)):
                print(f"  {attr}: {value}")
            else:
                print(f"  {attr}: {type(value).__name__}")
        except:
            print(f"  {attr}: <could not access>")


def debug_circuit_steps(circuit_topology, name="Circuit"):
    """
    Displays all intermediate steps of the QuantumSCC calculation.
    
    Args:
        circuit_topology: List of tuples (node1, node2, element)
        name: Descriptive name of the circuit
    
    Returns:
        circuit: Circuit object for further analysis
    """
    print(f"\n{'#'*75}")
    print(f"   DEBUGGING STEPS FOR: {name.upper()}")
    print(f"{'#'*75}")
    
    # Create circuit
    print(f"\nTopology: {circuit_topology}")
    circuit = Circuit(circuit_topology)
    
    # --- Step 1: Kirchhoff Analysis ---
    eqs_step1 = [
        ("Eq (1) - KCL & KVL", r"KCL: \sum_{b \in \mathcal{N}} dq^b = 0,   KVL: \sum_{b \in \mathcal{P}} d\phi^b = 0"),
        ("Eq (2) - Constraints", r"F dR = 0   (F includes F_{loop} and F_{cut})"),
        ("Eq (3) - Kernel K", r"K = [Kernel(F_{loop}), Kernel(F_{cut})]"),
        ("Eq (6) - Reduction", r"R = K Z")
    ]
    print_fancy_header("1. KIRCHHOFF ANALYSIS", eqs_step1)
    
    try:
        # F matrix - Complete matrix of Kirchhoff constraints
        F = circuit.F
        print(f"  > F matrix (complete Kirchhoff constraints): shape {F.shape}")
        # print(f"F =\n{F}") # Uncomment to see full matrix
        print(f"  > Rank of F: {np.linalg.matrix_rank(F)}")
        
        # K matrix - Independent variables (kernel of F)
        K = circuit.K  
        print(f"  > K matrix (independent variables kernel): shape {K.shape}")
        # print(f"K =\n{K}") # Uncomment to see full matrix
        print(f"  > Rank of K: {np.linalg.matrix_rank(K)}")
        
        # Specific Kirchhoff matrices
        if hasattr(circuit, 'Fcut') and hasattr(circuit, 'Floop'):
            print(f"  > Fcut shape: {circuit.Fcut.shape}")
            print(f"  > Floop shape: {circuit.Floop.shape}")
            
        # Verification: F @ K should be zero
        FK_product = F @ K
        max_err = np.max(np.abs(FK_product))
        print(f"  > Verification F @ K ≈ 0: {'PASS' if max_err < 1e-10 else 'FAIL'} (Error: {max_err:.2e})")
        
    except Exception as e:
        print(f"Error accessing Kirchhoff matrices: {e}")
    
    # --- Step 2: Symplectic Form ---
    eqs_step2 = [
        ("Eq (8) - Symplectic Form", r"\omega_{2B} = \frac{1}{2} dR^T \wedge \Omega_{2B} dR"),
        ("Eq (14) - Canonical Form", r"V^T \Omega V = J = [0, \mathbb{I}; -\mathbb{I}, 0]"),
        ("Eq (16) - Basis Change", r"Z = V (\xi, w)^T")
    ]
    print_fancy_header("2. SYMPLECTIC FORM", eqs_step2)
    
    try:
        # omega_2B - Original 2-body symplectic form
        omega_2B = circuit.omega_2B
        print(f"  > omega_2B shape: {omega_2B.shape}")
        print(f"  > Determinant of omega_2B: {np.linalg.det(omega_2B):.10f}")
        
        # omega_symplectic - Reduced symplectic form
        omega_symplectic = circuit.omega_symplectic
        print(f"  > omega_symplectic shape: {omega_symplectic.shape}")
        
        # Basis change matrix V
        if hasattr(circuit, 'V'):
            V = circuit.V
            print(f"  > V matrix (basis change): shape {V.shape}")
            
    except Exception as e:
        print(f"Error accessing symplectic forms: {e}")
    
    # --- Step 3: Classical Hamiltonian ---
    eqs_step3 = [
        ("Eq (13) - Lagrangian", r"L(Z,\dot{Z}) = \frac{1}{2}\dot{Z}^T \Omega Z - \frac{1}{2}Z^T E Z + NonLinear"),
        ("Eq (18) - Eff. Hamiltonian", r"H = \tilde{H}_{\xi\xi} - \tilde{H}_{\xi w}\tilde{H}_{ww}^{-1}\tilde{H}_{w\xi}"),
        ("Eq (19) - Final Classical H", r"H(\xi) = \frac{1}{2}\xi^T H \xi - \sum E_{J} \cos(...)")
    ]
    print_fancy_header("3. CLASSICAL HAMILTONIAN", eqs_step3)
    
    try:
        # Main quadratic Hamiltonian
        H_quad = circuit.quadratic_hamiltonian
        print(f"  > quadratic_hamiltonian shape: {H_quad.shape}")
        print(f"  > Trace: {np.trace(H_quad):.6f}")
        
        # Hamiltonians in different bases
        if hasattr(circuit, 'FS_quadratic_hamiltonian_phiq'):
            H_phiq = circuit.FS_quadratic_hamiltonian_phiq
            print(f"  > FS_hamiltonian_phiq (φ-q coordinates): shape {H_phiq.shape}")
            
    except Exception as e:
        print(f"Error accessing Hamiltonians: {e}")
    
    # --- Step 4: Variables and Transformations ---
    print("\n" + "-"*30 + " 4. VARIABLES " + "-"*30)
    
    try:
        # Variable counters
        print("Variable counts:")
        print(f"  Total elements: {circuit.no_elements}")
        print(f"  Nodes: {circuit.no_nodes}")
        print(f"  Capacitors: {circuit.no_Capacitors}")
        print(f"  Inductors: {circuit.no_Inductors}")
        print(f"  Josephson Junctions: {circuit.no_JJ}")
        print(f"  Independent variables (modes): {circuit.no_independent_variables}")
        print(f"  Final compact flux: {circuit.no_final_compact_flux}")
        
        # Josephson vectors if they exist
        if hasattr(circuit, 'vector_JJ') and circuit.vector_JJ.shape[1] > 0:
            print(f"  > vector_JJ (Josephson directions): shape {circuit.vector_JJ.shape}")
            
        # Basis transformations if they exist
        if hasattr(circuit, 'T'):
            print(f"  > T matrix (transformation): shape {circuit.T.shape}")
            
        # Dimensionality reduction info
        original_dim = len(circuit_topology) * 2
        final_dim = circuit.no_independent_variables
        print(f"  > Dimensionality reduction: {original_dim} -> {final_dim}")
        
    except Exception as e:
        print(f"Error accessing variables/transformations: {e}")
    
    # --- Step 5: Diagonalization and Quantization ---
    eqs_step5 = [
        ("Eq (20) - Quantum H", r"\hat{H}(\hat{\xi}) = \frac{1}{2}\hat{\xi}^T H \hat{\xi} - \sum E_{J} \cos(\hat{\varphi}_j)"),
        ("Eq (25) - Ladder Ops", r"\hat{\Psi} = G \hat{A}   (G transforms to creation/annihilation)"),
        ("Eq (26) - Diagonal H_e", r"\hat{H}_e = \sum_{j=1}^M \hbar \omega_j \hat{a}_j^\dagger \hat{a}_j")
    ]
    print_fancy_header("5. DIAGONALIZATION & QUANTIZATION", eqs_step5)
    
    try:
        # Check if the circuit is purely linear (no Josephson junctions)
        has_josephson = any(isinstance(element, Junction) for _, _, element in circuit_topology)
        
        if not has_josephson:
            print("   >>> LINEAR CIRCUIT DETECTED <<<")
            
            # Show eigenvalues and eigenvectors of the quadratic Hamiltonian
            if hasattr(circuit, 'quadratic_hamiltonian'):
                H = circuit.quadratic_hamiltonian
                eigenvals = np.linalg.eigvals(H)
                # Convert to GHz
                frequencies = np.sqrt(np.abs(eigenvals.real)) / (2 * np.pi) 
                # Filter practically zero frequencies
                valid_freqs = frequencies[frequencies > 1e-5]
                
                print(f"  > Hamiltonian eigenvalues: {eigenvals}")
                print(f"  > Mode frequencies (GHz): {valid_freqs}")
                
            # Show the diagonalized result
            print(f"\n  > Diagonalized result string:")
            print("    " + "-"*40)
            circuit.diagonal_harmonic_Hamiltonian_expression(precision=4)
            print("    " + "-"*40)
            
        else:
            print("   >>> NONLINEAR CIRCUIT DETECTED <<<")
            print(f"  > Circuit has {circuit.no_JJ} Josephson junction(s)")
            
            # Show the complete Hamiltonian with nonlinear terms
            print(f"\n  > Complete nonlinear Hamiltonian string:")
            print("    " + "-"*40)
            circuit.Hamiltonian_expression(precision=4)
            print("    " + "-"*40)
            
    except Exception as e:
        print(f"Error in diagonalization analysis: {e}")
    
    # --- Step 6: Mathematical Verifications ---
    print("\n" + "-"*30 + " 6. VERIFICATIONS " + "-"*30)
    
    try:
        # Check 1: Symplectic property ω^T = -ω
        if hasattr(circuit, 'omega_symplectic'):
            omega = circuit.omega_symplectic
            is_antisymmetric = np.allclose(omega.T, -omega)
            print(f"  [x] Symplectic form is antisymmetric: {is_antisymmetric}")
            
        # Check 2: Hamiltonian symmetry H^T = H  
        if hasattr(circuit, 'quadratic_hamiltonian'):
            H = circuit.quadratic_hamiltonian
            is_symmetric = np.allclose(H.T, H)
            print(f"  [x] Hamiltonian is symmetric: {is_symmetric}")
            
        # Check 3: Kirchhoff constraint F @ K = 0
        if hasattr(circuit, 'F') and hasattr(circuit, 'K'):
            FK = circuit.F @ circuit.K
            constraint_satisfied = np.allclose(FK, 0, atol=1e-10)
            print(f"  [x] Kirchhoff constraint F@K=0: {constraint_satisfied}")
            
        # Check 4: Basis change orthogonality
        if hasattr(circuit, 'V'):
            V = circuit.V
            VTV = V.T @ V
            is_orthogonal = np.allclose(VTV, np.eye(V.shape[1]))
            print(f"  [x] Basis change is orthogonal: {is_orthogonal}")
                
    except Exception as e:
        print(f"Error in mathematical verifications: {e}")
    
    print(f"\n{'='*75}")
    print(f"END OF DEBUGGING FOR {name}")
    print(f"{'='*75}")
    
    return circuit


def debug_all_examples():
    """Debugs all examples from the notebooks."""
    
    # --- Linear Examples ---
    print("\n" + "="*80)
    print("LINEAR CIRCUIT EXAMPLES (Sections IV.A & IV.B)")
    print("="*80)
    
    # 1. LC oscillator
    print("\nRunning LC Oscillator Example...")
    C = Capacitor(value=0.1, unit='nF')
    L = Inductor(value=1, unit='nH')
    LC_topology = [(0,1,L), (0,1,C)]
    debug_circuit_steps(LC_topology, "LC Oscillator")
    
    # 2. Coupled LC oscillators
    print("\nRunning Coupled LC Oscillators Example...")
    C = Capacitor(value=0.1, unit='nF')
    Cg = Capacitor(value=0.2, unit='nF')
    L = Inductor(value=1, unit='nH')
    # Note: Topology based on Fig 3 description (Loop 1, Coupling, Loop 2)
    coupled_LC_topology = [(0,1,L), (1,2,Cg), (2,0,L), (0,1,C), (2,0,C)]
    debug_circuit_steps(coupled_LC_topology, "Coupled LC Oscillators")
    
    # --- Nonlinear Examples ---
    print("\n" + "="*80)
    print("NONLINEAR CIRCUIT EXAMPLES (Sections V.A & V.B)") 
    print("="*80)
    
    # 3. Fluxonium
    print("\nRunning Fluxonium Example...")
    C_J = Capacitor(value=0.1, unit='nF')
    J = Junction(value=1, unit='GHz', cap=C_J)
    L = Inductor(value=1, unit='nH')
    fluxonium_topology = [(0,1,J), (0,1,L)]
    debug_circuit_steps(fluxonium_topology, "Fluxonium")
    
    # 4. Singular Circuit (from Fig 5)
    print("\nRunning Singular Circuit (Fig 5) Example...")
    # Parameters from Section V.B
    C = Capacitor(value=0.1, unit='nF')
    C_J = Capacitor(value=0.1, unit='nF')
    J = Junction(value=1, unit='GHz', cap=C_J) 
    L = Inductor(value=1, unit='nH')
    
    # Topology: Inductor in series with C, and J (with parallel C_J)
    # Based on the paper's description of nodes
    singular_topology = [(0,1,J), (1,2,L), (2,0,C)]
    debug_circuit_steps(singular_topology, "Singular Circuit")


if __name__ == "__main__":
    """
    To use this script:
    1. Ensure QuantumSCC is installed/importable.
    2. Run: python quantum_scc_debug.py
    """
    
    print("QuantumSCC Debug Tool")
    
    # First, perform introspection of the class
    introspect_circuit_class()
    
    print("\nRunning all examples...")
    debug_all_examples()