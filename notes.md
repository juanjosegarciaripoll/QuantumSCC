# Structure of content

- elements.py ha copiado código de SQCircuit
- units.py también de SQCircuit

# Scheme of operations

- Definición del circuito (Clase Circuit)
- Circuit.__init__() hace todo (Lagrangiano, Hamiltoniano y cuantización)

Order of variables:

- At the beginning of the circuit description [fJ, FC, FL, qJ, qC, QL]
  in order of junction, capacitor, linear inductor, first flux then
  charges.
- After the Kirchhoff, the variables are as in Z (after Eq. (6))
  [f_compact, f_extended, Q]
- The number of compact variables is returned by Kirchhoff
- After omega_function(), with the "V" transformation, the vector separates into chi and w
- chi = [f_compact, f_extended, q_conjugadas], de "w" no se sabe
- K * V es la transformación que lleva del espacio de cómputo al espacio de laboratorio (variables de rama)
- Key ingredients:
  1) "Detect and simplify compact flux variables without dynamics"
     Transform the "K" matrix so that in the 2-form the dependent variables appear as zeros.

  2) Construct the 2-form separating independent and dependent variables in Omega and V.
