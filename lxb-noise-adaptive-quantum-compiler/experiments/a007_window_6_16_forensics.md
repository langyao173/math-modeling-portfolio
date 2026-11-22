# A007 Window 6:16 Forensics

- Qubits: [0, 1, 3]
- Gates: ['cx', 'tdg', 'cx', 't', 'cx', 'tdg', 't', 'cx', 't', 'cx']
- Old CX: 5
- DAG closed: False
- Dependency cone: 0:35 q=[0, 1, 2, 3, 4, 5]
- H-free: True
- CNOT-only: False
- Clifford-only: False
- Clifford+T: True
- Permutation x diagonal phase: True
- GF(2) matrix rows: ['001', '011', '100']
- GF(2) delta rank: 1
- Unitary hash: `c140d2151147419f3092a636cbfdb1e1b1310f913339b2c635def49d8e8edd94`
- Current routed candidate: noise_adaptive_l1_balanced metrics={'cx': 18, 'depth': 25, 'elr': -0.27710686331225304, 'route_overhead': 3}
- Window physical placement: {'0': 1, '1': 4, '3': 0}
- Candidate replacements: 0
- Best candidate CX: None

Interpretation: The 6:16 interval is not dependency-cone closed; lower bounds on the fixed slice can be optimistic.

## Gate List

- 0: {'text': 'CNOT(q(1), q(3))', 'gate': 'cx', 'qubits': [1, 3]}
- 1: {'text': 'T**-1(q(3))', 'gate': 'tdg', 'qubits': [3]}
- 2: {'text': 'CNOT(q(0), q(3))', 'gate': 'cx', 'qubits': [0, 3]}
- 3: {'text': 'T(q(3))', 'gate': 't', 'qubits': [3]}
- 4: {'text': 'CNOT(q(1), q(3))', 'gate': 'cx', 'qubits': [1, 3]}
- 5: {'text': 'T**-1(q(3))', 'gate': 'tdg', 'qubits': [3]}
- 6: {'text': 'T(q(1))', 'gate': 't', 'qubits': [1]}
- 7: {'text': 'CNOT(q(0), q(3))', 'gate': 'cx', 'qubits': [0, 3]}
- 8: {'text': 'T(q(3))', 'gate': 't', 'qubits': [3]}
- 9: {'text': 'CNOT(q(0), q(1))', 'gate': 'cx', 'qubits': [0, 1]}

## Truth Table / Phase

- 000 -> 000 phase=[1.0, 0.0]
- 001 -> 001 phase=[1.0, -0.0]
- 010 -> 010 phase=[0.707107, 0.707107]
- 011 -> 011 phase=[0.707107, 0.707107]
- 100 -> 110 phase=[1.0, 0.0]
- 101 -> 111 phase=[1.0, -0.0]
- 110 -> 100 phase=[0.707107, -0.707107]
- 111 -> 101 phase=[-0.707107, 0.707107]
