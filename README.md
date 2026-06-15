
# Team Exploration

**Exploration** is a Python package for quantifying **ego-network team exploration dynamics** in scientific collaboration systems.

The package computes team exploration indices from co-authorship structures by integrating:

- **Arm exploration** (new/independent collaborations),
- **Clique exploration** (collaborations within larger teams),
- **Weighted and unweighted clustering structures**, and
- **Dynamic temporal exploration** across collaboration windows.

This framework is particularly useful for studying:

- Scientific collaboration networks
- Team exploration dynamics
- Ego-centric network evolution

---

## Installation

### Local installation (editable mode)

Clone/download the repository and navigate to the project folder:

```bash
cd exploration_project
````

Install the package:

```bash
pip install -e .
```

---

## Dependencies

The package requires:

* numpy
* scipy
* scikit-learn

These are installed automatically during package installation.

---

## Input Format

Each publication should be represented as a **whitespace-separated string of authors**.

Example:

```python
"a1 a8 a11"
```

This means:

* `a1` = ego author
* `a8`, `a11` = collaborators in one publication

A collaboration history is represented as a list:

```python
papers = [
    "a1 a8 a11",
    "a1 a11",
    "a1 a3 a5"
]
```

---

# Example Usage

## 1. Static Exploration

Compute exploration within a single collaboration window.

```python
import exploration

ego_author = "a1"

papers = [
    "a1 a8 a11",
    "a1 a10 a6 a7 a2",
    "a1 a11",
    "a1 a3 a5"
]

E, E_weighted = exploration.exploration_index(
    papers,
    ego_author,
    mode="static"
)

print(E)
print(E_weighted)
```

---

## 2. Dynamic Exploration

Compute exploration from a historical window (`t0`) to a new collaboration window (`t1`).

```python
import exploration

ego_author = "a1"

t0 = [
    "a1 a8 a11",
    "a1 a10 a6 a7 a2",
    "a1 a11",
    "a1 a3 a5"
]

t1 = [
    "a1 a4 a5"
]

data = [t0, t1]

E, E_weighted = exploration.exploration_index(
    data,
    ego_author,
    mode="dynamic"
)

print(E)
print(E_weighted)
```

---

## 3. Rolling Dynamic Update

Update cumulative history and evaluate exploration in a new time window.

```python
import exploration

ego_author = "a1"

t0 = [
    "a1 a8 a11",
    "a1 a10 a6 a7 a2",
    "a1 a11",
    "a1 a3 a5"
]

t1 = [
    "a1 a4 a5"
]

t2 = [
    "a1 a5 a9"
]

# cumulative history till t1
history_t1 = t0 + t1

data = [history_t1, t2]

E, E_weighted = exploration.exploration_index(
    data,
    ego_author,
    mode="dynamic"
)

print(E)
print(E_weighted)
```

---

## Methodological Notes

The package operationalizes exploration through:

### Arm Exploration

Captures collaborations formed outside established clique structures.

### Clique Exploration

Measures novelty and recombination potential within larger collaborative teams.

### Weighted Exploration

Accounts for collaboration intensity through weighted clustering statistics.

### Dynamic Exploration

Tracks how incoming collaborations reshape prior ego-network structures over time.

---

## Repository Structure

```text
exploration_project/
│
├── exploration/
│   ├── __init__.py
│   └── exploration.py
│
├── examples/
│   └── example.py
│
├── README.md
├── pyproject.toml
└── LICENSE
```

---

## Citation

If you use this package in academic work, please cite the associated methodological publication (forthcoming).

---

## Author

**Adarsh Raghuvanshi and Vinayak**

CSIR–National Institute of Science Communication and Policy Research (CSIR-NIScPR)

New Delhi, India

```
```
