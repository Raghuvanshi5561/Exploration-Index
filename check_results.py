
"""

Example: Computing Exploration Index
====================================
=================================================
@author: Raghuvanshi A. and Vinayak
=================================================

"""

import Team_exploration as exp


# ==========================================================
# Example ego collaboration history
# ==========================================================

ego_author = "a1"

# ------------------------------------------
# Collaboration window t0
# ------------------------------------------
t0 = ["a1 a8 a11", "a1 a10 a6 a7 a2", "a1 a11", "a1 a3 a5"]


# ------------------------------------------
# New collaborations at t1
# ------------------------------------------
t1 = ["a1 a4 a5"]

# ------------------------------------------
# New collaborations at t2
# ------------------------------------------
t2 = ["a1 a5 a9"]


# ==========================================================
# 1. Static Exploration
# ==========================================================
# Compute exploration for a single collaboration window

E_static, E_weighted_static = (exp.exploration_index(t0, 
                               ego_author,mode="static"))


print("\nStatic Exploration")
print("------------------")
print(f"E = {E_static:.4f}")
print(f"E_weighted = {E_weighted_static:.4f}")


# ==========================================================
# 2. Dynamic Exploration: t0 → t1
# ==========================================================
# Compare previous history (t0)
# with newly added collaborations (t1)

dynamic_input_t1 = [
    t0,
    t1
]

E_dynamic_t1, E_weighted_t1 = (
    exp.exploration_index(
        dynamic_input_t1,
        ego_author,
        mode="dynamic"
    )
)

print("\nDynamic Exploration (t0 → t1)")
print("-------------------------------")
print(f"E = {E_dynamic_t1:.4f}")
print(f"E_weighted = {E_weighted_t1:.4f}")


# ==========================================================
# 3. Dynamic Exploration: t1 → t2
# ==========================================================
# Update cumulative history and
# evaluate the next incoming window





history_t1 = t0 + t1

dynamic_input_t2 = [
    history_t1,
    t2
]

E_dynamic_t2, E_weighted_t2 = (
    exp.exploration_index(
        dynamic_input_t2,
        ego_author,
        mode="dynamic"
    )
)

print("\nDynamic Exploration (t1 → t2)")
print("-------------------------------")
print(f"E = {E_dynamic_t2:.4f}")
print(f"E_weighted = {E_weighted_t2:.4f}")

def test_static():

    E, Ew = exp.exploration_index(
        ["a1 a2"],
        "a1"
    )

    assert E >= 0
    


papers = [
    "a1 a8 a11",
    "a1 a11",
    "a1 a3 a5"
]

E, Ew = exp.exploration_index(
    papers,
    "a1"
)

print(E, Ew)