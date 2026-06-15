"""
Exploration Indices
=================================================
@author: Raghuvanshi A. and Vinayak
=================================================
Input format
------------
Each publication is represented as a whitespace-separated string
of author IDs/names.

The ego author must be included in every publication string.

Example:
'a1 a8 a11' means ego author 'a1' collaborated with 'a8' and 'a11' in one paper.
"""


def clustering_coefficients(X, ego_idx, mode="normalized"):
    import scipy.sparse as sp
    import numpy as np

    n_alpha = X.sum(axis=1).A1
    X_norm = sp.diags(1.0 / (n_alpha - 1)) @ X

    ### Adjacency Matrix  ##############################
    Admat = X.T @ X
    Admat.setdiag(0)
    Admat.data = (Admat.data > 0).astype(int)
    Admat.eliminate_zeros()

    ### Coupling Matrix ################################
    Cmat = X.T @ X_norm
    Cmat.setdiag(0)

    #### Degree ####################
    K = Admat.sum(axis=0).A1[ego_idx]

    ### Papers #####################
    S = Cmat.sum(axis=0).A1[ego_idx]

    ### Clustering Coefficients ###################
    #
    if mode == "normalized":
        WSCC = (Admat @ Admat @ Admat / K / (K - 1)).diagonal()[ego_idx]
        WCC = (Admat @ Admat @ Cmat / S / (K - 1)).diagonal()[ego_idx]
        return (WSCC, WCC, K, S)
    elif mode == "traingles":
        WSCC = (Admat @ Admat @ Admat).diagonal()[ego_idx]
        WCC = (Admat @ Admat @ Cmat).diagonal()[ego_idx]
        return (WSCC, WCC, K, S)


def vectorize(authors, egoauthor):
    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np

    vectorizer = CountVectorizer(token_pattern=r"(?u)[^\s]+")

    X = vectorizer.fit_transform(authors)
    feature = vectorizer.get_feature_names_out()
    # X.toarray()
    ego_idx = np.where(feature == egoauthor)[0][0]
    return (X, feature, ego_idx)


def arm_coauthors(X, feature, mode="explored_arm"):
    row_sum = X.sum(axis=1).A1  # team size

    ###=== ONLY WITH ONE CO-AUTHOR ==== ###
    row_mask = row_sum == 2

    X_arm = X[row_mask]

    arm_cols = X_arm.sum(axis=0).A1 > 0  # Authors of arms

    # ==================================================
    # ARM EXPLORED WITH ONE CO-AUTHOR
    # ==================================================
    if mode == "arm_explored":
        col_sum = X.sum(axis=0).A1  # number of papers by authors
        col_mask = col_sum == 1
        final_mask = col_mask & arm_cols  # Co-authors of arms with one paper
    elif mode == "arms_not_in_clique":
        rowC_mask = row_sum > 2
        X_clique = X[rowC_mask]
        clique_cols = X_clique.sum(axis=0).A1 > 0
        final_mask = arm_cols & (~clique_cols)
    # ================================================================
    # COLLECT CO-AUTHOR
    # ==============================================================
    selected_features = feature[final_mask]
    return selected_features


def update_X(X, feature, ego_idx, arms_not_in_clique):
    import numpy as np

    col_mask = ~np.isin(feature, list(arms_not_in_clique))

    X_updated = X[:, col_mask]
    X_updated = X_updated[X_updated.sum(axis=1).A1 >= 2]

    feature_updated = feature[col_mask]
    ego_idx_updated = np.sum(col_mask[:ego_idx])

    return (X_updated, feature_updated, ego_idx_updated)


def configuration(data, egoauthor, mode="static"):
    import numpy as np

    if not isinstance(egoauthor, str):
        raise TypeError("egoauthor must be a string")
    if mode not in {"static", "dynamic"}:
        raise ValueError("mode must be " "'static' or 'dynamic'")
    if mode == "dynamic":
        if len(data) != 2:
            raise ValueError("dynamic mode requires " "[t0, t1]")

    if mode == "static":
        ####=================================================================###
        ####========= DATA IS A LIST: CHANGE IF DATA IS STRING ==============###
        ####=================================================================###
        if isinstance(data, str):
            data = [data]

        ###=================================================================###
        ###================ COMPUTE AUTHORS ================================###
        ###=================================================================###

        authors = [x for x in data]

        ###==================================================================###
        ###================ VECTORIZE AUTHORS ===============================###
        ###==================================================================###

        X, feature, ego_idx = vectorize(authors, egoauthor)

        n_alpha = X.sum(axis=1).A1  # TEAM SIZE

        ####=================================================================###
        ####=============== ARM STATISTICS  =================================###
        ####=================================================================###

        arms_explored = arm_coauthors(X, feature, "arm_explored")

        all_arms = [
            next(au for au in s.split() if au != egoauthor)
            for s in data
            if len(s.split(" ")) == 2
        ]

        K_arm = len(arms_explored)
        A_arm = K_arm

        S_arm = len(all_arms)

        ###==================================================================###
        ###================ CLIQUE STATISTICS ====================================###
        ###==================================================================###

        mask_clique = n_alpha > 2

        X_clique = X[mask_clique]

        S_clique = np.shape(X_clique)[0]

        n_alpha_clique = np.array([x for x in n_alpha if x > 2])

        K_max_clique = np.sum(n_alpha_clique) - S_clique

        arms_not_in_clique = arm_coauthors(X, feature, "arms_not_in_clique")

        ###==================================================================###
        ###========== UPDATE X REMOVING COLUMN OF NON-REPEATED ARM ========###
        ###==================================================================###

        X_updated, feature_updated, ego_idx_updated = update_X(
            X, feature, ego_idx, arms_not_in_clique
        )

        ###==================================================================###
        ###================ COMPUTE CLUSTERING COFFS =========================###
        ###==================================================================###

        WSCC, WCC, K_clique, S_clique = clustering_coefficients(
            X_updated, ego_idx_updated, mode="normalized"
        )

        ###==================================================================###
        ###================ # CLIQUES +# ARMS IN CLIQUES=====================###
        ###==================================================================###

        SS = np.shape(X_updated)[0]

        ###==================================================================###
        ###================ DEGREE OF THE EGO ===============================###
        ###==================================================================###

        K = K_clique + K_arm

        ###==================================================================###
        ###================ CONF. VALUE FOR MAX EXPLORATION =================###
        ###==================================================================###

        WSCC0 = (
            (np.inner((n_alpha_clique - 2), (n_alpha_clique - 1)))
            / K_max_clique
            / (K_max_clique - 1)
        )
        WCC0 = (sum((n_alpha_clique - 2))) / SS / (K_max_clique - 1)

        return {
            "WSCC": WSCC,
            "WCC": WCC,
            "WSCC0": WSCC0,
            "WCC0": WCC0,
            "K": K,
            "K_arm": K_arm,
            "S_arm": S_arm,
            "A_arm": A_arm,
        }

    elif mode == "dynamic":
        ###=========================================== =====================###
        ###================ COMPUTE TEAMS AT t0 AND t1 =====================###
        ###=========================================== =====================###

        n_alphas_list = [[len(x.split(" ")) for x in data[i]] for i in range(len(data))]

        ###=================================================================###
        ###================ COMPUTE ARMS AT t0 AND t1=======================###
        ###=========================================== =====================###

        all_arms = [
            [
                next(au for au in s.split() if au != egoauthor)
                for s in data[i]
                if len(s.split(" ")) == 2
            ]
            for i in range(len(data))
        ]

        ###=========================================== =====================###
        ###================ COMPUTE AUTHORS AT t0 AND t1====================###
        ###=========================================== =====================###

        authors = [[x for x in data[i]] for i in range(len(data))]

        ###=========================================== =====================###
        ###================ ADD ALL CO-AUTHORS TILL t1 =====================###
        ###=========================================== =====================###

        authors.extend([[x for y in authors for x in y]])

        ###=================================================================###
        ###================ VECTORIZE ALL AUTHORS, I.E., ===================###
        ###==== AUTHORS AT t0, AUTHORS AT t1, AND AUTHORS TILL t1 ===========###
        ###=========================================== =====================###

        vectors = [vectorize(au, egoauthor) for au in authors]

        X_list, feature_list, ego_idx_list = zip(*vectors)

        ###=================================================================###
        ###================ UNEXPLORED ARMS TILL t1 ========================###
        ###=================================================================###

        arms_explored = arm_coauthors(X_list[-1], feature_list[-1], "arm_explored")
        # data
        arms_explored_t1 = list(set(arms_explored) - set(all_arms[0]))

        K_arm_t1 = len(arms_explored_t1)
        A_arm_t1 = K_arm_t1
        S_arm_t1 = len(all_arms[1])

        ###=================================================================###
        ###================ TEAM SIZE AT t0 AND t1 =========================###
        ###=================================================================###

        n_alphas_arrays = [np.array(x) for x in n_alphas_list]

        ###=================================================================###
        ###=============== CLIQUES AT t0 AND t1 ============================###
        ###=================================================================###

        # X_cliques =([X_list[i][X_list[i].sum(axis=1).A1>2] for i in
        #      range(len(X_list))]) ### THE CUMULATIVE X_list[-1] IS INCLUDED

        # S_cliques = [np.shape(X_clique)[0] for X_clique in X_cliques]

        n_alpha_cliques = [[x for x in n_alpha if x > 2] for n_alpha in n_alphas_arrays]

        # K_max_cliques = ([np.sum(n_alpha_cliques[i])-S_cliques[i] for i in
        #               range(len(S_cliques)-1)])

        ###=================================================================###
        ###===== IDENTIFY AUTHORS IN CLIQUES AT t0 AND t1 ==================###
        ###=================================================================###

        #   clique_authors =([ feature_list[i][(X_cliques[i].sum(axis=0).A1> 0)]
        #                    for i in range(len(feature_list))])

        # feature_list[-1][(X_cliques[-1].sum(axis=0).A1> 0)]

        arms_not_in_cliques = [
            arm_coauthors(X_list[i], feature_list[i], mode="arms_not_in_clique")
            for i in range(len(X_list))
        ]

        ###=================================================================###
        ###======== COMPUTE ARM STATISTICS AT t0 ======= ===================###
        ###=================================================================###

        vectors_updated = [
            update_X(
                X_list[i], feature_list[i], ego_idx_list[i], arms_not_in_cliques[i]
            )
            for i in range(len(X_list))
        ]
        X_updated, feature_updated, ego_idx_updated = zip(*vectors_updated)

        SS1 = np.shape(X_updated[2])[0]

        ###=================================================================###
        ###====== COMPUTE CLUSTING COFFS AT t0 AND TILL t1 =================###
        ###=================================================================###

        # clusterings = ([clustering_coefficients(X_updated[i], ego_idx_updated[i])
        #                for i in range(len(X_updated)) if i!=1])

        # WSCC_list, WCC_list, K_clique_list, S_clique_list=zip(*clusterings)

        (
            n_traingles,
            n_weighted_trainges,
            K_clique_t0,
            S_clique_t0,
        ) = clustering_coefficients(X_updated[0], ego_idx_updated[0], mode="traingles")
        WSCC_t1, WCC_t1, K_clique_t1, S_clique_t1 = clustering_coefficients(
            X_updated[2], ego_idx_updated[2], mode="normalized"
        )
        ###=================================================================###
        ###====== COMPUTE DEGREEs AT t0 AND TILL t1 =================###
        ###=================================================================###

        # K_t0=K_arm[0]+K_clique_list[0]

        K_t1 = K_arm_t1 + K_clique_t1

        ###=================================================================###
        ###==============  COMPUTE DELTAs AT t0 ============================###
        ###=================================================================###

        Delta_t0 = (
            n_traingles  # K_clique_list[0] * (K_clique_list[0] -1) * WSCC_list[0]
        )
        DeltaW_t0 = n_weighted_trainges  # SS0 * (K_clique_list[0] -1) * WCC_list[0]

        ###=================================================================###
        ###==== COMPUTE CLUSTERING COFFS FOR MAX EXPLORATION AT t1 =========###
        ###=================================================================###

        K_max_t1 = K_clique_t0 + sum(np.array(n_alpha_cliques[1]) - 1)
        #        data
        WSCC0_t1 = (
            (
                Delta_t0
                + (
                    np.inner(
                        np.array(n_alpha_cliques[1]) - 1,
                        np.array(n_alpha_cliques[1]) - 2,
                    )
                )
            )
            / K_max_t1
            / (K_max_t1 - 1)
        )
        WCC0_t1 = (
            (DeltaW_t0 + sum(np.array(n_alpha_cliques[1]) - 2)) / SS1 / (K_max_t1 - 1)
        )
        # SS=S_clique+n_arms_in_clique # S_clique+ number of arms inside clique
        return {
            "WSCC": WSCC_t1,
            "WCC": WCC_t1,
            "WSCC0": WSCC0_t1,
            "WCC0": WCC0_t1,
            "K": K_t1,
            "K_arm": K_arm_t1,
            "S_arm": S_arm_t1,
            "A_arm": A_arm_t1,
        }


# data
#from typing import List, Tuple


def exploration_index(data, egoauthor, mode="static") -> tuple[float, float]:
    """Compute the exploration index of an ego author's
    collaboration network.
    Parameters
    ----------
    data :
        list or list[list] Collaboration history.
    Static mode:
        List of publications.
    Dynamic mode:
        [past_window, new_window]
        Each publication must be a whitespace-separated string of
        author identifiers.
    Example:
        "a1 a4 a9"
    egoauthor :
        str Identifier of the focal (ego) author.
    mode :
        {"static", "dynamic"},
        default="static" Exploration mode.
        - "static": Compute exploration for one window.
        - "dynamic": Compute exploration from t0 → t1.
    Returns
    -------
    tuple(float, float) (E, E_weighted)

    Examples
    --------
    >>> exploration_index( ... ["a1 a2", "a1 a3 a4"], ... "a1" ... )"""
    ###=====================================================================###
    # ====================== COMPUTE CONFIGURATION VALUES ===================###
    ###=====================================================================###
    if mode == "static":
        configuration_values = configuration(data, egoauthor, mode="static")
    elif mode == "dynamic":
        configuration_values = configuration(data, egoauthor, mode="dynamic")
    #

    K = configuration_values["K"]
    S_arm = configuration_values["S_arm"]
    K_arm = configuration_values["K_arm"]
    A_arm = configuration_values["A_arm"]
    WSCC0 = configuration_values["WSCC0"]
    WCC0 = configuration_values["WCC0"]
    K_clique = K - K_arm
    WSCC = configuration_values["WSCC"]
    WCC = configuration_values["WCC"]

    if K:
        if S_arm:
            E_arm = (K_arm * A_arm) / K / S_arm
        else:
            E_arm = 0
        if WSCC0 < 1:
            E_clique = (K_clique * (1.0 - WSCC)) / K / (1.0 - WSCC0)
        else:
            E_clique = 0

        E_weighted_clique = (K_clique * (1.0 - WCC)) / K / (1.0 - WCC0)
        if WCC0 < 1:
            E_weighted_clique = (K_clique * (1.0 - WCC)) / K / (1.0 - WCC0)
        else:
            E_weighted_clique = 0

    E = E_arm + E_clique
    E_weighted = E_arm + E_weighted_clique

    return (E, E_weighted)
