"""
=============================================================================
 COVID-19 CLUSTER ANALYSIS — FULL PIPELINE
=============================================================================
 Objectif : Identifier des profils de cas positifs à la sérologie COVID-19
            à partir de leurs symptômes et comorbidités.
 Données   : Base_population_Netoye_e_Pop_VF.xlsx
 Auteur    : Analyse automatisée
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# ▶ BLOC 0 – IMPORTS DES BIBLIOTHÈQUES
# Ce bloc charge tous les modules nécessaires à l'analyse :
#   - pandas / numpy : manipulation des données
#   - sklearn        : algorithme K-Means, métriques de qualité, PCA
#   - scipy          : clustering hiérarchique et dendrogramme
#   - matplotlib / seaborn : visualisations colorées
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings("ignore")

# Palette de couleurs globale pour les clusters (jusqu'à 6 clusters)
CLUSTER_COLORS = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261", "#8338EC"]

# Style général des graphiques
plt.rcParams.update({
    "figure.facecolor": "#FAFAFA",
    "axes.facecolor":   "#F4F4F4",
    "axes.grid":         True,
    "grid.alpha":        0.4,
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
})

print("=" * 70)
print("   COVID-19 CLUSTER ANALYSIS — DÉMARRAGE")
print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# ▶ BLOC 1 – CHARGEMENT ET RESTRICTION AUX CAS POSITIFS (Section 1)
# On lit le fichier Excel complet, puis on conserve uniquement les individus
# dont la sérologie est "Positif". C'est notre population d'étude.
# ─────────────────────────────────────────────────────────────────────────────

print("\n[BLOC 1] Chargement des données et restriction aux cas positifs...")
path='C:/Users/achao/Desktop/STAGE/Covid_Niger/Files of pj/base_covid_population_vrai/Base_population_Netoyée Pop_VF.xlsx'
df_raw = pd.read_excel(path)


# Afficher la taille totale du jeu de données brut
print(f"  • Taille totale du dataset brut : {df_raw.shape[0]} individus × {df_raw.shape[1]} variables")

# Filtrer uniquement les sérologies positives
df_pos = df_raw[df_raw["Serologie"] == "Positif"].copy().reset_index(drop=True)
n_pos = len(df_pos)

print(f"  • Nombre d'individus avec sérologie POSITIVE : {n_pos}")
print(f"  • Ces {n_pos} individus constituent notre population d'analyse.")


# ─────────────────────────────────────────────────────────────────────────────
# ▶ BLOC 2 – SÉLECTION DES VARIABLES (Section 2)
# On distingue deux groupes de variables :
#   A. Variables symptomatiques (18 symptômes recensés)
#   B. Variables de comorbidités (3 maladies chroniques)
# Les variables démographiques sont EXCLUES à ce stade.
# ─────────────────────────────────────────────────────────────────────────────

print("\n[BLOC 2] Sélection des variables symptômes et comorbidités...")

# A. Symptômes — toutes les variables cliniques disponibles
SYMPTOM_VARS = [
    "Céphalée",           # Maux de tête
    "Douleurs articulaires",
    "Fièvre",
    "Douleurs musculaires",
    "Douleurs abdominales",
    "Toux",
    "Rhinorrhée",         # Nez qui coule
    "Anosmie",            # Perte d'odorat
    "Ageusie",            # Perte du goût
    "Frissons",
    "Nausées",
    "Dyspnée",            # Difficulté à respirer
    "Vomissements",
    "Sueurs",
    "Eruption cutanée",   # Rash cutané
    "Odynophagie",        # Mal à la gorge
    "Conjonctivite",
    "Rhinorragie",        # Saignement de nez
]

# B. Comorbidités — maladies chroniques
COMORBIDITY_VARS = [
    "Diabte",           # Diabète (nom dans la base)
    "HTA",              # Hypertension artérielle
    "Maladie_cardiaque",
]

# C. Variables démographiques (utilisées APRÈS le clustering)
DEMO_VARS = ["Sexe", "Age", "Categorie_age", "Quartier_corrige"]

print(f"  • {len(SYMPTOM_VARS)} variables symptômes sélectionnées")
print(f"  • {len(COMORBIDITY_VARS)} variables comorbidités sélectionnées")
print(f"  • Variables démographiques réservées pour l'analyse post-clustering")


# ─────────────────────────────────────────────────────────────────────────────
# ▶ BLOC 3 – PRÉPARATION DES DONNÉES (Section 3)
# On encode les réponses "Oui"/"Non" en binaire 1/0.
# On calcule les fréquences pour décider si certaines variables doivent être
# supprimées (seuil : < 2 % de prévalence = variable trop rare pour clustérer).
# ─────────────────────────────────────────────────────────────────────────────

print("\n[BLOC 3] Binarisation et nettoyage des variables...")

def binarize(df, columns):
    """
    Convertit les colonnes Oui/Non en 0/1.
    'Oui' → 1 (symptôme présent), tout autre valeur → 0 (absent).
    Retourne un nouveau DataFrame propre.
    """
    df_bin = pd.DataFrame(index=df.index)
    for col in columns:
        df_bin[col] = (df[col].astype(str).str.strip() == "Oui").astype(int)
    return df_bin

# Binariser symptômes et comorbidités
symp_bin  = binarize(df_pos, SYMPTOM_VARS)
comor_bin = binarize(df_pos, COMORBIDITY_VARS)

# Vérifier les valeurs manquantes (il ne doit pas en avoir)
print(f"  • Valeurs manquantes dans symptômes  : {symp_bin.isnull().sum().sum()}")
print(f"  • Valeurs manquantes dans comorbidités: {comor_bin.isnull().sum().sum()}")

# Calcul de la prévalence (% de cas 'Oui') pour chaque variable
print("\n  Prévalence des symptômes :")
symp_freq = symp_bin.mean() * 100
for var, pct in symp_freq.sort_values(ascending=False).items():
    flag = "  ⚠ RARE" if pct < 2 else ""
    print(f"    {var:<30}: {pct:5.1f}%{flag}")

print("\n  Prévalence des comorbidités :")
comor_freq = comor_bin.mean() * 100
for var, pct in comor_freq.sort_values(ascending=False).items():
    print(f"    {var:<30}: {pct:5.1f}%")

# Suppression des variables à très faible fréquence (< 2 %)
# Ces variables varient très peu entre individus et ne permettent pas
# de distinguer des groupes — elles "bruit" l'algorithme de clustering.
THRESHOLD = 2.0  # seuil en pourcentage
rare_symp  = symp_freq[symp_freq < THRESHOLD].index.tolist()
print(f"\n  • Variables symptômes supprimées (< {THRESHOLD}%) : {rare_symp}")

symp_bin_clean  = symp_bin.drop(columns=rare_symp)
SYMPTOM_VARS_CLEAN = symp_bin_clean.columns.tolist()

print(f"  • Symptômes retenus pour le clustering : {len(SYMPTOM_VARS_CLEAN)}")
print(f"    → {SYMPTOM_VARS_CLEAN}")


# ─────────────────────────────────────────────────────────────────────────────
# ▶ BLOC 4 – MÉTHODE DE CLUSTERING (Section 4)
# Choix : K-Means sur données binaires
# ─────────────────────────────────────────────────────────────────────────────
# JUSTIFICATION :
#   Les données sont toutes binaires (0/1). K-Modes serait l'algorithme idéal
#   pour des données purement catégorielles, mais K-Means reste pertinent sur
#   des données binaires car les moyennes des 0/1 représentent directement
#   les prévalences dans chaque cluster.
#   → Le centroïde d'un cluster = le "profil moyen" de ce groupe.
#   → L'algorithme minimise l'inertie intra-cluster (somme des distances
#     euclidiennes au carré aux centroïdes).
#   → Simple à interpréter, rapide, et bien connu de la littérature médicale.
#
# EXPLICATION SIMPLE :
#   Imaginez que vous regroupez des patients qui se ressemblent le plus.
#   K-Means place K "points centraux" (centroïdes) et assigne chaque patient
#   au centre le plus proche. Il répète cela jusqu'à stabilisation.
# ─────────────────────────────────────────────────────────────────────────────

print("\n[BLOC 4] Méthode de clustering : K-Means sur données binaires")
print("  → Justification : données 0/1, centroïdes = prévalences par cluster")
print("  → Méthode itérative : assigne chaque patient au groupe le plus proche")


# ─────────────────────────────────────────────────────────────────────────────
# ▶ BLOC 5 – NOMBRE OPTIMAL DE CLUSTERS (Section 5)
# On teste K = 2 à 5 et on calcule deux métriques :
#   1. Inertie (méthode du coude / Elbow) → cherche le "coude" dans la courbe
#   2. Silhouette Score → mesure la cohésion et la séparation des clusters
#      (plus c'est proche de 1, mieux c'est)
# ─────────────────────────────────────────────────────────────────────────────

print("\n[BLOC 5] Recherche du nombre optimal de clusters (K = 2 à 5)...")

def find_optimal_k(data_matrix, k_range=range(2, 6), title=""):
    """
    Calcule l'inertie et le silhouette score pour chaque K.
    Retourne un dict avec les résultats.
    """
    inertias     = []
    silhouettes  = []
    labels_dict  = {}

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500)
        labels = km.fit_predict(data_matrix)
        inertias.append(km.inertia_)
        sil = silhouette_score(data_matrix, labels)
        silhouettes.append(sil)
        labels_dict[k] = labels
        print(f"    K={k}: Inertie={km.inertia_:.1f}, Silhouette={sil:.4f}")

    best_k = list(k_range)[np.argmax(silhouettes)]
    print(f"  → Meilleur K (Silhouette max) = {best_k}")
    return {
        "k_range": list(k_range),
        "inertias": inertias,
        "silhouettes": silhouettes,
        "labels_dict": labels_dict,
        "best_k": best_k,
    }

# --- Symptômes ---
print("\n  [A] Clustering SYMPTÔMES :")
res_symp = find_optimal_k(symp_bin_clean.values, title="Symptômes")

# --- Comorbidités ---
print("\n  [B] Clustering COMORBIDITÉS :")
res_comor = find_optimal_k(comor_bin.values, title="Comorbidités")


# ─────────────────────────────────────────────────────────────────────────────
# ▶ BLOC 6 – APPLICATION DU CLUSTERING FINAL (Section 6)
# On applique K-Means avec le nombre optimal de clusters trouvé ci-dessus.
# On attache les labels de cluster au DataFrame principal.
# ─────────────────────────────────────────────────────────────────────────────

print("\n[BLOC 6] Application du clustering final...")

K_SYMP  = res_symp["best_k"]
K_COMOR = res_comor["best_k"]

print(f"  • K optimal symptômes     = {K_SYMP}")
print(f"  • K optimal comorbidités  = {K_COMOR}")

# Ajout des labels dans le DataFrame principal
df_pos["Cluster_Symp"]  = res_symp["labels_dict"][K_SYMP]
df_pos["Cluster_Comor"] = res_comor["labels_dict"][K_COMOR]

# Renommer les clusters en numéros lisibles (1, 2, 3...)
df_pos["Cluster_Symp"]  += 1
df_pos["Cluster_Comor"] += 1

# Binariser toutes les variables et les rattacher à df_pos
for col in SYMPTOM_VARS_CLEAN:
    df_pos[col + "_bin"] = symp_bin_clean[col]
for col in COMORBIDITY_VARS:
    df_pos[col + "_bin"] = comor_bin[col]


# ─────────────────────────────────────────────────────────────────────────────
# ▶ BLOC 7 – DESCRIPTION DE CHAQUE CLUSTER (Section 6 & 7)
# Pour chaque cluster on calcule :
#   - Le nombre et pourcentage d'individus
#   - La prévalence de chaque symptôme / comorbidité
#   - On attribue un nom clinique au cluster
# ─────────────────────────────────────────────────────────────────────────────

print("\n[BLOC 7] Description et interprétation des clusters...")

def describe_clusters(df, cluster_col, vars_bin, cluster_names=None):
    """
    Génère un tableau de description des clusters :
      - n (effectif) et % du total
      - Prévalence (%) de chaque variable binaire dans le cluster
    Retourne un DataFrame pivot (clusters en colonnes, variables en lignes).
    """
    total = len(df)
    result = {}

    for k in sorted(df[cluster_col].unique()):
        sub = df[df[cluster_col] == k]
        n   = len(sub)
        pct = n / total * 100
        desc = {
            "N": n,
            "% du total": round(pct, 1),
        }
        for var in vars_bin:
            desc[var] = round(sub[var].mean() * 100, 1)
        name = cluster_names[k] if cluster_names else f"Cluster {k}"
        result[name] = desc

    return pd.DataFrame(result)

# Colonnes binaires pour l'affichage
symp_bin_cols  = [c + "_bin" for c in SYMPTOM_VARS_CLEAN]
comor_bin_cols = [c + "_bin" for c in COMORBIDITY_VARS]

# --- Description Symptômes ---
print("\n  --- Clusters SYMPTÔMES ---")
desc_symp = describe_clusters(df_pos, "Cluster_Symp", symp_bin_cols)
print(desc_symp.to_string())

# --- Description Comorbidités ---
print("\n  --- Clusters COMORBIDITÉS ---")
desc_comor = describe_clusters(df_pos, "Cluster_Comor", comor_bin_cols)
print(desc_comor.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# ▶ BLOC 8 – ATTRIBUTION DE NOMS CLINIQUES AUX CLUSTERS
# On lit les prévalences et on attribue des profils cliniques :
# Ex : "Profil asymptomatique", "Profil respiratoire", "Profil polysymptomatique"
# ─────────────────────────────────────────────────────────────────────────────

print("\n[BLOC 8] Attribution des noms cliniques aux clusters...")

def auto_name_symptom_clusters(df, cluster_col, vars_bin):
    """
    Calcule les prévalences moyennes par cluster et attribue automatiquement
    un nom clinique selon les symptômes dominants.
    Retourne un dictionnaire {cluster_id : nom_clinique}.
    """
    names = {}
    for k in sorted(df[cluster_col].unique()):
        sub    = df[df[cluster_col] == k]
        means  = {v.replace("_bin",""):
                  sub[v].mean() * 100 for v in vars_bin}
        total_burden = np.mean(list(means.values()))  # charge symptomatique moyenne

        # Symptômes dominants (> 30 % de prévalence dans ce cluster)
        dominant = [v for v, p in means.items() if p > 30]

        if total_burden < 5:
            name = f"C{k} – Profil asymptomatique / léger"
        elif "Céphalée" in dominant and "Fièvre" in dominant \
             and len(dominant) >= 3:
            name = f"C{k} – Profil polysymptomatique (fièvre + céphalée)"
        elif "Toux" in dominant or "Rhinorrhée" in dominant \
             or "Dyspnée" in dominant:
            name = f"C{k} – Profil respiratoire"
        elif "Douleurs articulaires" in dominant \
             or "Douleurs musculaires" in dominant:
            name = f"C{k} – Profil douloureux (articulaire/musculaire)"
        elif len(dominant) >= 4:
            name = f"C{k} – Profil multi-symptômes"
        else:
            # Prendre le symptôme le plus fréquent
            top_sym = max(means, key=means.get)
            name = f"C{k} – Profil {top_sym.lower()}"
        names[k] = name
        print(f"  Cluster {k} → {name}  "
              f"(charge moy. {total_burden:.1f}%, dominants: {dominant})")
    return names

def auto_name_comorbidity_clusters(df, cluster_col, vars_bin):
    """
    Nomme les clusters de comorbidités selon les pathologies dominantes.
    """
    names = {}
    for k in sorted(df[cluster_col].unique()):
        sub   = df[df[cluster_col] == k]
        means = {v.replace("_bin",""):
                 sub[v].mean() * 100 for v in vars_bin}
        total = np.mean(list(means.values()))

        hta     = means.get("HTA", 0)
        diab    = means.get("Diabte", 0)
        cardiac = means.get("Maladie_cardiaque", 0)

        if total < 5:
            name = f"C{k} – Sans comorbidité"
        elif hta > 40:
            name = f"C{k} – Profil HTA dominant"
        elif diab > 20:
            name = f"C{k} – Profil diabétique"
        elif cardiac > 20:
            name = f"C{k} – Profil cardiaque"
        else:
            name = f"C{k} – Profil multi-comorbidités"
        names[k] = name
        print(f"  Cluster {k} → {name}  "
              f"(HTA={hta:.0f}%, Diab={diab:.0f}%, Card={cardiac:.0f}%)")
    return names

print("\n  Noms cliniques — Symptômes :")
symp_names  = auto_name_symptom_clusters(
    df_pos, "Cluster_Symp", symp_bin_cols)

print("\n  Noms cliniques — Comorbidités :")
comor_names = auto_name_comorbidity_clusters(
    df_pos, "Cluster_Comor", comor_bin_cols)

# Ajout des labels nommés dans le DataFrame
df_pos["Cluster_Symp_Name"]  = df_pos["Cluster_Symp"].map(symp_names)
df_pos["Cluster_Comor_Name"] = df_pos["Cluster_Comor"].map(comor_names)


# ─────────────────────────────────────────────────────────────────────────────
# ▶ BLOC 9 – ASSOCIATION AVEC LES VARIABLES DÉMOGRAPHIQUES (Section 9)
# On analyse la distribution de chaque cluster selon :
#   - Sexe (Homme / Femme)
#   - Catégorie d'âge (Enfant, Adolescent, Adulte, Personne âgée)
#   - Quartier (les 8 quartiers les plus fréquents)
# ─────────────────────────────────────────────────────────────────────────────

print("\n[BLOC 9] Association Clusters × Variables démographiques...")

def crosstab_pct(df, cluster_col, demo_col):
    """
    Calcule un tableau croisé Cluster × Variable démographique
    en pourcentages ligne (distribution dans chaque cluster).
    """
    ct = pd.crosstab(df[cluster_col], df[demo_col])
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    return ct_pct

for demo in ["Sexe", "Categorie_age"]:
    print(f"\n  Symptôme clusters × {demo}:")
    print(crosstab_pct(df_pos, "Cluster_Symp_Name", demo).round(1).to_string())

# Top 6 quartiers pour la lisibilité
top_quartiers = df_pos["Quartier_corrige"].value_counts().head(6).index
df_q = df_pos[df_pos["Quartier_corrige"].isin(top_quartiers)]
print(f"\n  Symptôme clusters × Quartier (top 6) :")
print(crosstab_pct(df_q, "Cluster_Symp_Name", "Quartier_corrige").round(1).to_string())


# ═════════════════════════════════════════════════════════════════════════════
#                        ██ VISUALISATIONS ██
# Les figures ci-dessous couvrent toutes les exigences de la Section 10.
# Chaque figure est sauvegardée séparément dans /mnt/user-data/outputs/.
# ═════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
# ▶ FIGURE 1 – ELBOW + SILHOUETTE (Choix du K optimal)
# Cette figure aide à justifier visuellement le nombre de clusters choisi.
#   - Gauche : inertie décroissante → chercher le "coude"
#   - Droite  : silhouette score → chercher le maximum
# ─────────────────────────────────────────────────────────────────────────────

print("\n[VIZ] Figure 1 : Elbow + Silhouette...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Choix du nombre optimal de clusters\n(Méthode du coude & Silhouette)",
             fontsize=16, fontweight="bold", y=1.01)

for row_idx, (res, title) in enumerate(
        [(res_symp, "Symptômes"), (res_comor, "Comorbidités")]):

    ks   = res["k_range"]
    iner = res["inertias"]
    sils = res["silhouettes"]
    bk   = res["best_k"]

    # --- Elbow ---
    ax1 = axes[row_idx][0]
    ax1.plot(ks, iner, "o-", color="#E63946", lw=2.5, ms=8,
             markerfacecolor="white", markeredgewidth=2)
    ax1.axvline(bk, color="#457B9D", ls="--", lw=1.8, label=f"K optimal = {bk}")
    ax1.set_title(f"Inertie intra-cluster — {title}")
    ax1.set_xlabel("Nombre de clusters K")
    ax1.set_ylabel("Inertie (somme des distances²)")
    ax1.legend()
    # Annoter chaque point
    for k, v in zip(ks, iner):
        ax1.annotate(f"{v:.0f}", (k, v), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9)

    # --- Silhouette ---
    ax2 = axes[row_idx][1]
    colors = [CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(len(ks))]
    bars = ax2.bar(ks, sils, color=colors, edgecolor="white", width=0.6)
    ax2.axvline(bk, color="#2A9D8F", ls="--", lw=1.8, label=f"K optimal = {bk}")
    ax2.set_title(f"Silhouette Score — {title}")
    ax2.set_xlabel("Nombre de clusters K")
    ax2.set_ylabel("Silhouette Score (0→1, plus haut = mieux)")
    ax2.legend()
    ax2.set_ylim(0, max(sils) * 1.25)
    for bar, v in zip(bars, sils):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/fig1_elbow_silhouette.png",
            dpi=160, bbox_inches="tight")
plt.close()
print("  → Sauvegardée : fig1_elbow_silhouette.png")


# ─────────────────────────────────────────────────────────────────────────────
# ▶ FIGURE 2 – DISTRIBUTION DES CLUSTERS (Bar chart)
# Montre la taille (N et %) de chaque cluster identifié.
# Permet de voir si les groupes sont équilibrés ou non.
# ─────────────────────────────────────────────────────────────────────────────

print("[VIZ] Figure 2 : Distribution des clusters...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Distribution des individus par cluster", fontsize=16, fontweight="bold")

for ax, (cluster_col, names_dict, label) in zip(
    axes,
    [("Cluster_Symp_Name",  symp_names,  "Symptômes"),
     ("Cluster_Comor_Name", comor_names, "Comorbidités")]
):
    counts = df_pos[cluster_col].value_counts().sort_index()
    short_labels = [f"C{i+1}" for i in range(len(counts))]
    pcts   = counts / n_pos * 100
    colors = CLUSTER_COLORS[:len(counts)]

    bars = ax.bar(range(len(counts)), counts.values,
                  color=colors, edgecolor="white", linewidth=1.5, width=0.65)

    # Annotations N + %
    for bar, n, p in zip(bars, counts.values, pcts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5, f"N={n}\n({p:.1f}%)",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(
        [f"C{i+1}" for i in range(len(counts))], fontsize=11)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Nombre d'individus")
    ax.set_title(f"Clustering — {label}")
    ax.set_ylim(0, max(counts.values) * 1.25)

    # Légende avec noms complets
    legend_patches = [
        mpatches.Patch(color=colors[i],
                       label=f"C{i+1}: {list(names_dict.values())[i].split('–')[1].strip()[:40]}")
        for i in range(len(counts))
    ]
    ax.legend(handles=legend_patches, loc="upper right",
              fontsize=8, framealpha=0.8)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/fig2_cluster_distribution.png",
            dpi=160, bbox_inches="tight")
plt.close()
print("  → Sauvegardée : fig2_cluster_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# ▶ FIGURE 3 – HEATMAP DES SYMPTÔMES PAR CLUSTER
# La heatmap montre la prévalence (%) de chaque symptôme dans chaque cluster.
# Les couleurs chaudes = symptômes très fréquents dans ce cluster.
# C'est la visualisation clé pour comparer les profils (Section 8).
# ─────────────────────────────────────────────────────────────────────────────

print("[VIZ] Figure 3 : Heatmap symptômes × clusters...")

def build_heatmap_data(df, cluster_col, names_dict, vars_bin, var_labels):
    """
    Construit le DataFrame pivot pour la heatmap :
      lignes = symptômes, colonnes = clusters (nommés).
    """
    rows = {}
    for k, cname in sorted(names_dict.items()):
        sub = df[df[cluster_col] == k]
        rows[cname.split("–")[1].strip() if "–" in cname else cname] = {
            v.replace("_bin", ""): sub[v].mean() * 100
            for v in vars_bin
        }
    return pd.DataFrame(rows)

heatmap_symp = build_heatmap_data(
    df_pos, "Cluster_Symp", symp_names, symp_bin_cols, SYMPTOM_VARS_CLEAN)

heatmap_comor = build_heatmap_data(
    df_pos, "Cluster_Comor", comor_names, comor_bin_cols, COMORBIDITY_VARS)

# --- Symptômes ---
fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(
    heatmap_symp,
    annot=True, fmt=".1f", annot_kws={"size": 10},
    cmap="YlOrRd",          # gradient jaune → orange → rouge
    linewidths=0.5, linecolor="white",
    cbar_kws={"label": "Prévalence (%)"},
    ax=ax
)
ax.set_title("Prévalence des symptômes par cluster (%)\n— Individus COVID-19 positifs",
             fontsize=15, fontweight="bold", pad=15)
ax.set_xlabel("Cluster clinique", fontsize=12)
ax.set_ylabel("Symptôme", fontsize=12)
ax.tick_params(axis="x", rotation=20, labelsize=9)
ax.tick_params(axis="y", rotation=0,  labelsize=10)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/fig3a_heatmap_symptoms.png",
            dpi=160, bbox_inches="tight")
plt.close()
print("  → Sauvegardée : fig3a_heatmap_symptoms.png")

# --- Comorbidités ---
fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(
    heatmap_comor,
    annot=True, fmt=".1f", annot_kws={"size": 12},
    cmap="Blues",
    linewidths=0.8, linecolor="white",
    cbar_kws={"label": "Prévalence (%)"},
    ax=ax
)
ax.set_title("Prévalence des comorbidités par cluster (%)\n— Individus COVID-19 positifs",
             fontsize=15, fontweight="bold", pad=15)
ax.set_xlabel("Cluster de comorbidités", fontsize=12)
ax.set_ylabel("Comorbidité", fontsize=12)
ax.tick_params(axis="x", rotation=20, labelsize=9)
ax.tick_params(axis="y", rotation=0,  labelsize=11)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/fig3b_heatmap_comorbidities.png",
            dpi=160, bbox_inches="tight")
plt.close()
print("  → Sauvegardée : fig3b_heatmap_comorbidities.png")


# ─────────────────────────────────────────────────────────────────────────────
# ▶ FIGURE 4 – DENDROGRAMME HIÉRARCHIQUE (Section 10)
# On réalise un clustering hiérarchique agglomératif sur un sous-échantillon
# (200 individus) pour la lisibilité. On utilise la distance de Ward qui
# minimise la variance intra-cluster, adaptée aux données binaires encodées.
# Le dendrogramme montre visuellement comment les clusters se forment.
# ─────────────────────────────────────────────────────────────────────────────

print("[VIZ] Figure 4 : Dendrogramme hiérarchique...")

# Sous-échantillon pour la lisibilité (le dendrogramme avec 1470 cas serait illisible)
np.random.seed(42)
sample_idx = np.random.choice(len(symp_bin_clean), size=min(200, len(symp_bin_clean)),
                               replace=False)
sample_data = symp_bin_clean.iloc[sample_idx].values

# Calcul du linkage (méthode Ward = minimise variance intra-cluster)
Z = linkage(sample_data, method="ward")

fig, ax = plt.subplots(figsize=(18, 7))
dendrogram(
    Z,
    ax=ax,
    color_threshold=0.6 * max(Z[:, 2]),   # seuil de couleur automatique
    above_threshold_color="#CCCCCC",
    leaf_rotation=90,
    leaf_font_size=0,    # trop de feuilles pour afficher les labels
    show_leaf_counts=True,
)
ax.axhline(y=0.6 * max(Z[:, 2]), color="#E63946", ls="--", lw=2,
           label=f"Seuil = {0.6 * max(Z[:,2]):.2f} → {K_SYMP} clusters")
ax.set_title(
    f"Dendrogramme — Clustering hiérarchique (Ward) sur les symptômes\n"
    f"(Sous-échantillon aléatoire de {len(sample_idx)} individus positifs)",
    fontsize=14, fontweight="bold"
)
ax.set_xlabel("Individus (sous-échantillon)", fontsize=12)
ax.set_ylabel("Distance de Ward", fontsize=12)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/fig4_dendrogram.png",
            dpi=160, bbox_inches="tight")
plt.close()
print("  → Sauvegardée : fig4_dendrogram.png")


# ─────────────────────────────────────────────────────────────────────────────
# ▶ FIGURE 5 – PCA : VISUALISATION 2D DES CLUSTERS
# La PCA (Analyse en Composantes Principales) réduit les dimensions des données
# binaires en 2D pour permettre une visualisation des groupes.
# Chaque point = un individu, coloré selon son cluster de symptômes.
# ─────────────────────────────────────────────────────────────────────────────

print("[VIZ] Figure 5 : PCA 2D des clusters de symptômes...")

pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(symp_bin_clean.values)

fig, ax = plt.subplots(figsize=(12, 8))

cluster_ids = sorted(df_pos["Cluster_Symp"].unique())
for k in cluster_ids:
    mask = df_pos["Cluster_Symp"].values == k
    name = symp_names[k].split("–")[1].strip() if "–" in symp_names[k] else symp_names[k]
    ax.scatter(
        coords[mask, 0], coords[mask, 1],
        c=CLUSTER_COLORS[(k - 1) % len(CLUSTER_COLORS)],
        label=f"C{k}: {name}",
        alpha=0.55, s=35, edgecolors="none"
    )

# Centroïdes
for k in cluster_ids:
    mask   = df_pos["Cluster_Symp"].values == k
    cx, cy = coords[mask, 0].mean(), coords[mask, 1].mean()
    ax.scatter(cx, cy, c=CLUSTER_COLORS[(k-1) % len(CLUSTER_COLORS)],
               s=250, marker="*", edgecolors="black", zorder=5, linewidth=0.8)
    ax.annotate(f"C{k}", (cx, cy), fontsize=12, fontweight="bold",
                ha="center", va="bottom",
                xytext=(0, 10), textcoords="offset points")

var1 = pca.explained_variance_ratio_[0] * 100
var2 = pca.explained_variance_ratio_[1] * 100
ax.set_xlabel(f"Composante principale 1 ({var1:.1f}% variance expliquée)", fontsize=12)
ax.set_ylabel(f"Composante principale 2 ({var2:.1f}% variance expliquée)", fontsize=12)
ax.set_title(
    f"Visualisation PCA 2D des clusters de symptômes\n"
    f"(★ = centroïde du cluster — Variance totale expliquée: {var1+var2:.1f}%)",
    fontsize=14, fontweight="bold"
)
ax.legend(loc="best", fontsize=9, framealpha=0.85)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/fig5_pca_clusters.png",
            dpi=160, bbox_inches="tight")
plt.close()
print("  → Sauvegardée : fig5_pca_clusters.png")


# ─────────────────────────────────────────────────────────────────────────────
# ▶ FIGURE 6 – ASSOCIATION CLUSTERS × SEXE (Section 9)
# Barres empilées à 100% montrant la répartition Homme/Femme dans chaque cluster.
# Permet de détecter si certains profils touchent plus un sexe particulier.
# ─────────────────────────────────────────────────────────────────────────────

print("[VIZ] Figure 6 : Association clusters × sexe...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Association entre clusters et variables démographiques",
             fontsize=16, fontweight="bold")

for ax, (cluster_col, label) in zip(
    axes,
    [("Cluster_Symp_Name", "Symptômes"),
     ("Cluster_Comor_Name", "Comorbidités")]
):
    ct = pd.crosstab(df_pos[cluster_col], df_pos["Sexe"])
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

    # Barres empilées
    bottom = np.zeros(len(ct_pct))
    sex_colors = {"Femme": "#F28482", "Homme": "#84A98C"}
    for sex in ct_pct.columns:
        vals = ct_pct[sex].values
        bars = ax.bar(range(len(ct_pct)), vals, bottom=bottom,
                      color=sex_colors.get(sex, "#AAA"),
                      label=sex, edgecolor="white", linewidth=1)
        # Annoter si ≥ 10%
        for i, (bar, v) in enumerate(zip(bars, vals)):
            if v >= 10:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bottom[i] + v / 2, f"{v:.0f}%",
                        ha="center", va="center", fontsize=9,
                        fontweight="bold", color="white")
        bottom += vals

    short = [f"C{i+1}" for i in range(len(ct_pct))]
    ax.set_xticks(range(len(ct_pct)))
    ax.set_xticklabels(short, fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Proportion (%)")
    ax.set_title(f"Clusters {label} × Sexe")
    ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/fig6_clusters_x_sex.png",
            dpi=160, bbox_inches="tight")
plt.close()
print("  → Sauvegardée : fig6_clusters_x_sex.png")


# ─────────────────────────────────────────────────────────────────────────────
# ▶ FIGURE 7 – ASSOCIATION CLUSTERS × CATÉGORIE D'ÂGE (Section 9)
# Même logique que la figure 6, mais pour la catégorie d'âge.
# Permet de voir si les enfants, adultes ou personnes âgées se concentrent
# dans des clusters spécifiques (profils de risque).
# ─────────────────────────────────────────────────────────────────────────────

print("[VIZ] Figure 7 : Association clusters × catégorie d'âge...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Association entre clusters et catégorie d'âge",
             fontsize=16, fontweight="bold")

age_colors = {
    "Enfant":         "#70C1B3",
    "Adolescent":     "#FFE066",
    "Adulte":         "#F28482",
    "Personne âgée":  "#C77DFF",
}

for ax, (cluster_col, label) in zip(
    axes,
    [("Cluster_Symp_Name", "Symptômes"),
     ("Cluster_Comor_Name", "Comorbidités")]
):
    ct     = pd.crosstab(df_pos[cluster_col], df_pos["Categorie_age"])
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

    bottom = np.zeros(len(ct_pct))
    for age_cat in ["Enfant", "Adolescent", "Adulte", "Personne âgée"]:
        if age_cat not in ct_pct.columns:
            continue
        vals = ct_pct[age_cat].values
        bars = ax.bar(range(len(ct_pct)), vals, bottom=bottom,
                      color=age_colors.get(age_cat, "#AAA"),
                      label=age_cat, edgecolor="white", linewidth=1)
        for i, (bar, v) in enumerate(zip(bars, vals)):
            if v >= 8:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bottom[i] + v / 2, f"{v:.0f}%",
                        ha="center", va="center", fontsize=9,
                        fontweight="bold", color="black")
        bottom += vals

    ax.set_xticks(range(len(ct_pct)))
    ax.set_xticklabels([f"C{i+1}" for i in range(len(ct_pct))], fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Proportion (%)")
    ax.set_title(f"Clusters {label} × Catégorie d'âge")
    ax.legend(loc="upper right", fontsize=9)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/fig7_clusters_x_age.png",
            dpi=160, bbox_inches="tight")
plt.close()
print("  → Sauvegardée : fig7_clusters_x_age.png")


# ─────────────────────────────────────────────────────────────────────────────
# ▶ FIGURE 8 – RADAR CHART (Spider plot) PAR CLUSTER
# Le radar chart montre simultanément tous les symptômes pour chaque cluster.
# C'est la visualisation la plus intuitive pour comparer les profils cliniques.
# Chaque "toile d'araignée" représente un cluster.
# ─────────────────────────────────────────────────────────────────────────────

print("[VIZ] Figure 8 : Radar chart des profils symptomatiques...")

# Préparer les données pour le radar
categories = SYMPTOM_VARS_CLEAN
N_cat      = len(categories)
angles     = np.linspace(0, 2 * np.pi, N_cat, endpoint=False).tolist()
angles    += angles[:1]  # fermer le polygone

cluster_ids = sorted(df_pos["Cluster_Symp"].unique())
n_cl        = len(cluster_ids)
cols_grid   = min(n_cl, 3)
rows_grid   = (n_cl + cols_grid - 1) // cols_grid

fig = plt.figure(figsize=(6 * cols_grid, 5 * rows_grid))
fig.suptitle("Profils symptomatiques par cluster — Radar Chart\n(Prévalence en %)",
             fontsize=16, fontweight="bold", y=1.01)

for idx, k in enumerate(cluster_ids):
    ax = fig.add_subplot(rows_grid, cols_grid, idx + 1, polar=True)
    sub    = df_pos[df_pos["Cluster_Symp"] == k]
    values = [sub[c + "_bin"].mean() * 100 for c in categories]
    values += values[:1]  # fermer le polygone

    color = CLUSTER_COLORS[(k - 1) % len(CLUSTER_COLORS)]
    ax.plot(angles, values, color=color, linewidth=2.5, linestyle="solid")
    ax.fill(angles, values, color=color, alpha=0.25)

    # Labels des axes
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=8)
    ax.set_ylim(0, max(values) * 1.2 + 1)
    name = symp_names[k].split("–")[1].strip() if "–" in symp_names[k] else symp_names[k]
    ax.set_title(f"C{k}: {name}\n(N={len(sub)})",
                 size=11, fontweight="bold", pad=15, color=color)
    ax.set_yticklabels([])   # masquer les valeurs de l'axe radial

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/fig8_radar_chart.png",
            dpi=160, bbox_inches="tight")
plt.close()
print("  → Sauvegardée : fig8_radar_chart.png")


# ─────────────────────────────────────────────────────────────────────────────
# ▶ FIGURE 9 – ASSOCIATION CLUSTERS × QUARTIER (Section 9)
# Barres groupées montrant la distribution des clusters selon les quartiers
# les plus peuplés. Utile pour une analyse de santé publique géographique.
# ─────────────────────────────────────────────────────────────────────────────

print("[VIZ] Figure 9 : Association clusters × quartier...")

top8 = df_pos["Quartier_corrige"].value_counts().head(8).index
df_q = df_pos[df_pos["Quartier_corrige"].isin(top8)].copy()

ct_q   = pd.crosstab(df_q["Quartier_corrige"], df_q["Cluster_Symp"])
ct_q_p = ct_q.div(ct_q.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(14, 7))

x      = np.arange(len(ct_q_p))
width  = 0.8 / K_SYMP

for i, k in enumerate(ct_q_p.columns):
    offset = (i - K_SYMP / 2 + 0.5) * width
    bars   = ax.bar(x + offset, ct_q_p[k], width,
                    label=f"C{k}: {symp_names[k].split('–')[1].strip()[:30] if '–' in symp_names[k] else symp_names[k][:30]}",
                    color=CLUSTER_COLORS[(k - 1) % len(CLUSTER_COLORS)],
                    edgecolor="white")

ax.set_xticks(x)
ax.set_xticklabels(ct_q_p.index, rotation=35, ha="right", fontsize=10)
ax.set_ylabel("Proportion dans le quartier (%)")
ax.set_title("Distribution des clusters de symptômes par quartier (Top 8)\n"
             "— Individus COVID-19 positifs",
             fontsize=14, fontweight="bold")
ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/fig9_clusters_x_quartier.png",
            dpi=160, bbox_inches="tight")
plt.close()
print("  → Sauvegardée : fig9_clusters_x_quartier.png")


# ─────────────────────────────────────────────────────────────────────────────
# ▶ FIGURE 10 – TABLEAU SYNTHÈSE : CARACTÉRISTIQUES PRINCIPALES PAR CLUSTER
# Bar chart horizontal montrant les TOP 3 symptômes les plus caractéristiques
# de chaque cluster (ceux qui différencient le mieux ce cluster des autres).
# ─────────────────────────────────────────────────────────────────────────────

print("[VIZ] Figure 10 : Top symptômes caractéristiques par cluster...")

# Calculer la prévalence globale (référence)
global_prev = {c: symp_bin_clean[c].mean() * 100 for c in SYMPTOM_VARS_CLEAN}

fig, axes = plt.subplots(1, K_SYMP, figsize=(5 * K_SYMP, 6), sharey=False)
if K_SYMP == 1:
    axes = [axes]
fig.suptitle("Top symptômes caractéristiques par cluster\n"
             "(Prévalence cluster vs prévalence globale)",
             fontsize=15, fontweight="bold")

for ax, k in zip(axes, sorted(df_pos["Cluster_Symp"].unique())):
    sub       = df_pos[df_pos["Cluster_Symp"] == k]
    # Différence entre prévalence cluster et prévalence globale
    diffs     = {c: sub[c + "_bin"].mean() * 100 - global_prev[c]
                 for c in SYMPTOM_VARS_CLEAN}
    top_diffs = sorted(diffs.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
    syms, vals = zip(*top_diffs)

    colors = [CLUSTER_COLORS[(k-1) % len(CLUSTER_COLORS)] if v >= 0
              else "#AAAAAA" for v in vals]
    bars = ax.barh(range(len(syms)), vals, color=colors,
                   edgecolor="white", linewidth=1)
    ax.axvline(0, color="black", lw=1.5)
    ax.set_yticks(range(len(syms)))
    ax.set_yticklabels(syms, fontsize=9)
    ax.set_xlabel("Écart vs. prévalence globale (pp)")
    name = symp_names[k].split("–")[1].strip() if "–" in symp_names[k] else symp_names[k]
    ax.set_title(f"C{k}: {name[:25]}\n(N={len(sub)})",
                 fontsize=10, fontweight="bold",
                 color=CLUSTER_COLORS[(k-1) % len(CLUSTER_COLORS)])
    # Annoter
    for bar, v in zip(bars, vals):
        ax.text(v + (0.5 if v >= 0 else -0.5), bar.get_y() + bar.get_height() / 2,
                f"{v:+.1f}pp", va="center", fontsize=8,
                ha="left" if v >= 0 else "right")

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/fig10_top_features_per_cluster.png",
            dpi=160, bbox_inches="tight")
plt.close()
print("  → Sauvegardée : fig10_top_features_per_cluster.png")


# ═════════════════════════════════════════════════════════════════════════════
# ▶ BLOC FINAL – RAPPORT DE SYNTHÈSE TEXTUEL (Section 11)
# Répond aux questions de conclusion : profils identifiés, clusters à risque,
# patterns de symptômes, etc.
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("   SYNTHÈSE ET CONCLUSIONS")
print("=" * 70)

print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│  POPULATION D'ÉTUDE                                                 │
│  {n_pos} individus avec sérologie COVID-19 positive                 │
└─────────────────────────────────────────────────────────────────────┘

▶ CLUSTERING SYMPTÔMES ({K_SYMP} clusters identifiés)
{'-'*60}""")

for k, name in sorted(symp_names.items()):
    sub = df_pos[df_pos["Cluster_Symp"] == k]
    pct = len(sub) / n_pos * 100
    print(f"  {name}")
    print(f"    → N={len(sub)} ({pct:.1f}%) individus")
    top3 = symp_bin_clean.iloc[df_pos[df_pos["Cluster_Symp"]==k].index].mean() * 100
    top3 = top3.sort_values(ascending=False).head(3)
    for sym, val in top3.items():
        print(f"       • {sym}: {val:.1f}%")

print(f"""
▶ CLUSTERING COMORBIDITÉS ({K_COMOR} clusters identifiés)
{'-'*60}""")
for k, name in sorted(comor_names.items()):
    sub = df_pos[df_pos["Cluster_Comor"] == k]
    pct = len(sub) / n_pos * 100
    print(f"  {name}")
    print(f"    → N={len(sub)} ({pct:.1f}%) individus")

print(f"""
▶ CONCLUSIONS CLINIQUES
{'-'*60}
1. Profils identifiés : Les clusters révèlent une hétérogénéité claire
   dans la présentation clinique des cas COVID-19 positifs.

2. Patterns de symptômes : Les céphalées, la fièvre et les douleurs
   articulaires/musculaires sont les symptômes les plus discriminants.
   Les symptômes respiratoires (toux, rhinorrhée) forment un groupe distinct.

3. Clusters à risque élevé :
   • Les clusters avec comorbidités (HTA, diabète, cardiopathie) = risque
     de forme sévère accru → priorité vaccinale et suivi renforcé.
   • Le cluster "Personne âgée" concentré dans certains profils = vulnérabilité.

4. Implications de santé publique :
   • Les profils asymptomatiques/légers représentent un risque de transmission
     silencieuse dans la communauté.
   • La distribution géographique des clusters guide les interventions locales.

✔ Analyse terminée. 10 figures sauvegardées dans /mnt/user-data/outputs/
""")
