#=
https://scikit-learn.org/stable/modules/clustering.html#overview-of-clustering-methods

NOTE: there is a predict method associated (but in general with unsupervised
there may not be so should be careful)

XXX: In  some case there's not even a  transform (e.g.  AffinityPropag)

XXX: add the predict logic macro

XXX: add inverse_transform

need to see, some unsupervised algos may have inverse  transform, some not
likewise some may have predict, some not. Some may have transform some not. So might need to code everything case by case, a bit annoying.

=#
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html#sklearn.cluster.AffinityPropagation
AffinityPropagation_ = SKCL.AffinityPropagation
@sk_trf mutable struct AffinityPropagation <: MLJBase.Unsupervised
end

# PREDICT AND FIT_PREDICT, no transform.

metadata_model(AffinityPropagation,
    input   = MLJBase.Table(MLJBase.Continuous),
    output  = MLJBase.Table(MLJBase.Continuous),
    weights = false,
    descr   = "Perform Affinity Propagation Clustering of data."
    )

# ============================================================================
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering
AgglomerativeClustering_ = SKCL.AgglomerativeClustering
@sk_trf mutable struct AgglomerativeClustering <: MLJBase.Unsupervised
end

# ONLY FIT_PREDICT

metadata_model(AgglomerativeClustering,
    input   = MLJBase.Table(MLJBase.Continuous),
    output  = MLJBase.Table(MLJBase.Continuous),
    weights = false,
    descr   = "Recursively merges the pair of clusters that minimally increases a given linkage distance."
    )

# ============================================================================
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html#sklearn.cluster.Birch
Birch_ = SKCL.Birch
@sk_trf mutable struct Birch <: MLJBase.Unsupervised
end

# PREDICT
# TRANSFORM
# NO INVERSE TRANSFORM

metadata_model(Birch,
    input   = MLJBase.Table(MLJBase.Continuous),
    output  = MLJBase.Table(MLJBase.Continuous),
    weights = false,
    descr   = "Memory-efficient, online-learning algorithm provided as an alternative to MiniBatchKMeans."
    )

# ============================================================================
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN
DBSCAN_ = SKCL.DBSCAN
@sk_trf mutable struct DBSCAN <: MLJBase.Unsupervised
end

# ONLY FIT_PREDICT
# NO TRANSFORM
# NO INVERSE TRANSFORM

metadata_model(DBSCAN,
    input   = MLJBase.Table(MLJBase.Continuous),
    output  = MLJBase.Table(MLJBase.Continuous),
    weights = false,
    descr   = "Density-Based Spatial Clustering of Applications with Noise. Finds core samples of high density and expands clusters from them. Good for data which contains clusters of similar density."
    )

# ============================================================================
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.FeatureAgglomeration.html#sklearn.cluster.FeatureAgglomeration
FeatureAgglomeration_ = SKCL.FeatureAgglomeration
@sk_trf mutable struct FeatureAgglomeration <: MLJBase.Unsupervised
end

# NO PREDICT
# TRANSFORM
# INVERSE TRANSFORM

metadata_model(FeatureAgglomeration,
    input   = MLJBase.Table(MLJBase.Continuous),
    output  = MLJBase.Table(MLJBase.Continuous),
    weights = false,
    descr   = "Similar to AgglomerativeClustering, but recursively merges features instead of samples."
    )

# ============================================================================
KMeans_ = SKCL.KMeans
@sk_trf mutable struct KMeans <: MLJBase.Unsupervised
    n_clusters::Int     = 8::(_ ≥ 1)
    n_init::Int         = 10::(_ ≥ 1)
    max_iter::Int       = 300::(_ ≥ 1)
    tol::Float64        = 1e-4::(_ > 0)
    verbose::Int        = 0::(_ ≥ 0)
    random_state::Any   = nothing
    copy_x::Bool        = true
    n_jobs::Option{Int} = nothing
    algorithm::String   = "auto"::(_ in ("auto", "full", "elkane"))
    # long
    precompute_distances::Union{Bool,String} = "auto"::(_ isa Bool || _ == "auto")
    init::Union{AbstractArray,String} = "k-means++"::(_ isa AbstractArray || _ in ("k-means++", "random"))
end
MLJBase.fitted_params(m::KMeans, f) = (
    cluster_centers = f.cluster_centers_,
    labels          = f.labels_,
    inertia         = f.inertia_,
    )
metadata_model(KMeans,
    input   = MLJBase.Table(MLJBase.Continuous),
    output  = MLJBase.Table(MLJBase.Continuous),
    weights = false,
    descr   = "K-Means algorithm: find K centroids corresponding to K clusters in the data."
    )

# ============================================================================
MiniBatchKMeans_ = SKCL.MiniBatchKMeans
@sk_trf mutable struct MiniBatchKMeans <: MLJBase.Unsupervised
end

# ============================================================================
MeanShift_ = SKCL.MeanShift
@sk_trf mutable struct MeanShift <: MLJBase.Unsupervised
end

# ============================================================================
OPTICS_ = SKCL.OPTICS
@sk_trf mutable struct OPTICS <: MLJBase.Unsupervised
end

# ============================================================================
SpectralClustering_ = SKCL.SpectralClustering
@sk_trf mutable struct SpectralClustering <: MLJBase.Unsupervised
end

# ============================================================================
SpectralBiclustering_ = SKCL.SpectralBiclustering
@sk_trf mutable struct SpectralBiclustering <: MLJBase.Unsupervised
end

# ============================================================================
SpectralCoclustering_ = SKCL.SpectralCoclustering
@sk_trf mutable struct SpectralCoclustering <: MLJBase.Unsupervised
end
