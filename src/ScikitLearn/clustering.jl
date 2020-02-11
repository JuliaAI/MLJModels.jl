#=
https://scikit-learn.org/stable/modules/clustering.html#overview-of-clustering-methods
=#

# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html#sklearn.cluster.AffinityPropagation
AffinityPropagation_ = SKCL.AffinityPropagation
@sk_uns mutable struct AffinityPropagation <: MMI.Unsupervised
    damping::Float64      = 0.5::(0.5 ≤ _ ≤ 1)
    max_iter::Int         = 200::(_ ≥ 1)
    convergence_iter::Int = 15::(_ ≥ 1)
    copy::Bool            = true
    preference::Any       = nothing
    affinity::String      = "euclidean"::(_ in ("euclidean", "precomputed"))
    verbose::Bool         = false
end
@sku_predict AffinityPropagation

function MMI.fitted_params(m::AffinityPropagation, f)
    nc   = length(f.cluster_centers_indices_)
    catv = MMI.categorical(1:nc)
    return (
        cluster_centers_indices = f.cluster_centers_indices_,
        cluster_centers         = f.cluster_centers_,
        labels                  = nc == 0 ? nothing : catv[f.labels_ .+ 1],
        affinity_matrix         = f.affinity_matrix_)
end
metadata_model(AffinityPropagation,
    input   = Table(Continuous),
    target  = AbstractVector{Multiclass},
    # no transform so no output
    weights = false,
    descr   = "Perform Affinity Propagation Clustering of data."
    )

# ============================================================================
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering
AgglomerativeClustering_ = SKCL.AgglomerativeClustering
@sk_uns mutable struct AgglomerativeClustering <: MMI.Unsupervised
    n_clusters::Int     = 2::(_ ≥ 1)
    affinity::String    = "euclidean"::(_ in ("euclidean", "l1", "l2", "manhattan", "cosine", "precomputed"))
    memory::Any         = nothing
    connectivity::Any   = nothing
    compute_full_tree::Union{String,Bool} = "auto"::(_ isa Bool || _ == "auto")
    linkage::String     = "ward"::(_ in ("ward", "complete", "average", "single"))
    distance_threshold::Option{Float64}   = nothing::(_ === nothing || _ > 0)
end
function MMI.fitted_params(m::AgglomerativeClustering, f)
    nc   = f.n_clusters_
    catv = MMI.categorical(1:nc)
    return (
        n_clusters = f.n_clusters_,
        labels     = catv[f.labels_ .+ 1],
        n_leaves   = f.n_leaves_,
        n_connected_components = f.n_connected_components_,
        children   = f.children_)
end
metadata_model(AgglomerativeClustering,
    input   = Table(Continuous),
    # no predict nor transform so no target nor output
    weights = false,
    descr   = "Recursively merges the pair of clusters that minimally increases a given linkage distance. Note: there is no `predict` or `transform` method
    associated with it; instead, inspect the `fitted_params`."
    )

# ============================================================================
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html#sklearn.cluster.Birch
Birch_ = SKCL.Birch
@sk_uns mutable struct Birch <: MMI.Unsupervised
    threshold::Float64    = 0.5::(_ > 0)
    branching_factor::Int = 50::(_ > 0)
    n_clusters::Int       = 3::(_ > 0)
    compute_labels::Bool  = true
    copy::Bool            = true
end
@sku_predict Birch
@sku_transform Birch

function MMI.fitted_params(m::Birch, f)
    nc   = m.n_clusters
    catv = MMI.categorical(1:nc)
    return (
        root               = f.root_,
        dummy_leaf         = f.dummy_leaf_,
        subcluster_centers = f.subcluster_centers_,
        subcluster_labels  = f.subcluster_labels_,
        labels             = catv[f.labels_ .+ 1])
end
metadata_model(Birch,
    input   = Table(Continuous),
    target  = AbstractVector{Multiclass},
    output  = Table(Continuous),
    weights = false,
    descr   = "Memory-efficient, online-learning algorithm provided as an alternative to MiniBatchKMeans. Note: noisy samples are given the label -1."
    )

# ============================================================================
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN
DBSCAN_ = SKCL.DBSCAN
@sk_uns mutable struct DBSCAN <: MMI.Unsupervised
    eps::Float64        = 0.5::(_ > 0)
    min_samples::Int    = 5::(_ > 1)
    metric::String      = "euclidean"::(_ in ("euclidean", "precomputed"))
    algorithm::String   = "auto"::(_ in ("auto", "ball_tree", "kd_tree", "brute"))
    leaf_size::Int      = 30::(_ > 1)
    p::Option{Float64}  = nothing
    n_jobs::Option{Int} = nothing
end
function MMI.fitted_params(m::DBSCAN, f)
    nc   = length(f.core_sample_indices_)
    catv = MMI.categorical([-1, (1:nc)...])
    return (
        core_sample_indices = f.core_sample_indices_,
        components          = f.components_,
        labels              = catv[f.labels_ .+ 2])
end
metadata_model(DBSCAN,
    input   = Table(Continuous),
    weights = false,
    descr   = "Density-Based Spatial Clustering of Applications with Noise. Finds core samples of high density and expands clusters from them. Good for data which contains clusters of similar density."
    )

# ============================================================================
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.FeatureAgglomeration.html#sklearn.cluster.FeatureAgglomeration
FeatureAgglomeration_ = SKCL.FeatureAgglomeration
@sk_uns mutable struct FeatureAgglomeration <: MMI.Unsupervised
    n_clusters::Int        = 2::(_ > 0)
    memory::Any            = nothing
    connectivity::Any      = nothing
    # XXX unclear how to pass a proper callable here; just passing mean = nok
    # pooling_func::Function = mean
    affinity::Any          = "euclidean"::(_ isa Function || _ in ("euclidean", "l1", "l2", "manhattan", "cosine",  "precomputed"))
    compute_full_tree::Union{String,Bool} = "auto"::(_ isa Bool || _ == "auto")
    linkage::String        = "ward"::(_ in ("ward", "complete", "average", "single"))
    distance_threshold::Option{Float64}   = nothing
end
@sku_transform FeatureAgglomeration
@sku_inverse_transform FeatureAgglomeration

function MMI.fitted_params(m::FeatureAgglomeration, f)
    nc   = m.n_clusters
    catv = MMI.categorical(1:nc)
    return (
        n_clusters = f.n_clusters_,
        labels     = catv[f.labels_ .+ 1],
        n_leaves   = f.n_leaves_,
        n_connected_components = f.n_connected_components_,
        children   = f.children_,
        distances  = m.distance_threshold === nothing ? nothing : f.distances_)
end
metadata_model(FeatureAgglomeration,
    input   = Table(Continuous),
    output  = Table(Continuous),
    weights = false,
    descr   = "Similar to AgglomerativeClustering, but recursively merges features instead of samples."
    )

# ============================================================================
KMeans_ = SKCL.KMeans
@sk_uns mutable struct KMeans <: MMI.Unsupervised
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
    init::Union{AbstractArray,String}        = "k-means++"::(_ isa AbstractArray || _ in ("k-means++", "random"))
end
@sku_transform KMeans
@sku_predict KMeans

function MMI.fitted_params(m::KMeans, f)
    nc   = m.n_clusters
    catv = MMI.categorical(1:nc)
    return (
        cluster_centers = f.cluster_centers_,
        labels          = catv[f.labels_ .+ 1],
        inertia         = f.inertia_)
end
metadata_model(KMeans,
    input   = Table(Continuous),
    target  = AbstractVector{Multiclass},
    output  = Table(Continuous),
    weights = false,
    descr   = "K-Means algorithm: find K centroids corresponding to K clusters in the data."
    )

# ============================================================================
MiniBatchKMeans_ = SKCL.MiniBatchKMeans
@sk_uns mutable struct MiniBatchKMeans <: MMI.Unsupervised
    n_clusters::Int         = 8::(_ ≥ 1)
    max_iter::Int           = 100::(_ > 1)
    batch_size::Int         = 100::(_ > 1)
    verbose::Int            = 0
    compute_labels::Bool    = true
    random_state::Any       = nothing
    tol::Float64            = 0.0::(_ ≥ 0)
    max_no_improvement::Int = 10::(_ > 1)
    init_size::Option{Int}  = nothing
    n_init::Int             = 3::(_ > 0)
    init::Union{AbstractArray,String} = "k-means++"::(_ isa AbstractArray || _ in ("k-means++", "random"))
    reassignment_ratio::Float64       = 0.01::(_ > 0)
end
@sku_predict MiniBatchKMeans
@sku_transform MiniBatchKMeans

function MMI.fitted_params(m::MiniBatchKMeans, f)
    nc   = m.n_clusters
    catv = MMI.categorical(1:nc)
    return (
        cluster_centers = f.cluster_centers_,
        labels          = catv[f.labels_ .+ 1],
        inertia         = f.inertia_)
end
metadata_model(MiniBatchKMeans,
    input   = Table(Continuous),
    target  = AbstractVector{Multiclass},
    output  = Table(Continuous),
    weights = false,
    descr   = "Mini-Batch K-Means clustering."
    )

# ============================================================================
MeanShift_ = SKCL.MeanShift
@sk_uns mutable struct MeanShift <: MMI.Unsupervised
    bandwidth::Option{Float64}   = nothing
    seeds::Option{AbstractArray} = nothing
    bin_seeding::Bool            = false
    min_bin_freq::Int            = 1::(_ ≥ 1)
    cluster_all::Bool            = true
    n_jobs::Option{Int}          = nothing
    # max_iter::Int                = 300::(_ > 1)
end
@sku_predict MeanShift

function MMI.fitted_params(m::MeanShift, f)
    nc   = size(f.cluster_centers_, 1)
    catv = MMI.categorical(1:nc)
    return (
        cluster_centers = f.cluster_centers_,
        labels          = catv[f.labels_ .+ 1])
end
metadata_model(MeanShift,
    input   = Table(Continuous),
    target  = AbstractVector{Multiclass},
    weights = false,
    descr   = "Mean shift clustering using a flat kernel. Mean shift clustering aims to discover \"blobs\" in a smooth density of samples. It is a centroid-based algorithm, which works by updating candidates for centroids to be the mean of the points within a given region. These candidates are then filtered in a post-processing stage to eliminate near-duplicates to form the final set of centroids."
    )

# ============================================================================
OPTICS_ = SKCL.OPTICS
@sk_uns mutable struct OPTICS <: MMI.Unsupervised
    min_samples::Union{Float64,Int} = 5::((_ isa Int && _ > 1) || 0 < _ < 1)
    max_eps::Float64       = Inf
    metric::String         = "minkowski"::(_ in ("precomputed", "cityblock", "cosine", "euclidean", "l1", "l2", "manhattan", "braycurtis", "canberra", "chebyshev", "correlation", "dice", "hamming", "jaccard", "kulsinski", "mahalanobis", "minkowski", "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener", "sokalsneath", "sqeuclidean", "yule"))
    p::Int                 = 2::(_ ≥ 1)
    cluster_method::String = "xi"
    eps::Option{Float64}   = nothing
    xi::Float64            = 0.05::(0 < _ < 1)
    predecessor_correction::Bool = true
    min_cluster_size::Union{Nothing,Float64,Int} = nothing::(_ === nothing || (_ isa Int && _ > 1) || (_ isa Float64 && 0 < _ < 1))
    algorithm::String      = "auto"::(_ in ("auto", "ball_tree", "kd_tree", "brute"))
    leaf_size::Int         = 30::(_ > 1)
    n_jobs::Option{Int}    = nothing
end
function MMI.fitted_params(m::OPTICS, f)
    nc   = size(f.cluster_hierarchy_, 1)
    catv = MMI.categorical([-1, (1:nc)...])
    return (
        labels            = catv[f.labels_ .+ 2],
        reachability      = f.reachability_,
        ordering          = f.ordering_,
        core_distances    = f.core_distances_,
        predecessor       = f.predecessor_,
        cluster_hierarchy = f.cluster_hierarchy_)
end
metadata_model(OPTICS,
    input   = Table(Continuous),
    weights = false,
    descr   = "OPTICS (Ordering Points To Identify the Clustering Structure), closely related to DBSCAN, finds core sample of high density and expands clusters from them. Unlike DBSCAN, keeps cluster hierarchy for a variable neighborhood radius. Better suited for usage on large datasets than the current sklearn implementation of DBSCAN."
    )

# ============================================================================
SpectralClustering_ = SKCL.SpectralClustering
@sk_uns mutable struct SpectralClustering <: MMI.Unsupervised
    n_clusters::Int      = 8::(_ ≥ 1)
    eigen_solver::Option{String} = nothing::(_ === nothing || _ in ("arpack", "lobpcg", "amg"))
#    n_components::Option{Int}    = nothing::(_ === nothing || _ ≥ 1)
    random_state::Any     = nothing
    n_init::Int           = 10::(_ ≥ 1)
    gamma::Float64        = 1.0::(_ > 0)
    affinity::String      = "rbf"::(_ in ("nearest_neighbors", "rbf", "precomputed", "precomputed_nearest_neighbors"))
    n_neighbors::Int      = 10::(_ > 0)
    eigen_tol::Float64    = 0.0::(_ ≥ 0)
    assign_labels::String = "kmeans"::(_ in ("kmeans", "discretize"))
    n_jobs::Option{Int}   = nothing
end
function MMI.fitted_params(m::SpectralClustering, f)
    nc   = m.n_clusters
    catv = MMI.categorical(1:nc)
    return (
        labels          = catv[f.labels_ .+ 1],
        affinity_matrix = f.affinity_matrix_)
end
metadata_model(SpectralClustering,
    input   = Table(Continuous),
    weights = false,
    descr   = "Apply clustering to a projection of the normalized Laplacian.
    In practice Spectral Clustering is very useful when the structure of the individual clusters is highly non-convex or more generally when a measure of the center and spread of the cluster is not a suitable description of the complete cluster. For instance when clusters are nested circles on the 2D plane."
    )

# NOTE: the two models below are weird, not bothering with them for now
# # ============================================================================
# SpectralBiclustering_ = SKCL.SpectralBiclustering
# @sk_uns mutable struct SpectralBiclustering <: MMI.Unsupervised
# end
#
# # ============================================================================
# SpectralCoclustering_ = SKCL.SpectralCoclustering
# @sk_uns mutable struct SpectralCoclustering <: MMI.Unsupervised
# end
