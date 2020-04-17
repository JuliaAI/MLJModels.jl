module TestTransformer

using Test, MLJModels
using Tables, CategoricalArrays, Random
using ScientificTypes
using StatsBase

import MLJBase

#### FEATURE SELECTOR ####

@testset "Feat Selector" begin
    N = 100
    X = (Zn   = rand(N),
         Crim = rand(N),
         x3   = categorical(rand("YN", N)),
         x4   = categorical(rand("YN", N)))

    namesX   = Tables.schema(X).names |> collect
    selector = FeatureSelector()
    f,       = MLJBase.fit(selector, 1, X)

    @test f == namesX

    Xt = MLJBase.transform(selector, f, MLJBase.selectrows(X, 1:2))

    @test Set(Tables.schema(Xt).names) == Set(namesX)
    @test length(Xt.Zn) == 2

    selector = FeatureSelector(features=[:Zn, :Crim])
    f,       = MLJBase.fit(selector, 1, X)

    @test MLJBase.transform(selector, f, MLJBase.selectrows(X, 1:2)) ==
            MLJBase.select(X, 1:2, [:Zn, :Crim])

    infos = MLJBase.info_dict(selector)
    @test infos[:input_scitype]  == MLJBase.Table(MLJBase.Scientific)
    @test infos[:output_scitype] == MLJBase.Table(MLJBase.Scientific)
end


#  To be added with FeatureSelectorRule X = (n1=["a", "b", "a"], n2=["g", "g", "g"], n3=[7, 8, 9],
#               n4 =UInt8[3,5,10],  o1=[4.5, 3.6, 4.0], )
# MLJBase.schema(X)
# Xc = coerce(X,  :n1=>Multiclass, :n2=>Multiclass)

# t = Discretizer(features=[:o1, :n3, :n2, :n1])
# @test Xt.features == [:o1, :n3, :n2, :n1]
# @test Xt.is_ordinal == [true, false, false, false]
# @test Xt.A == [512 1 1 1; 1 2 1 2; 256 3 1 1]


#### UNIVARIATE DISCRETIZATION ####

@testset "U-Discr" begin
    # TODO: move this test to MLJBase:
    # test helper function:
    v = collect("qwertyuiopasdfghjklzxcvbnm1")
    X = reshape(v, (3, 9))
    Xcat = categorical(X)
    Acat = Xcat[:, 1:4] # cat vector with unseen levels
    element = Acat[1]
    @test MLJBase.transform(element, X) == Xcat
    @test MLJBase.transform(element, X[5]) == Xcat[5]

    v = randn(10000)
    t = UnivariateDiscretizer(n_classes=100);
    result, = MLJBase.fit(t, 1, v)
    w = MLJBase.transform(t, result, v)
    bad_values = filter(v - MLJBase.inverse_transform(t, result, w)) do x
        abs(x) > 0.05
    end
    @test length(bad_values)/length(v) < 0.06

    # scalars:
    @test MLJBase.transform(t, result, v[42]) == w[42]
    r =  MLJBase.inverse_transform(t, result, w)[43]
    @test MLJBase.inverse_transform(t, result, w[43]) ≈ r

    # test of permitted abuses of argument:
    @test MLJBase.inverse_transform(t, result, get(w[43])) ≈ r
    @test MLJBase.inverse_transform(t, result, map(get, w)) ≈
        MLJBase.inverse_transform(t, result, w)

    # all transformed vectors should have an identical pool (determined in
    # call to fit):
    v2 = v[1:3]
    w2 = MLJBase.transform(t, result, v2)
    @test levels(w2) == levels(w)

    #### UNIVARIATE STANDARDIZER ####

    stand = UnivariateStandardizer()
    f,    = MLJBase.fit(stand, 1, [0, 2, 4])

    @test round.(Int, MLJBase.transform(stand, f, [0,4,8])) == [-1.0,1.0,3.0]
    @test round.(Int, MLJBase.inverse_transform(stand, f, [-1, 1, 3])) == [0, 4, 8]

    infos = MLJBase.info_dict(stand)
end

#### STANDARDIZER ####

@testset "Standardizer" begin
    N = 5
    rand_char = rand("abcefgh", N)
    while length(unique(rand_char)) < 2
        rand_char = rand("abcefgh", N)
    end
    X = (OverallQual  = rand(UInt8, N),
         GrLivArea    = rand(N),
         Neighborhood = categorical(rand_char, ordered=true),
         x1stFlrSF    = sample(1:10, N, replace=false),
         TotalBsmtSF  = rand(N))

    # introduce a field of type `Char`:
    x1 = categorical(map(Char, (X.OverallQual |> collect)))

    X = (x1=x1, x2=X[2], x3=X[3], x4=X[4], x5=X[5])

    stand = Standardizer()
    f,    = MLJBase.fit(stand, 1, X)
    Xnew  = MLJBase.transform(stand, f, X)

    @test Xnew[1] == X[1]
    @test MLJBase.std(Xnew[2]) ≈ 1.0
    @test Xnew[3] == X[3]
    @test Xnew[4] == X[4]
    @test MLJBase.std(Xnew[5]) ≈ 1.0

    stand.features = [:x1, :x5]
    f,   = MLJBase.fit(stand, 1, X)
    Xnew = MLJBase.transform(stand, f, X)

    @test issubset(Set(keys(f)), Set(Tables.schema(X).names[[5,]]))

    Xt = MLJBase.transform(stand, f, X)

    @test Xnew[1] == X[1]
    @test Xnew[2] == X[2]
    @test Xnew[3] == X[3]
    @test Xnew[4] == X[4]
    @test MLJBase.std(Xnew[5]) ≈ 1.0

    # test on ignoring a feature, even if it's listed in the `features`
    stand.ignore = true
    f,   = MLJBase.fit(stand, 1, X)
    Xnew = MLJBase.transform(stand, f, X)

    @test issubset(Set(keys(f)), Set(Tables.schema(X).names[[2,]]))

    Xt = MLJBase.transform(stand, f, X)

    @test Xnew[1] == X[1]
    @test MLJBase.std(Xnew[2]) ≈ 1.0
    @test Xnew[3] == X[3]
    @test Xnew[4] == X[4]
    @test Xnew[5] == X[5]

    stand = Standardizer(features=[:x1, :mickey_mouse])
    @test_logs(
        (:warn, r"Some specified"),
        (:warn, r"No features left"),
        MLJBase.fit(stand, 1, X)
    )

    stand.ignore = true
    @test_logs (:warn, r"Some specified") MLJBase.fit(stand, 1, X)

    @test_throws ArgumentError Standardizer(ignore=true)

    stand = Standardizer(features=[:x3, :x4], count=true, ordered_factor=true)
    f,   = MLJBase.fit(stand, 1, X)
    Xnew = MLJBase.transform(stand, f, X)
    @test issubset(Set(keys(f)), Set(Tables.schema(X).names[3:4,]))

    Xt = MLJBase.transform(stand, f, X)

    @test Xnew[1] == X[1]
    @test Xnew[2] == X[2]
    @test elscitype(X[3]) <: OrderedFactor
    @test elscitype(Xnew[3]) <: Continuous
    @test MLJBase.std(Xnew[3]) ≈ 1.0
    @test elscitype(X[4]) == Count
    @test elscitype(Xnew[4]) <: Continuous
    @test MLJBase.std(Xnew[4]) ≈ 1.0
    @test Xnew[5] == X[5]

    stand = Standardizer(features= x-> x == (:x2))
    f,    = MLJBase.fit(stand, 1, X)
    Xnew  = MLJBase.transform(stand, f, X)

    @test Xnew[1] == X[1]
    @test MLJBase.std(Xnew[2]) ≈ 1.0
    @test Xnew[3] == X[3]
    @test Xnew[4] == X[4]
    @test Xnew[5] == X[5]

    infos = MLJBase.info_dict(stand)

    @test infos[:name] == "Standardizer"
    @test infos[:input_scitype] == MLJBase.Table(MLJBase.Scientific)
    @test infos[:output_scitype] == MLJBase.Table(MLJBase.Scientific)
end

#### UNIVARIATE BOX COX TRANSFORMER ####

@testset "U-boxcox" begin
    # create skewed non-negative vector with a zero value:
    Random.seed!(1551)
    v = abs.(randn(1000))
    v = v .- minimum(v)

    t  = UnivariateBoxCoxTransformer(shift=true)
    f, = MLJBase.fit(t, 2, v)

    e = v - MLJBase.inverse_transform(t, f, MLJBase.transform(t, f, v))
    @test sum(abs, e) <= 5000*eps()

    infos = MLJBase.info_dict(t)

    @test infos[:name] == "UnivariateBoxCoxTransformer"
    @test infos[:input_scitype] == AbstractVector{MLJBase.Continuous}
    @test infos[:output_scitype] == AbstractVector{MLJBase.Continuous}
end

#### ONE HOT ENCODER ####

@testset "One-Hot" begin
    X = (name   = categorical(["Ben", "John", "Mary", "John"], ordered=true),
         height = [1.85, 1.67, 1.5, 1.67],
         favourite_number = categorical([7, 5, 10, 5]),
         age    = [23, 23, 14, 23])

    t  = OneHotEncoder()
    f, _, report = @test_logs((:info, r"Spawning 3"),
                    (:info, r"Spawning 3"), MLJBase.fit(t, 1, X))

    Xt = MLJBase.transform(t, f, X)

    @test Xt.name__John == float.([false, true, false, true])
    @test Xt.height == X.height
    @test Xt.favourite_number__10 == float.([false, false, true, false])
    @test Xt.age == X.age
    @test MLJBase.schema(Xt).names == (:name__Ben, :name__John, :name__Mary,
                               :height, :favourite_number__5,
                               :favourite_number__7, :favourite_number__10,
                               :age)

    @test report.new_features == collect(MLJBase.schema(Xt).names)

    # test that *entire* pool of categoricals is used in fit, including
    # unseen levels:
    f, = @test_logs((:info, r"Spawning 3"), (:info, r"Spawning 3"),
                          MLJBase.fit(t, 1, MLJBase.selectrows(X,1:2)))
    Xtsmall = MLJBase.transform(t, f, X)
    @test Xt == Xtsmall

    # test that transform can be applied to subset of the data:
    @test MLJBase.transform(t, f, MLJBase.selectcols(X, [:name, :age])) ==
        MLJBase.selectcols(MLJBase.transform(t, f, X),
                           [:name__Ben, :name__John, :name__Mary, :age])

    # test ignore
    t = OneHotEncoder(features=[:name,], ignore=true)
    f, = MLJBase.fit(t, 0, X)
    Xt = MLJBase.transform(t, f, X)

    # test exclusion of ordered factors:
    t  = OneHotEncoder(ordered_factor=false)
    f, = MLJBase.fit(t, 0, X)
    Xt = MLJBase.transform(t, f, X)
    @test keys(Xt) == (:name, :height, :favourite_number__5,
                       :favourite_number__7, :favourite_number__10, :age)

    @test :name in Tables.schema(Xt).names
    @test :favourite_number__5 in Tables.schema(Xt).names
    @test MLJBase.schema(Xt).scitypes == (OrderedFactor{3}, Continuous,
                                          Continuous, Continuous,
                                          Continuous, Count)

    # test that one may not add new columns:
    X = (name = categorical(["Ben", "John", "Mary", "John"], ordered=true),
         height     = [1.85, 1.67, 1.5, 1.67],
         favourite_number = categorical([7, 5, 10, 5]),
         age        = [23, 23, 14, 23],
         gender     = categorical(['M', 'M', 'F', 'M']))
    @test_throws Exception MLJBase.transform(t, f, X)

    infos = MLJBase.info_dict(t)

    @test infos[:name] == "OneHotEncoder"
    @test infos[:input_scitype] == MLJBase.Table(MLJBase.Scientific)
    @test infos[:output_scitype] == MLJBase.Table(MLJBase.Scientific)
end

#### FILL IMPUTER ####

@testset "Imputer" begin
    X = (
        x = [missing,ones(10)...],
        y = [missing,ones(10)...],
        z = [missing,ones(10)...]
        )

    imp = FillImputer()
    f,  = MLJBase.fit(imp, 1, X)
    Xt  = MLJBase.transform(imp, f, X)
    @test all(.!ismissing.(Xt.x))
    @test Xt.x isa Vector{Float64} # no missing
    @test all(Xt.x .== 1.0)

    imp = FillImputer(features=[:x,:y])
    f,  = MLJBase.fit(imp, 1, X)
    @test_throws ErrorException MLJBase.transform(imp, f, X) # in X there's :z which we haven't trained on

    X = (x = categorical([missing, missing, missing, missing, "Old", "Young", "Middle", "Young",
                          "Old", "Young", "Middle", "Young"]), )

    mode_ = MLJBase.mode(["Old", "Young", "Middle", "Young", "Old", "Young", "Middle", "Young"])

    imp = FillImputer()
    f,  = MLJBase.fit(imp, 1, X)
    Xt  = MLJBase.transform(imp, f, X)
    @test all(.!ismissing.(Xt.x))
    @test all(Xt.x[ismissing.(X.x)] .== mode_)

    X  = (x = [missing, missing, 1, 1, 1, 1, 1, 5], )
    f, = MLJBase.fit(imp, 1, X)
    Xt = MLJBase.transform(imp, f, X)
    @test Xt.x == [1, 1, 1, 1, 1, 1, 1, 5]

    X = (x = categorical([missing, missing, missing, missing, "Old", "Young", "Middle", "Young",
                          "Old", "Young", "Middle", "Young"]),
         y = [missing, ones(11)...],
         z = [missing, missing, 1,1,1,1,1,5,1,1,1,1],
         a = rand("abc", 12))

    f, = MLJBase.fit(imp, 1, X)
    Xt = MLJBase.transform(imp, f, X)

    @test all(.!ismissing.(Xt.x))
    @test all(.!ismissing.(Xt.y))
    @test all(.!ismissing.(Xt.z))
    @test all(.!ismissing.(Xt.a))

    @test Xt.x[1] == mode_
    @test Xt.y[1] == 1
    @test Xt.z[1] == 1
end

#### CONTINUOUS ENCODER ####

@testset "Continuous encoder" begin

    X = (name  = categorical(["Ben", "John", "Mary", "John"], ordered=true),
         height = [1.85, 1.67, 1.5, 1.67],
         rubbish = ["a", "b", "c", "a"],
         favourite_number = categorical([7, 5, 10, 5]),
         age    = [23, 23, 14, 23])

    t  = ContinuousEncoder()
    f, _, _ = @test_logs((:info, r"Some features cannot be replaced "*
                               "with `Continuous` features and will be "*
                               "dropped: Symbol[:rubbish]"),
                              MLJBase.fit(t, 1, X))

    Xt = MLJBase.transform(t, f, X)
    @test scitype(Xt) <: MLJBase.Table(MLJBase.Continuous)
    s = MLJBase.schema(Xt)
    @test s.names == (:name, :height, :favourite_number__5,
                      :favourite_number__7, :favourite_number__10, :age)

    t  = ContinuousEncoder(drop_last=true, one_hot_ordered_factors=true)
    f, _, r = MLJBase.fit(t, 0, X)
    Xt = MLJBase.transform(t, f, X)
    @test scitype(Xt) <: MLJBase.Table(MLJBase.Continuous)
    s = MLJBase.schema(Xt)
    @test s.names == (:name__Ben, :name__John, :height, :favourite_number__5,
                      :favourite_number__7, :age)

end

end
true
