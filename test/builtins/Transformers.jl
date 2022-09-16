module TestTransformer

using Test, MLJModels
using Tables, CategoricalArrays, Random
using ScientificTypes
using StatsBase
using Statistics
using StableRNGs
stable_rng = StableRNGs.StableRNG(123)
using Dates: DateTime, Date, Time, Day, Hour
import MLJBase

_get(x) = CategoricalArrays.DataAPI.unwrap(x)


#### FEATURE SELECTOR ####

@testset "Feat Selector" begin
    N = 100
    X = (Zn   = rand(N),
         Crim = rand(N),
         x3   = categorical(rand("YN", N)),
         x4   = categorical(rand("YN", N)))

    # Test feature selection with `features=Symbol[]`
    namesX   = Tables.schema(X).names |> collect
    selector = FeatureSelector()
    f,       = MLJBase.fit(selector, 1, X)
    @test f == namesX
    Xt = MLJBase.transform(selector, f, MLJBase.selectrows(X, 1:2))
    @test Set(Tables.schema(Xt).names) == Set(namesX)
    @test length(Xt.Zn) == 2

    # Test on selecting features if `features` keyword is defined
    selector = FeatureSelector(features=[:Zn, :Crim])
    f,       = MLJBase.fit(selector, 1, X)
    @test MLJBase.transform(selector, f, MLJBase.selectrows(X, 1:2)) ==
            MLJBase.select(X, 1:2, [:Zn, :Crim])

    # test on ignoring a feature, even if it's listed in the `features`
    selector.ignore = true
    f,   = MLJBase.fit(selector, 1, X)
    Xnew = MLJBase.transform(selector, f, X)
    @test MLJBase.transform(selector, f, MLJBase.selectrows(X, 1:2)) ==
         MLJBase.select(X, 1:2, [:x3, :x4])

    # test error about features selected or excluded in fit.
    selector = FeatureSelector(features=[:x1, :mickey_mouse])
    @test_throws(
        ArgumentError,
        MLJBase.fit(selector, 1, X)
    )
    selector.ignore = true
    @test_logs(
        (:warn, r"Excluding non-existent"),
        MLJBase.fit(selector, 1, X)
    )

    # features must be specified if ignore=true
    @test_throws ArgumentError FeatureSelector(ignore=true)

    # test logs for no features selected when using Bool-Callable function interface:
    selector = FeatureSelector(features= x-> x == (:x1))
   @test_throws(
        ArgumentError,
        MLJBase.fit(selector, 1, X)
    )
    selector.ignore = true
    selector.features = x-> x in [:Zn, :Crim, :x3, :x4]
     @test_throws(
        ArgumentError,
        MLJBase.fit(selector, 1, X)
    )

    # Test model Metadata
    infos = MLJModels.info_dict(selector)
    @test infos[:input_scitype]  == MLJBase.Table
    @test infos[:output_scitype] == MLJBase.Table
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
    @test MLJBase.inverse_transform(t, result, _get(w[43])) ≈ r
    @test MLJBase.inverse_transform(t, result, map(_get, w)) ≈
        MLJBase.inverse_transform(t, result, w)

    # all transformed vectors should have an identical pool (determined in
    # call to fit):
    v2 = v[1:3]
    w2 = MLJBase.transform(t, result, v2)
    @test levels(w2) == levels(w)

end

#### STANDARDIZER ####

@testset begin "standardization"

    # UnivariateStandardizer:
    stand = UnivariateStandardizer()
    f,    = MLJBase.fit(stand, 1, [0, 2, 4])
    @test round.(Int, MLJBase.transform(stand, f, [0,4,8])) == [-1.0,1.0,3.0]
    @test round.(Int, MLJBase.inverse_transform(stand, f, [-1, 1, 3])) ==
        [0, 4, 8]
    infos = MLJModels.info_dict(stand)

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

    # test inverse:
    XX = MLJBase.inverse_transform(stand, f, Xnew)
    @test MLJBase.schema(X) == MLJBase.schema(XX)
    @test XX.x1 == X.x1
    @test XX.x2 ≈ X.x2
    @test XX.x3 == X.x3
    @test XX.x4 == X.x4
    @test XX.x5 ≈ X.x5

    # test transformation:
    @test Xnew[1] == X[1]
    @test MLJBase.std(Xnew[2]) ≈ 1.0
    @test Xnew[3] == X[3]
    @test Xnew[4] == X[4]
    @test MLJBase.std(Xnew[5]) ≈ 1.0

    # test feature specification (ignore=false):
    stand.features = [:x1, :x5]
    f,   = MLJBase.fit(stand, 1, X)
    Xnew = MLJBase.transform(stand, f, X)
    @test issubset(Set(keys(f[3])), Set(Tables.schema(X).names[[5,]]))
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
    @test issubset(Set(keys(f[3])), Set(Tables.schema(X).names[[2,]]))
    Xt = MLJBase.transform(stand, f, X)
    @test Xnew[1] == X[1]
    @test MLJBase.std(Xnew[2]) ≈ 1.0
    @test Xnew[3] == X[3]
    @test Xnew[4] == X[4]
    @test Xnew[5] == X[5]

    # test warnings about features not encountered in fit or no
    # features need transforming:
    stand = Standardizer(features=[:x1, :mickey_mouse])
    @test_logs(
        (:warn, r"Some specified"),
        (:warn, r"No features"),
        MLJBase.fit(stand, 1, X)
    )
    stand.ignore = true
    @test_logs (:warn, r"Some specified") MLJBase.fit(stand, 1, X)

    # features must be specified if ignore=true
    @test_throws ArgumentError Standardizer(ignore=true)

    # test count, ordered_factor options:
    stand = Standardizer(features=[:x3, :x4], count=true, ordered_factor=true)
    f,   = MLJBase.fit(stand, 1, X)
    Xnew = MLJBase.transform(stand, f, X)
    @test issubset(Set(keys(f[3])), Set(Tables.schema(X).names[3:4,]))
    Xt = MLJBase.transform(stand, f, X)
    @test_throws Exception MLJBase.inverse_transform(stand, f, Xt)

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

    infos = MLJModels.info_dict(stand)

    @test infos[:name] == "Standardizer"
    @test infos[:input_scitype] ==
        Union{MLJBase.Table, AbstractVector{<:Continuous}}
    @test infos[:output_scitype] ==
        Union{MLJBase.Table, AbstractVector{<:Continuous}}

    # univariate case
    stand = Standardizer()
    f, _, _   = MLJBase.fit(stand, 1, [0, 2, 4])
    @test round.(Int, MLJBase.transform(stand, f, [0,4,8])) == [-1.0,1.0,3.0]
    @test [(MLJBase.fitted_params(stand, f).mean_and_std)...] ≈
        [2, MLJBase.std([0, 2, 4])]

end

### TIMETYPE TO CONTINUOUS

@testset "TimeTypeToContinuous" begin
    let dt = [Date(2018, 6, 15) + Day(i) for i=0:10],
        transformer = UnivariateTimeTypeToContinuous(; step=Day(1))
        fr, _, _ = MLJBase.fit(transformer, 1, dt)
        @test fr == (Date(2018, 6, 15), Day(1))
        dt_continuous = MLJBase.transform(transformer, fr, dt)
        @test all(dt_continuous .== Float64.(0:10))
    end

    let dt = [Date(2018, 6, 15) + Day(i) for i=0:10],
        transformer = UnivariateTimeTypeToContinuous()
        fr, _, _ = @test_logs(
            (:warn, r"Cannot add `TimePeriod` `step`"),
            MLJBase.fit(transformer, 1, dt)
        )
        fr, _, _ = @test_logs (:warn, r"C") MLJBase.fit(transformer, 1, dt)
        @test fr == (Date(2018, 6, 15), Day(1))
        dt_continuous = MLJBase.transform(transformer, fr, dt)
        @test all(dt_continuous .== Float64.(0:10))
    end

    let dt = [Time(0, 0, 0) + Hour(i) for i=0:3:30],
        transformer = UnivariateTimeTypeToContinuous(;
            step = Hour(1),
            zero_time = Time(7, 0, 0),
        )
        fr, _, _ = MLJBase.fit(transformer, 1, dt)
        @test fr == (Time(7, 0, 0), Hour(1))
        dt_continuous = MLJBase.transform(transformer, fr, dt)
        ex = collect(0:3:30) .% 24 .- 7.0
        diff = map(dt_continuous .- ex) do d
            mod(d, 24.0)
        end
        @test all(diff .≈ 0.0)
    end

    let dt = [Time(0, 0, 0) + Hour(i) for i=0:3:30],
        transformer = UnivariateTimeTypeToContinuous()
        fr, _, _ = MLJBase.fit(transformer, 1, dt)
        @test fr == (Time(0, 0, 0), Hour(24))
        dt_continuous = MLJBase.transform(transformer, fr, dt)
        ex = collect(0:3:30) .% 24 ./ 24
        diff = map(dt_continuous .- ex) do d
            mod(d, 1.0)
        end
        @test all(diff .≈ 0.0)
    end

    # test log messages
    let dt = [DateTime(2018, 6, 15) + Day(i) for i=0:10],
        step=Hour(1),
        zero_time=Date(2018, 6, 15),
        transformer = @test_logs(
            (:warn, "Cannot add `TimePeriod` `step` to `Date` `zero_time`. Converting `zero_time` to `DateTime`."),
            UnivariateTimeTypeToContinuous(;
                step=step,
                zero_time=zero_time,
            )
        )
        fr, _, _ = MLJBase.fit(transformer, 1, dt)

        @test fr == (zero_time, step)
        dt_continuous = MLJBase.transform(transformer, fr, dt)
        @test all(dt_continuous .== Float64.(0:10).*24)
    end

    let dt = [Time(0, 0, 0) + Hour(i) for i=0:3:30],
        zero_time=Time(0, 0, 0),
        step=Day(1),
        transformer = @test_logs(
            (:warn, "Cannot add `DatePeriod` `step` to `Time` `zero_time`. Converting `step` to `Hour`."),
            UnivariateTimeTypeToContinuous(;
                step=step,
                zero_time=zero_time,
            )
        )
        fr, _, _ = MLJBase.fit(transformer, 1, dt)

        @test fr == (zero_time, convert(Hour, step))
        dt_continuous = MLJBase.transform(transformer, fr, dt)
        ex = Float64.((0:3:30) .% 24)./24
        diff = map(dt_continuous .- ex) do d
            mod(d, 1.0)
        end
        @test all(diff .≈ 0.0)
    end

    let dt = [DateTime(2018, 6, 15) + Day(i) for i=0:10],
        step=Day(1),
        zero_time=Date(2018, 6, 15),
        transformer = UnivariateTimeTypeToContinuous(;
            step=step,
            zero_time=zero_time,
        )
        fr, _, _ = @test_logs(
            (:warn, r"`Date"),
            MLJBase.fit(transformer, 1, dt)
        )

        @test fr == (zero_time, step)
        dt_continuous = MLJBase.transform(transformer, fr, dt)
        @test all(dt_continuous .== Float64.(0:10))
    end
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

    infos = MLJModels.info_dict(t)

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
    @test MLJBase.schema(Xt).names == (:name, :height, :favourite_number__5,
                               :favourite_number__7, :favourite_number__10,
                               :age)

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

    # test to throw exception when category level mismatch is found
    X = (name   = categorical(["Ben", "John", "Mary", "John"], ordered=true),
         height = [1.85, 1.67, 1.5, 1.67],
         favourite_number = categorical([7, 5, 10, 5]),
         age    = [23, 23, 14, 23])
    Xmiss = (name   = categorical(["John", "Mary", "John"], ordered=true),
             height = X.height,
             favourite_number = X.favourite_number,
             age    = X.age)
    t  = OneHotEncoder()
    f, = MLJBase.fit(t, 0, X)
    @test_throws Exception MLJBase.transform(t, f, Xmiss)

    infos = MLJModels.info_dict(t)

    @test infos[:name] == "OneHotEncoder"
    @test infos[:input_scitype] == MLJBase.Table
    @test infos[:output_scitype] == MLJBase.Table

    # test the work on missing values
    X = (name   = categorical(["Ben", "John", "Mary", "John", missing], ordered=true),
         height = [1.85, 1.67, 1.5, 1.67, 1.56],
         favourite_number = categorical([7, 5, 10, missing, 5]),
         age    = [23, 23, 14, 23, 21])

    t  = OneHotEncoder()
    f, _, report = @test_logs((:info, r"Spawning 3"),
                         (:info, r"Spawning 3"), MLJBase.fit(t, 1, X))

    Xt = MLJBase.transform(t, f, X)

    @test length(Xt.name__John) == 5
    @test collect(skipmissing(Xt.name__John)) == float.([false, true, false, true])
    @test ismissing(Xt.name__John[5])
    @test Xt.height == X.height
    @test length(Xt.favourite_number__10) == 5
    @test collect(skipmissing(Xt.favourite_number__10)) == float.([false, false, true, false])
    @test ismissing(Xt.favourite_number__10[4])
    @test Xt.age == X.age
    @test MLJBase.schema(Xt).names == (:name__Ben, :name__John, :name__Mary,
                               :height, :favourite_number__5,
                               :favourite_number__7, :favourite_number__10,
                               :age)

    @test report.new_features == collect(MLJBase.schema(Xt).names)

    # test the work on missing values with drop_last = true

    X = (name   = categorical(["Ben", "John", "Mary", "John", missing], ordered=true),
         height = [1.85, 1.67, 1.5, 1.67, 1.56],
         favourite_number = categorical([7, 5, 10, missing, 5]),
         age    = [23, 23, 14, 23, 21])

    t  = OneHotEncoder(drop_last = true)
    f, _, report = @test_logs((:info, r"Spawning 2"),
                        (:info, r"Spawning 2"), MLJBase.fit(t, 1, X))

    Xt = MLJBase.transform(t, f, X)

    @test length(Xt.name__John) == 5
    @test collect(skipmissing(Xt.name__John)) == float.([false, true, false, true])
    @test ismissing(Xt.name__John[5])
    @test Xt.height == X.height
    @test ismissing(Xt.favourite_number__5[4])
    @test collect(skipmissing(Xt.favourite_number__5)) == float.([false, true, false, true])
    @test ismissing(Xt.favourite_number__5[4])
    @test Xt.age == X.age
    @test MLJBase.schema(Xt).names == (:name__Ben, :name__John,
                            :height, :favourite_number__5,
                            :favourite_number__7,
                            :age)

    @test_throws Exception Xt.favourite_number__10
    @test_throws Exception Xt.name__Mary
    @test report.new_features == collect(MLJBase.schema(Xt).names)
end


#### FILL IMPUTER ####'

@testset "UnivariateFillImputer" begin
    vpure = rand(stable_rng, 10)
    v = vcat([missing, ], vpure)
    filler = median(vpure)
    imp = MLJModels.UnivariateFillImputer()
    f, = MLJBase.fit(imp, 1, v)
    vnew = [missing, 1.0, missing, 2.0, 3.0]
    @test MLJBase.transform(imp, f, vnew) ≈ [filler, 1.0, filler, 2.0, 3.0]

    vpure = MLJBase.coerce(rand(stable_rng, "abc", 100), OrderedFactor);
    v = vcat([missing, ], vpure)
    filler = mode(vpure)
    imp = MLJModels.UnivariateFillImputer()
    f, = MLJBase.fit(imp, 1, v)
    vnew = vcat([missing, ], vpure[end-10:end], [missing, ])
    @test MLJBase.transform(imp, f, vnew) ==
        vcat([filler, ], vpure[end-10:end], [filler, ])

    vpure = rand(stable_rng, Int, 10)
    v = vcat([missing, ], vpure)
    filler = round(Int, median(vpure))
    imp = MLJModels.UnivariateFillImputer()
    f, = MLJBase.fit(imp, 1, v)
    vnew = [missing, 1, missing, 2, 3]
    @test MLJBase.transform(imp, f, vnew) == [filler, 1, filler, 2, 3]

    @test_throws Exception MLJBase.transform(imp, f, [missing, "1", "2"])

    @test_throws ArgumentError MLJBase.fit(imp, 1, [missing, "1", "2"])

end

@testset "FillImputer" begin
    X = (
        x = [missing,ones(10)...],
        y = [missing,ones(10)...],
        z = [missing,ones(10)...]
        )

    imp = FillImputer()
    f,  = MLJBase.fit(imp, 1, X)

    fp = MLJBase.fitted_params(imp, f)
    @test fp.features_seen_in_fit == [:x, :y, :z]
    @test fp.univariate_transformer == MLJModels.UnivariateFillImputer()
    @test fp.filler_given_feature[:x] ≈ 1.0
    @test fp.filler_given_feature[:x] ≈ 1.0
    @test fp.filler_given_feature[:x] ≈ 1.0

    Xnew = MLJBase.selectrows(X, 1:5)
    Xt  = MLJBase.transform(imp, f, Xnew)
    @test all(.!ismissing.(Xt.x))
    @test Xt.x isa Vector{Float64} # no missing
    @test all(Xt.x .== 1.0)

    imp = FillImputer(features=[:x,:y])
    f,  = MLJBase.fit(imp, 1, X)
    Xt = MLJBase.transform(imp, f, Xnew)
    @test all(Xt.x .== 1.0)
    @test all(Xt.y .== 1.0)
    @test ismissing(Xt.z[1])

    # adding a new feature not seen in fit:
    Xnew = (x = X.x, y=X.y, a=X.x)
    @test_throws ArgumentError  MLJBase.transform(imp, f, Xnew)

    # mixture of features:
    X = (x = categorical([missing, missing, missing, missing,
                          "Old", "Young", "Middle", "Young",
                          "Old", "Young", "Middle", "Young"]),
         y = [missing, ones(11)...],
         z = [missing, missing, 1,1,1,1,1,5,1,1,1,1],
         a = rand("abc", 12))

    imp = FillImputer()
    f, = MLJBase.fit(imp, 1, X)
    Xnew = MLJBase.selectrows(X, 1:4)
    Xt = MLJBase.transform(imp, f, Xnew)

    @test all(.!ismissing.(Xt.x))
    @test all(.!ismissing.(Xt.y))
    @test all(.!ismissing.(Xt.z))
    @test all(.!ismissing.(Xt.a))

    @test Xt.x[1] == mode(skipmissing(X.x))
    @test Xt.y[1] == 1
    @test Xt.z[1] == 1

    # user specifies a feature explicitly that's not supported:
    imp = FillImputer(features=[:x, :a]) # :a of Unknown scitype not supported
    @test_logs (:info, r"Feature a will not") MLJBase.fit(imp, 1, X)

end


#### CONTINUOUS ENCODER ####

@testset "Continuous encoder" begin

    X = (name  = categorical(["Ben", "John", "Mary", "John"], ordered=true),
         height = [1.85, 1.67, 1.5, 1.67],
         rubbish = ["a", "b", "c", "a"],
         favourite_number = categorical([7, 5, 10, 5]),
         age    = [23, 23, 14, 23])

    t  = ContinuousEncoder()
    f, _, _ = @test_logs((:info, r"Some.*dropped\:.*\:rubbish\]"),
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

#### INTERACTION TRANSFORMER ####

@testset "Interaction Transformer functions" begin
    # No column provided, A has scitype Continuous, B has scitype Count
    table = (A = [1., 2., 3.], B = [4, 5, 6], C = ["x₁", "x₂", "x₃"])
    @test MLJModels.actualfeatures(nothing, table) == (:A, :B)
    # Column provided
    @test MLJModels.actualfeatures([:A, :B], table) == (:A, :B)
    # Column provided, not in table
    @test_throws ArgumentError("Column(s) D are not in the dataset.") MLJModels.actualfeatures([:A, :D], table)
    # Non Infinite scitype column provided
    @test_throws ArgumentError("Column C's scitype is not Infinite.") MLJModels.actualfeatures([:A, :C], table)
end


@testset "Interaction Transformer" begin
    # Check constructor sanity checks: order > 1, length(features) > 1
    @test_logs (:warn, "Constraint `model.order > 1` failed; using default: order=2.") InteractionTransformer(order = 1)
    @test_logs (:warn, "Constraint `if model.features !== nothing\n"*
                       "    length(model.features) > 1\nelse\n    true\nend` failed; "*
                       "using default: features=nothing.") InteractionTransformer(features = [:A])

    X = (A = [1, 2, 3], B = [4, 5, 6], C = [7, 8, 9])
    # Default order=2, features=nothing, ie all columns
    Xt = MLJBase.transform(InteractionTransformer(), nothing, X)
    @test Xt == (
        A = [1, 2, 3],
        B = [4, 5, 6],
        C = [7, 8, 9],
        A_B = [4, 10, 18],
        A_C = [7, 16, 27],
        B_C = [28, 40, 54]
    )
    # order=3, features=nothing, ie all columns
    Xt = MLJBase.transform(InteractionTransformer(order=3), nothing, X)
    @test Xt == (
        A = [1, 2, 3],
        B = [4, 5, 6],
        C = [7, 8, 9],
        A_B = [4, 10, 18],
        A_C = [7, 16, 27],
        B_C = [28, 40, 54],
        A_B_C = [28, 80, 162]
    )
    # order=2, features=[:A, :B], ie all columns
    Xt =MLJBase.transform(InteractionTransformer(order=2, features=[:A, :B]), nothing, X)
    @test Xt == (
        A = [1, 2, 3],
        B = [4, 5, 6],
        C = [7, 8, 9],
        A_B = [4, 10, 18]
    )
    # order=3, features=[:A, :B, :C], some non continuous columns
    X = merge(X, (D = ["x₁", "x₂", "x₃"],))
    Xt = MLJBase.transform(InteractionTransformer(order=3, features=[:A, :B, :C]), nothing, X)
    @test Xt == (
        A = [1, 2, 3],
        B = [4, 5, 6],
        C = [7, 8, 9],
        D = ["x₁", "x₂", "x₃"],
        A_B = [4, 10, 18],
        A_C = [7, 16, 27],
        B_C = [28, 40, 54],
        A_B_C = [28, 80, 162]
    )
    # order=2, features=nothing, only continuous columns are dealt with
    Xt = MLJBase.transform(InteractionTransformer(order=2), nothing, X)
    @test Xt == (
        A = [1, 2, 3],
        B = [4, 5, 6],
        C = [7, 8, 9],
        D = ["x₁", "x₂", "x₃"],
        A_B = [4, 10, 18],
        A_C = [7, 16, 27],
        B_C = [28, 40, 54],
    )
end

end
true
