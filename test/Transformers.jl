module TestTransformer
using Test, MLJBase, MLJModels
using Tables, CategoricalArrays, Random

#### STATIC TRANSFORMER ####

t  = StaticTransformer(f=log)
f, = fit(t, 1, nothing)
@test transform(t, f, 5) ≈ log(5)

infos = info_dict(t)
@test infos[:input_scitype]  == MLJBase.Table(MLJBase.Scientific)
@test infos[:output_scitype] == MLJBase.Table(Scientific)

#### FEATURE SELECTOR ####

N = 100
X = (Zn   = rand(N),
     Crim = rand(N),
     x3   = categorical(rand("YN", N)),
     x4   = categorical(rand("YN", N)))

namesX   = Tables.schema(X).names |> collect
selector = FeatureSelector()
typeof(selector.rule)
typeof(selector.rule)
f,       = fit(selector, 1, X)

@test f == namesX

Xt = transform(selector, f, selectrows(X, 1:2))

@test Set(Tables.schema(Xt).names) == Set(namesX)
@test length(Xt.Zn) == 2

selector = FeatureSelector(features=[:Zn, :Crim])
f,       = fit(selector, 1, X)

@test transform(selector, f, selectrows(X, 1:2)) ==
        selectcols(selectrows(X, 1:2), [:Zn, :Crim])

infos = info_dict(selector)
@test infos[:input_scitype]  == MLJBase.Table(Scientific)
@test infos[:output_scitype] == MLJBase.Table(Scientific)

#### FeatureSelectorRule ####
X = (name       = categorical(["Ben", "John", "Mary", "John"], ordered=true),
     height     = [1.85, 1.67, 1.5, 1.67],
     dummy     = [1.0, 1.00, 1.0, 1.0],
     favourite_number = categorical([7, 5, 10, 5]),
     age        = [23, 23, 14, 23],
     gender     = categorical(['M', 'M', 'F', 'M']))


mutable struct StdRule <: MLJModels.SelectorRule
     threshold::Float64
end
(sr::StdRule)(X,name,type,scitypes) =  scitypes <: Continuous ? (std(X[name])>sr.threshold ? true : false) : true
fsr=FeatureSelector(rule=StdRule(0.2))
fsr_fit,=fit(fsr,1,X)
Xt=MLJBase.transform(fsr,fsr_fit,X)
@test !(:height in schema(Xt).names)

# mutable struct StdFeatureSelector
#     threshold::Float64
# end
# StdFeatureSelector(;threshold=1.0)=FeatureSelector(rule=StdRule(threshold))
# StdFeatureSelector(threshold=0.1)



#### UNIVARIATE STANDARDIZER ####

stand = UnivariateStandardizer()
f,    = fit(stand, 1, [0, 2, 4])

@test round.(Int, transform(stand, f, [0,4,8])) == [-1.0,1.0,3.0]
@test round.(Int, inverse_transform(stand, f, [-1, 1, 3])) == [0, 4, 8]

infos = info_dict(stand)
@test infos[:package_name] == "MLJModels"
@test infos[:name] == "UnivariateStandardizer"
@test infos[:input_scitype] == AbstractVector{<:Infinite}
@test infos[:output_scitype] == AbstractVector{Continuous}

#### STANDARDIZER ####

N = 5
X = (OverallQual  = rand(UInt8, N),
     GrLivArea    = rand(N),
     Neighborhood = categorical(rand("abc", N)),
     x1stFlrSF    = rand(N),
     TotalBsmtSF  = rand(N))

# introduce a field of type `Char`:
x1 = categorical(map(Char, (X.OverallQual |> collect)))

# introduce field of Int type:
x4 = [round(Int, x) for x in X.x1stFlrSF]

X = (x1=x1, x2=X[2], x3=X[3], x4=x4, x5=X[5])

stand = Standardizer()
f,    = fit(stand, 1, X)
Xnew  = transform(stand, f, X)

@test Xnew[1] == X[1]
@test std(Xnew[2]) ≈ 1.0
@test Xnew[3] == X[3]
@test Xnew[4] == X[4]
@test std(Xnew[5]) ≈ 1.0

stand.features = [:x1, :x5]
f,   = fit(stand, 1, X)
Xnew = transform(stand, f, X)
f,   = fit(stand, 1, X)

@test issubset(Set(keys(f)), Set(Tables.schema(X).names[[5,]]))

Xt = transform(stand, f, X)

@test Xnew[1] == X[1]
@test Xnew[2] == X[2]
@test Xnew[3] == X[3]
@test Xnew[4] == X[4]
@test std(Xnew[5]) ≈ 1.0

infos = info_dict(stand)

@test infos[:name] == "Standardizer"
@test infos[:input_scitype] == MLJBase.Table(Scientific)
@test infos[:output_scitype] == MLJBase.Table(Scientific)


#### UNIVARIATE BOX COX TRANSFORMER ####


# create skewed non-negative vector with a zero value:
Random.seed!(1551)
v = abs.(randn(1000))
v = v .- minimum(v)

t  = UnivariateBoxCoxTransformer(shift=true)
f, = fit(t, 2, v)

@test sum(abs.(v - inverse_transform(t, f, transform(t, f, v)))) <= 5000*eps()

infos = info_dict(t)

@test infos[:name] == "UnivariateBoxCoxTransformer"
@test infos[:input_scitype] == AbstractVector{Continuous}
@test infos[:output_scitype] == AbstractVector{Continuous}


#### ONE HOT ENCODER ####


X = (name   = categorical(["Ben", "John", "Mary", "John"], ordered=true),
     height = [1.85, 1.67, 1.5, 1.67],
     favourite_number = categorical([7, 5, 10, 5]),
     age    = [23, 23, 14, 23])

t  = OneHotEncoder()
f, = @test_logs((:info, r"Spawning 3"), (:info, r"Spawning 3"), fit(t, 1, X))

Xt = transform(t, f, X)

@test Xt.name__John == float.([false, true, false, true])
@test Xt.height == X.height
@test Xt.favourite_number__10 == float.([false, false, true, false])
@test Xt.age == X.age
@test schema(Xt).names == (:name__Ben, :name__John, :name__Mary,
                           :height, :favourite_number__5,
                           :favourite_number__7, :favourite_number__10,
                           :age)

# test that *entire* pool of categoricals is used in fit, including
# unseen levels:
f, = @test_logs((:info, r"Spawning 3"), (:info, r"Spawning 3"),
                      fit(t, 1, MLJBase.selectrows(X,1:2)))
Xtsmall = transform(t, f, X)
@test Xt == Xtsmall

# test that transform can be applied to subset of the data:
@test transform(t, f, MLJBase.selectcols(X, [:name, :age])) ==
    MLJBase.selectcols(transform(t, f, X),
                       [:name__Ben, :name__John, :name__Mary, :age])

# test exclusion of ordered factors:
t  = OneHotEncoder(ordered_factor=false)
f, = fit(t, 1, X)
Xt = transform(t, f, X)
@test :name in Tables.schema(Xt).names
@test :favourite_number__5 in Tables.schema(Xt).names

# test that one may not add new columns:
X = (name       = categorical(["Ben", "John", "Mary", "John"], ordered=true),
     height     = [1.85, 1.67, 1.5, 1.67],
     favourite_number = categorical([7, 5, 10, 5]),
     age        = [23, 23, 14, 23],
     gender     = categorical(['M', 'M', 'F', 'M']))
@test_throws Exception transform(t, f, X)

infos = info_dict(t)

@test infos[:name] == "OneHotEncoder"
@test infos[:input_scitype] == MLJBase.Table(Scientific)
@test infos[:output_scitype] == MLJBase.Table(Scientific)


#### FILL IMPUTER ####

X = (
    x = [missing,ones(10)...],
    y = [missing,ones(10)...],
    z = [missing,ones(10)...]
    )

imp = FillImputer()
f,  = fit(imp, 1, X)
Xt  = transform(imp, f, X)
@test all(.!ismissing.(Xt.x))
@test Xt.x isa Vector{Float64} # no missing
@test all(Xt.x .== 1.0)

imp = FillImputer(features=[:x,:y])
f,  = fit(imp, 1, X)
@test_throws ErrorException transform(imp, f, X) # in X there's :z which we haven't trained on

X = (x = categorical([missing, missing, missing, missing, "Old", "Young", "Middle", "Young",
                      "Old", "Young", "Middle", "Young"]), )

mode_ = mode(["Old", "Young", "Middle", "Young", "Old", "Young", "Middle", "Young"])

imp = FillImputer()
f,  = fit(imp, 1, X)
Xt  = transform(imp, f, X)
@test all(.!ismissing.(Xt.x))
@test all(Xt.x[ismissing.(X.x)] .== mode_)

X  = (x = [missing, missing, 1, 1, 1, 1, 1, 5], )
f, = fit(imp, 1, X)
Xt = transform(imp, f, X)
@test Xt.x == [1, 1, 1, 1, 1, 1, 1, 5]

X = (x = categorical([missing, missing, missing, missing, "Old", "Young", "Middle", "Young",
                      "Old", "Young", "Middle", "Young"]),
     y = [missing, ones(11)...],
     z = [missing, missing, 1,1,1,1,1,5,1,1,1,1],
     a = rand("abc", 12))

f, = fit(imp, 1, X)
Xt = transform(imp, f, X)

@test all(.!ismissing.(Xt.x))
@test all(.!ismissing.(Xt.y))
@test all(.!ismissing.(Xt.z))
@test all(.!ismissing.(Xt.a))

@test Xt.x[1] == mode_
@test Xt.y[1] == 1
@test Xt.z[1] == 1

end
true
