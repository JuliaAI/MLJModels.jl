module TestThresholdPredictors
using Test, MLJModels, CategoricalArrays

import MLJBase

@testset "BinaryThresholdPredictor" begin
    X = NamedTuple{(:x1,:x2,:x3)}((rand(4), rand(4), rand(4)))
    yraw = ["MLJ", "ScikitLearn", "ScikitLearn", "ScikitLearn"]
    y = categorical(yraw, ordered = true) # scitype `AbstractArray{OrderedFactor{2}, 1}`
    y1 = categorical(yraw) # scitype `AbstractArray{Multiclass{2}, 1}
    y2 = categorical(yraw[2:end], ordered=true)
    
    model = BinaryThresholdPredictor()
    
    # Check if warning shows up when target (y) is of  
    # scitype `AbstractArray{Multiclass{2},1}`
    @test_logs((:warn, r"Taking positive class as"),
        MLJBase.fit(model, 1, X, y1))
        
    # Test for case where `AbstractVector{<:UnivariateFinite}` predictions 
    # contains a single class
    # Note that ideally fitting a `BinaryThresholdPredictor` model requires using a
    # `Binary` target `y` but for testing this case a target vector of scitype 
    # `AbstractArray{OrderedFactor{1}, 1} is used to simulate single class predctions.
    f, _, _ = MLJBase.fit(model, 1, MLJBase.selectrows(X, 2:4), y2)
    @test_logs((:warn, r"Predicted `AbstractVector{<:UnivariateFinite}`"),
        MLJBase.predict(model, f, X))
    @test MLJBase.predict(model, f, X) == 
        MLJBase.predict_mode(model.wrapped_model, f[1], X)
        
    # Check predictions containing two classes
    @test_throws ArgumentError BinaryThresholdPredictor(wrapped_model=ConstantRegressor())
    @test_logs((:warn, r"`threshold` should be"), BinaryThresholdPredictor(threshold=-1))
    @test_logs((:warn, r"`threshold` should be"), BinaryThresholdPredictor(threshold=1))
    
    # Compare fitresult and fitted_params with that of wrapped_model 
    model_fr, model_cache, model_report = MLJBase.fit(model, 1, X, y)
    wrapped_model_fr, wrapped_model_cache, wrapped_model_report = 
        MLJBase.fit(model.wrapped_model, 1, X, y) 
    @test model_fr[1] == wrapped_model_fr
    
    # Check model update  
    model_up, model_cache_up, model_report_up = 
        MLJBase.update(model, 1, model_fr, model_cache, X, y)
    wrapped_model_up, wrapped_model_cache_up, wrapped_model_report_up = 
        MLJBase.update(model.wrapped_model, 1, wrapped_model_fr, wrapped_model_cache, X, y)
    @test model_up[1] == wrapped_model_up
    @test model_cache_up[1] == wrapped_model_cache_up
    @test model_report_up[1] == wrapped_model_report_up
    
    # Check fitted_params
    @test MLJBase.fitted_params(model, model_fr).wrapped_model_fitted_params == 
         MLJBase.fitted_params(model.wrapped_model, wrapped_model_fr)
    
    # Check deterministic predictions
    @test MLJBase.predict(model, model_fr, X) == 
        MLJBase.predict_mode(model.wrapped_model, wrapped_model_fr, X)
     
    model.threshold = 0.8
    model_fr, cache, report = MLJBase.fit(model, 1, X, y)
    @test MLJBase.predict(model, model_fr, X) == 
        [y[1] for i in 1:MLJBase.nrows(X)]

    d = MLJBase.info_dict(model)
    @test d[:supports_weights] == MLJBase.supports_weights(model.wrapped_model)
    @test d[:input_scitype] == MLJBase.input_scitype(model.wrapped_model)
    @test d[:target_scitype] == AbstractVector{<:MLJBase.Finite{2}}
    @test d[:is_pure_julia] == MLJBase.is_pure_julia(model.wrapped_model)
    @test d[:name] == "BinaryThresholdPredictor"
    @test d[:load_path] == "MLJModels.BinaryThresholdPredictor"
end

@testset "_predict_threshold" begin
    v1 = categorical([:a, :b, :a])
    v2 = categorical([:a, :b, :a, :c])    
    # Test with UnivariateFinite object
    d1 = MLJBase.UnivariateFinite(MLJBase.classes(v1), [0.4, 0.6])
    @test_throws ArgumentError MLJModels._predict_threshold(d1, 0.7)
    @test MLJModels._predict_threshold(d1, (0.7, 0.3)) == v1[2]
    @test MLJModels._predict_threshold(d1, [0.5, 0.5]) == v1[2]
    @test MLJModels._predict_threshold(d1, (0.4, 0.6)) == v1[1]
    @test MLJModels._predict_threshold(d1, [0.2, 0.8]) == v1[1]
    d2 = MLJBase.UnivariateFinite(MLJBase.classes(v2), [0.4, 0.3, 0.3])
    @test_throws ArgumentError MLJModels._predict_threshold(d2, (0.7, 0.3))
    @test MLJModels._predict_threshold(d2, (0.2, 0.5, 0.3)) == v2[1]
    @test MLJModels._predict_threshold(d2, [0.3, 0.2, 0.5]) == v2[2]
    @test MLJModels._predict_threshold(d2, (0.4, 0.4, 0.2)) == v2[4]
    @test MLJModels._predict_threshold(d2, [0.2, 0.5, 0.3]) == v2[1]
    
    # Test with Array{UnivariateFinite, 1} object
    d1_arr = [d1 for i in 1:3]
    @test_throws ArgumentError MLJModels._predict_threshold(d1_arr, 0.7)
    @test MLJModels._predict_threshold(d1_arr, (0.7, 0.3)) == [v1[2] for i in 1:3]
    @test MLJModels._predict_threshold(d1_arr, [0.5, 0.5]) == [v1[2] for i in 1:3]
    @test MLJModels._predict_threshold(d1_arr, (0.4, 0.6)) == [v1[1] for i in 1:3]
    @test MLJModels._predict_threshold(d1_arr, [0.2, 0.8]) == [v1[1] for i in 1:3]
    d2_arr = [d2 for i in 1:3]
    @test_throws ArgumentError MLJModels._predict_threshold(d2_arr, (0.7, 0.3))
    @test MLJModels._predict_threshold(d2_arr, (0.2, 0.5, 0.3)) == [v2[1] for i in 1:3]
    @test MLJModels._predict_threshold(d2_arr, [0.3, 0.2, 0.5]) == [v2[2] for i in 1:3]
    @test MLJModels._predict_threshold(d2_arr, (0.4, 0.4, 0.2)) == [v2[4] for i in 1:3]
    @test MLJModels._predict_threshold(d2_arr, [0.2, 0.5, 0.3]) == [v2[1] for i in 1:3]
    
    # Test with UnivariateFiniteArray oject
    probs1 = [0.2 0.8; 0.7 0.3; 0.1 0.9]    
    unf_arr1 = MLJBase.UnivariateFinite(MLJBase.classes(v1), probs1)
    @test_throws ArgumentError MLJModels._predict_threshold(unf_arr1, 0.7)
    @test MLJModels._predict_threshold(unf_arr1, (0.7, 0.3)) == [v1[2], v1[1], v1[2]]
    @test MLJModels._predict_threshold(unf_arr1, [0.5, 0.5]) == [v1[2], v1[1], v1[2]]
    @test MLJModels._predict_threshold(unf_arr1, (0.4, 0.6)) == [v1[2], v1[1], v1[2]]
    @test MLJModels._predict_threshold(unf_arr1, [0.2, 0.8]) == [v1[1], v1[1], v1[2]]
    probs2 = [0.2 0.3 0.5;0.1 0.6 0.3; 0.4 0.0 0.6]
    unf_arr2 = MLJBase.UnivariateFinite(MLJBase.classes(v2), probs2)
    @test_throws ArgumentError MLJModels._predict_threshold(unf_arr2, (0.7, 0.3))
    @test MLJModels._predict_threshold(unf_arr2, (0.2, 0.5, 0.3)) == [v2[4], v2[2], v2[1]]
    @test MLJModels._predict_threshold(unf_arr2, [0.3, 0.2, 0.5]) == [v2[2], v2[2], v2[1]]
    @test MLJModels._predict_threshold(unf_arr2, (0.4, 0.4, 0.2)) == [v2[4], v2[2], v2[4]]
    @test MLJModels._predict_threshold(unf_arr2, [0.2, 0.5, 0.3]) == [v2[4], v2[2], v2[1]]
end

end
true
