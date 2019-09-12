"""
docstring_ext

Helper function to generate the docstring for an external package.
"""
function docstring_ext(T; descr::String="")
    package_name = MLJBase.package_name(T)
    package_url  = MLJBase.package_url(T)
    model_name   = MLJBase.name(T)
    # the message to return
    message      = "$descr"
    message     *= "\n→ based on [$package_name]($package_url)"
    message     *= "\n→ do `@load $model_name` to use the model"
    message     *= "\n→ do `?$model_name` for documentation."
end

"""
metadata_pkg

Helper function to write the metadata for a package.
"""
function metadata_pkg(T; name::String="unknown", uuid::String="unknown", url::String="unknown",
                         julia::Union{Missing,Bool}=missing, license::String="unknown",
                         wrapper::Bool=false)
    ex = quote
        MLJBase.package_name(::Type{<:$T})    = $name
        MLJBase.package_uuid(::Type{<:$T})    = $uuid
        MLJBase.package_url(::Type{<:$T})     = $url
        MLJBase.is_pure_julia(::Type{<:$T})   = $julia
        MLJBase.package_license(::Type{<:$T}) = $license
        MLJBase.is_wrapper(::Type{<:$T})      = $wrapper
    end
    eval(ex)
end

"""
metadata_mod

Helper function to write the metadata for a single model of a package (complements
[`metadata_ext`](@ref)).
"""
function metadata_mod(T; input=MLJBase.Unknown, target=MLJBase.Unknown,
                         output=MLJBase.Unknown, weights::Bool=false,
                         descr::String="", path::String="")
    isempty(path) && (path = "MLJModels.$(MLJBase.package_name(T))_.$T")

    ex = quote
        MLJBase.input_scitype(::Type{<:$T})    = $input
        MLJBase.output_scitype(::Type{<:$T})   = $output
        MLJBase.target_scitype(::Type{<:$T})   = $target
        MLJBase.supports_weights(::Type{<:$T}) = $weights
        MLJBase.docstring(::Type{<:$T})        = docstring_ext($T, descr=$descr)
        MLJBase.load_path(::Type{<:$T})        = $path
    end
    eval(ex)
end
