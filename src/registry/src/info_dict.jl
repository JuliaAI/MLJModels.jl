# `info_dict` returns a dictionary of model traits which, after
# encoding, can be serializing to TOML file to create the "model
# registry". Not intended to be exposed to user. Note that `info` gets
# the list of traits from the registry but `info_dict` gets the list
# from MLJModelInterface.MODEL_TRAITS, which is larger when new traits are
# added but the registry is not yet updated.

info_dict(model::Model) = info_dict(typeof(model))

ismissing_or_isa(x, T) = ismissing(x) || x isa T

function info_dict(M::Type{<:Model})
    message = "$M has a bad trait declaration.\n"
    ismissing_or_isa(is_pure_julia(M), Bool) ||
        error(message * "`is_pure_julia($M)` must return true or false")
    ismissing_or_isa(supports_weights(M), Bool) ||
        error(message * "`supports_weights($M)` must return true, "*
              "false or missing. ")
    ismissing_or_isa(supports_class_weights(M), Bool) ||
        error(message * "`supports_class_weights($M)` must return true, "*
              "false or missing. ")
    is_wrapper(M) isa Bool ||
        error(message * "`is_wrapper($M)` must return true, false. ")

    return LittleDict{Symbol,Any}(trait => eval(:($trait))(M)
                                  for trait in MLJModelInterface.MODEL_TRAITS)
end
