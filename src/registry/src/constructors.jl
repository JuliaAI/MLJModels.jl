"""
    model_type_given_constructor()

**Private method.**

Return a dictionary of all subtypes of MLJ.Model, keyed on constructor. Where multiple
types share a single constructor, there can only be one key, and which key appears is
ambiguous.

Typically a model type and it's constructor have the same name, but for wrappers, such as
`TunedModel`, several types share the same constructor (e.g., `DeterministicTunedModel`,
`ProbabilisticTunedModel`).

"""
function model_type_given_constructor()

    # Note that wrappers are required to overload `MLJModelInterface.constructor` and the
    # fallback is `nothing`.
    modeltypes =
        MLJModels.Registry.finaltypes(MLJModels.Model)
    filter!(modeltypes) do T
        !isabstracttype(T)
    end

    return Dict(
        map(modeltypes) do M
            C =  MLJModelInterface.constructor(M)
            Pair(isnothing(C) ? M : C, M)
        end...,
    )
end
