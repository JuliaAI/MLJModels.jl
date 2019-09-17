# This defines a macro `mlj_model` which is a simpler version than the
# @sk_model macro defined to help import sklearn models.
# The difference is that the `mlj_model` macro only defines the constructor and the `clean!`
# and does not automatically define the `fit` and `predict` methods
#
# NOTE: it does NOT handle parametric types yet.

import Base: @__doc__

"""
_unpack!(ex, rep)

Internal function to allow to read a constraint given after a default value for a parameter
and transform it in an executable condition (which is returned to be executed later).
For instance if we have

    alpha::Int = 0.5::(arg > 0.0)

Then it would transform the `(arg > 0.0)` in `(alpha > 0.0)` which is executable.
"""
function _unpack!(ex::Expr, rep)
    for i in eachindex(ex.args)
        if ex.args[i] ∈ (:_, :arg)
            ex.args[i] = rep
        end
        _unpack!(ex.args[i], rep)
    end
    return ex
end
_unpack!(ex, _) = ex # when it's been unpacked, it's not an expression anymore


"""
mlj_model

Macro to help define MLJ models with constraints on the default parameters, this can be seen as
a tweaked version of the `@with_kw` macro from `Parameters`.
"""
macro mlj_model(ex)
    # pull out defaults and constraints
    defaults    = Dict{Symbol,Any}()
    constraints = Dict{Symbol,Any}()
    # name of the model
    Model = ex.args[2] isa Symbol ? ex.args[2] : ex.args[2].args[1]

    # inspect all fields, retrieve the names and the constraints
    fnames = Symbol[]
    for i in 1:length(ex.args[3].args)
        # retrieve meaningful lines
        f = ex.args[3].args[i]
        f isa LineNumberNode && continue

        # line without information (e.g. just a name "param")
        if f isa Symbol
            push!(fnames, f)
            defaults[f] = missing
        else
            # A meaningful line will look like
            #   f.args[1] = f.args[2]
            #
            # where f.args[1] will either be just `name`  or `name::Type`
            # and   f.args[2] will either be just `value` or `value::constraint`

            # -- decompose `f.args[1]` appropriately to retrieve the field name

            if f.args[1] isa Symbol
                # :a
                fname = f.args[1]
                ftype = length(f.args)>1 ? f.args[2] : :Any
            else
                # :(a::Int)
                fname, ftype = f.args[1].args[1:2]
            end
            push!(fnames, fname)

            # -- decompose `f.args[2]` appropriately to retrieve the value and constraint

            if f.head == :(=) # assignment for default
                default = f.args[2]
                # if a constraint is given (value::constraint)
                if default isa Expr
                    if length(default.args) > 1
                        constraints[fname] = default.args[2] # constraint
                        default = default.args[1]
                    end
                end
                defaults[fname]    = default    # this will be a value not an expr
                ex.args[3].args[i] = f.args[1]  # name or name::Type (for the constructor)
            else
                # these are simple heuristics when no default value is given for the
                # field but an "obvious" one can be provided implicitly (ideally this
                # should not be used as it's not very clear that the intention matches the usage)
                eff_ftype = eval(ftype)
                if eff_ftype <: Number
                    defaults[fname] = zero(eff_ftype)
                elseif eff_ftype <: AbstractString
                    defaults[fname] = ""
                elseif eff_type == Any          # e.g. Any or no type given
                    defaults[fname] = nothing
                elseif eff_type >: Nothing      # e.g. Union{Nothing, ...}
                    defaults[fname] = nothing
                elseif eff_ftype >: Missing     # e.g. Union{Missing, ...} (unlikely)
                    defaults[fname] = missing
                else
                    @error "A default value for parameter '$fname' of type '$ftype' must be " *
                           "provided."
                end
            end
        end
    end

    # Build the kw constructor
    const_ex = Expr(:function, Expr(:call, Model, Expr(:parameters,
     	                  [Expr(:kw, fname, defaults[fname]) for fname in fnames]...)),
                    # body of the function
                    Expr(:block,
                         Expr(:(=), :model, Expr(:call, :new, [fname for fname in fnames]...)),
                         :(message = MLJBase.clean!(model)),
            			 :(isempty(message) || @warn message),
            			 :(return model)
        				 )
    			 	)
    push!(ex.args[3].args, const_ex)

    clean_ex = Expr(:function, :(MLJBase.clean!(model::$Model)),
                # body of the function
                Expr(:block,
                     :(warning = ""),
                     # condition and action for each constraint
                     # each parameter is given as field::Type = default::constraint
                     # here we recuperate the constraint and express it as an if statement
                     # for instance if we had
                     #     alpha::Real = 0.0::(arg > 0.0)
                     # this would become
                     #     if !(alpha > 0.0)
                     [Expr(:if, Expr(:call, :!, _unpack!(constr, :(model.$param))),
                           # action of the constraint is violated:
                           # add a message and use default for the parameter
                           Expr(:block,
                                :(warning *= $("Constraint `$constr` failed; using default: $param=$(defaults[param]).")),
                                :(model.$param = $(defaults[param]))
                                )
                           ) for (param, constr) in constraints]...,
                     # return full message
                     :(return warning)
                    )
                )

    esc(
        quote
            Base.@__doc__ $ex
            export $Model
            $ex
            $clean_ex
        end
    )
end
