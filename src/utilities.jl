function finaltypes(T::Type)
    s = InteractiveUtils.subtypes(T)
    if isempty(s)
        return [T, ]
    else
        return reduce(vcat, [finaltypes(S) for S in s])
    end
end

"""
    request(query, options...)

Query user at terminal by sending `query` to `stdout` and providing a
list of `options`. The index of the chosen option within `options` is
returned. If the user enters `q` or `control-C`, then an
`InterruptException` is thrown.

"""

function request(query, options...)
    if stdout isa Base.TTY
        println()
        menu = REPL.TerminalMenus.RadioMenu([options...])
        choice = REPL.TerminalMenus.request(query*"\n", menu) # index of option
        choice == -1 && throw(InterruptException)
        println()
    else
        invalid = true
        while invalid
            println()
            println(query)
            println()
            for i in eachindex(options)
                println("$i - $(options[i])")
            end
            raw_choice = readline()
            choice =
                if raw_choice == "q" || raw_choice == "Q"
                    throw(InterruptException)
                else
                    try
                        c = parse(Int, raw_choice)
                        invalid = false
                        c
                    catch exception
                        exception isa ArgumentError || throw(exception)
                    end
                end
        end
    end
    return choice
end


"""
    MLJModels.prepend(::Symbol, ::Union{Symbol,Expr,Nothing})

For prepending symbols in expressions like `:(y.w)` and `:(x1.x2.x3)`.

julia> prepend(:x, :y)
:(x.y)

julia> prepend(:x, :(y.z))
:(x.y.z)

julia> prepend(:w, ans)
:(w.x.y.z)

If the second argument is `nothing`, then `nothing` is returned.

"""
prepend(s::Symbol, ::Nothing) = nothing
prepend(s::Symbol, t::Symbol) = Expr(:(.), s, QuoteNode(t))
prepend(s::Symbol, ex::Expr) = Expr(:(.), prepend(s, ex.args[1]), ex.args[2])
