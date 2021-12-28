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
