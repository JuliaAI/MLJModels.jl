# # DECODING MODEL METADATA
# (deserializing TOML dictionary)

function decode_dic(s::String)
    if !isempty(s)
        if  s[1] == ':'
            return Symbol(s[2:end])
        elseif s[1] == '`' && s[2] != '`' # to exclude strings starting with ```
            ex = Meta.parse(s[2:end-1])
            # We need a `try` here because `constructor` trait generally returns a
            # function not in the namespace, because pkg defining it has not been loaded.
            return try
                eval(ex)
            catch
                ex
            end
        else
            return s
        end
    else
        return ""
    end
end

decode_dic(v::Vector) = decode_dic.(v)
function decode_dic(d::AbstractDict)
    ret = LittleDict()
    for (k, v) in d
        ret[decode_dic(k)] = decode_dic(v)
    end
    return ret
end


# # MODEL HANDLES

Handle = NamedTuple{(:name, :pkg), Tuple{String,String}}
(::Type{Handle})(name,string) = NamedTuple{(:name, :pkg)}((name, string))

function Base.isless(h1::Handle, h2::Handle)
    if isless(h1.name, h2.name)
        return true
    elseif h1.name == h2.name
        return isless(h1.pkg, h2.pkg)
    else
        return false
    end
end

function (::Type{Handle})(name::String)
    if name in AMBIGUOUS_NAMES
        return Handle(name, missing)
    else
        return Handle(name, first(PKGS_GIVEN_NAME[name]))
    end
end


# # FUNCTIONS TO BUILD GLOBAL METADATA CONSTANTS

# to define INFO_GIVEN_HANDLE
function info_given_handle(metadata_file)
    metadata = LittleDict(TOML.parsefile(metadata_file))
    metadata_given_api_pkg = decode_dic(metadata)

    # build info_given_handle dictionary:
    ret = Dict{Handle}{Any}()
    packages = keys(metadata_given_api_pkg)
    for api_pkg in packages
        info_given_name = metadata_given_api_pkg[api_pkg]
        for name in keys(info_given_name)
            info = info_given_name[name]
            pkg = info[:package_name]
            handle = Handle(name, pkg)
            ret[handle] = info
        end
    end
    return ret

end

# to define AMBIGUOUS_NAMES
function ambiguous_names(info_given_handle)
    names_with_duplicates = map(keys(info_given_handle) |> collect) do handle
        handle.name
    end
    frequency_given_name = countmap(names_with_duplicates)
    return filter(keys(frequency_given_name) |> collect) do name
        frequency_given_name[name] > 1
    end
end

# to define PKGS_GIVEN_NAME
function pkgs_given_name(info_given_handle)
    handles = keys(info_given_handle) |> collect
    ret = Dict{String,Vector{String}}()
    for handle in handles
        if haskey(ret, handle.name)
           push!(ret[handle.name], handle.pkg)
        else
            ret[handle.name] =[handle.pkg, ]
        end
    end
    return ret
end

function model_names(info_given_handle)
    names_allowing_duplicates = map(keys(info_given_handle) |> collect) do handle
        handle.name
    end
    return unique(names_allowing_duplicates)
end

function model_traits_in_registry(info_given_handle)
    first_entry = info_given_handle[Handle("ConstantRegressor")]
    return keys(first_entry) |> collect
end
