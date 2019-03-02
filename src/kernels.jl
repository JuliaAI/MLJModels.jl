abstract type SmoothingKernel end

struct UniformKernel <: SmoothingKernel end
struct TriangleKernel <: SmoothingKernel end
struct EpanechnikovKernel <: SmoothingKernel end
struct BiweightKernel <: SmoothingKernel end
struct TriweightKernel <: SmoothingKernel end
struct TricubeKernel <: SmoothingKernel end
struct GaussianKernel <: SmoothingKernel end
struct CosineKernel <: SmoothingKernel end
struct LogisticKernel <: SmoothingKernel end
struct SigmoigKernel <: SmoothingKernel end
struct SilvermanKernel <: SmoothingKernel end

function smooth(k :: UniformKernel, u)
    return 1/2 * (abs(u) <= 1.0)
end

function smooth(k :: TriangleKernel, u)
    return (1 - abs(u))*(abs(u) <= 1)
end

function smooth(k :: EpanechnikovKernel, u)
    return 3/4 * (1 - u^2) * (abs(u) <= 1)
end

function smooth(k :: BiweightKernel, u)
    return 15/16 * (1 - u^2)^2 * (abs(u) <= 1)
end

function smooth(k :: TriweightKernel, u)
    return 35/32 * (1 - u^2)^3 * (abs(u) <= 1)
end

function smooth(k :: TricubeKernel, u)
    return 70/81 * (1 - abs(u)^3) * (abs(u) <= 1)
end

function smooth(k :: GaussianKernel, u)
    return (1 / sqrt(2 * pi)) *  exp(-1/2 * u^2)
end

function smooth(k :: CosineKernel, u)
    return (pi / 4) * cos((pi / 2) * u) * float64(abs(u) <= 1.0)
end 

function smooth(k :: LogisticKernel, u)
    return 1 / (exp(u) + 2 + exp(-u))
end

function smooth(k :: SigmoigKernel, u)
    return 2/pi * 1 / (exp(u) + exp(-u))
end

function smooth(k :: SilvermanKernel, u)
    return 1/2 * exp.(-abs.(u) / sqrt(2)) .* sin.(abs.(u)/ sqrt(2) .+ pi/4)
end

function smooth(k :: SilvermanKernel, u)
    return 1/2 * exp(-abs(u) / sqrt(2)) * sin(abs(u)/ sqrt(2) + pi/4)
end
