using LinearAlgebra
using Distributions
using SequentialMonteCarlo
using StaticArrays
using Random
using LogExpFunctions

function density_product(dist1::MvNormal, dist2::MvNormal)
    constMvN = MvNormal(dist2.μ, dist1.Σ + dist2.Σ) # constant multiplier
    Ω = inv(dist1.Σ) + inv(dist2.Σ) # precision matrix
    h = (dist1.Σ \ dist1.μ) + (dist2.Σ \ dist2.μ)
    return logpdf(constMvN, dist1.μ), MvNormalCanon(h, Ω)
end

# # density_product2 uses standard parametrisation
# function density_product2(dist1::MvNormal, dist2::MvNormal)
#     constMvN = MvNormal(dist2.μ, dist1.Σ + dist2.Σ) # constant multiplier
#     Σ = inv(inv(dist1.Σ) + inv(dist2.Σ))
#     μ = Σ * (dist1.Σ \ dist1.μ) + (dist2.Σ \ dist2.μ)
#     return logpdf(constMvN, dist1.μ), MvNormal(μ, Σ)
# end

struct MvNormalScaledDensityProduct
    Ω₁::Matrix{Float64}
    Ω₂::Matrix{Float64}
    h₁::Vector{Float64}
    h₂::Vector{Float64}
    Σ₁::Matrix{Float64} # for log constant calculation
    Σ₂::Matrix{Float64} # for log constant calculation
    Δμ::Vector{Float64} # for log constant calculation
end

MvNormalScaledDensityProduct(dist1::MvNormal, dist2::MvNormal) = MvNormalScaledDensityProduct(inv(dist1.Σ), inv(dist2.Σ), dist1.Σ \ dist1.μ, dist2.Σ \ dist2.μ, dist1.Σ, dist2.Σ, dist1.μ - dist2.μ)

distr(dist::MvNormalScaledDensityProduct, s₁::Float64, s₂::Float64) = MvNormalCanon( (dist.h₁ / s₁) + (dist.h₂ / s₂), (dist.Ω₁ / s₁) + (dist.Ω₂ / s₂))
distr(dist::MvNormalScaledDensityProduct, s₁::Float64) = distr(dist, s₁, 1.0)

distr(dist::MvNormalScaledDensityProduct, s₁::Float64, x₁::MVector{d, Float64}) where d = MvNormalCanon( ((dist.h₁ + (dist.Σ₁ \ x₁))/ s₁) + dist.h₂, (dist.Ω₁ / s₁) + dist.Ω₂)

logconstant(dist::MvNormalScaledDensityProduct, s₁::Float64, s₂::Float64) = begin
    constMvN = MvNormal(dist.Δμ, (dist.Σ₁ * s₁) + (dist.Σ₂ * s₂)) # constant multiplier
    return logpdf(constMvN, zeros(length(dist.Δμ)))
end
logconstant(d::MvNormalScaledDensityProduct, s₁::Float64) = logconstant(d, s₁, 1.0)

logconstant(dist::MvNormalScaledDensityProduct, s₁::Float64, x₁::MVector{d, Float64}) where d = begin
    constMvN = MvNormal(x₁ + dist.Δμ, (dist.Σ₁ * s₁) + dist.Σ₂) # constant multiplier
    return logpdf(constMvN, zeros(length(dist.Δμ)))
end


# distr(MvNormalScaledDensityProduct(latentgauss,obsgauss), 1.0) # should be the same as 
# density_product(latentgauss, obsgauss)
# distr(MvNormalScaledDensityProduct(MvNormal([1.0], Σ),obsgauss), 2.0) 
# density_product(MvNormal([1.0], 2.0*Σ), obsgauss)
# logconstant(MvNormalScaledDensityProduct(MvNormal([1.0], Σ),obsgauss), 2.0)

# logpdf(MvNormal([1.0], 2.0*Σ), [1.0]) + logpdf(obsgauss, [1.0]) - 
# logpdf(distr(MvNormalScaledDensityProduct(MvNormal([1.0], Σ),obsgauss), 2.0), [1.0]) -
# logconstant(MvNormalScaledDensityProduct(MvNormal([1.0], Σ),obsgauss), 2.0)

# mvec = MVector{1, Float64}([20.0])

# logpdf(MvNormal(mvec, 2.0*Σ), [0.5]) + logpdf(obsgauss, [0.5]) - 
# logpdf(distr(MvNormalScaledDensityProduct(MvNormal([0.0], Σ),obsgauss), 2.0, mvec), [0.5]) -
# logconstant(MvNormalScaledDensityProduct(MvNormal([0.0], Σ),obsgauss), 2.0, mvec)

d = 5
Σ = diagm(ones(d)) # latent noise matrix
Ω = diagm(ones(d)) # observation noise matrix
ν = 4.0

latentgauss = MvNormal(zeros(d), Σ)
latentscale = Chisq(ν) # latent scale = multivariate t with ν degrees of freedom
obsgauss = MvNormal(zeros(d), Ω)

# function samplescalenormalmix(scale::Distribution{Univariate}, mvn::MvNormal, rng)
#     # zero mean
#     z = rand(rng, mvn)
#     s = sqrt(scale.ν / rand(rng, scale))
#     return z * s
# end

function f(x, t::Int64)
    # Adapted from: Monte Carlo Filter and Smoother for Non-Gaussian Nonlinear State Space Models
    d = length(x)
    ( 0.5 .* x) .+ (25 .* x ./ (1 .+ x .^ 2)) .+ (8 .* cos.(1.2 .* ((d:-1:1) ./ d) .* t))
end

# observations
n = 10
x = [zeros(d) for _ in 1:n]
y = [zeros(d) for _ in 1:n]
for t in 1:n
  if t == 1
    s = sqrt(latentscale.ν / rand(Random.GLOBAL_RNG, latentscale))
    x[t] = s*rand(Random.GLOBAL_RNG, latentgauss)
  else
    s = sqrt(latentscale.ν / rand(Random.GLOBAL_RNG, latentscale))
    x[t] = f(x[t-1], t) .+ s*rand(Random.GLOBAL_RNG, latentgauss)
  end
  y[t] = x[t] .+ rand(obsgauss)
end


twistedgauss = [MvNormalScaledDensityProduct(latentgauss, MvNormal(y[t], Ω)) for t in 1:n]


struct MVFloat64Particle{d}
  x::MVector{d, Float64}
  MVFloat64Particle{d}() where d = new(MVector{d, Float64}(undef))
end

function Base.:(==)(x::MVFloat64Particle{d}, y::MVFloat64Particle{d}) where d
  return x.x == y.x
end

function M_BPF!(newParticle::MVFloat64Particle{d}, rng, p::Int64, particle::MVFloat64Particle{d}, ::Nothing) where d
  if p == 1
    s = sqrt(latentscale.ν / rand(rng, latentscale))
    newParticle.x .= s * rand(rng, latentgauss)
  else
    s = sqrt(latentscale.ν / rand(rng, latentscale))
    newParticle.x .= f(particle.x, p) .+ s * rand(rng, latentgauss)
  end
end

# potential function
function logG_BPF(p::Int64, particle::MVFloat64Particle{d}, ::Nothing) where d
  return logpdf(obsgauss, particle.x - y[p])
end

model_BPF = SMCModel(M_BPF!, logG_BPF, n, MVFloat64Particle{d}, Nothing)
smcio_BPF = SMCIO{model_BPF.particle, model_BPF.pScratch}(2^10, n, 1, true, 0.5)

smc!(model_BPF, smcio_BPF)
SequentialMonteCarlo.V(smcio_BPF, (x) -> 1, true, false, n)

smcio_BPF.logZhats[end]
smcio_BPF.resample
smcio_BPF.esses


struct MVRFloat64Particle{d}
  x::MVector{d, Float64}
  s::MVector{1, Float64}
  MVRFloat64Particle{d}() where d = new(MVector{d, Float64}(undef), MVector{1, Float64}(undef))
end

function Base.:(==)(x::MVRFloat64Particle{d}, y::MVRFloat64Particle{d}) where d
  return (x.x == y.x) && (x.s == y.s)
end


function M_KPF!(newParticle::MVRFloat64Particle{d}, rng, p::Int64, particle::MVRFloat64Particle{d}, ::Nothing) where d
  if p == 1
    newParticle.s .= latentscale.ν / rand(rng, latentscale)
    newParticle.x .= zeros(d) # dummy for use at time 2
  else
    newParticle.s .= latentscale.ν / rand(rng, latentscale)
    newParticle.x .= f(rand(rng, distr(twistedgauss[p-1], particle.s[1], particle.x)), p) # use old scale!
  end
end

function logG_KPF(p::Int64, particle::MVRFloat64Particle{d}, ::Nothing) where d
  return logconstant(twistedgauss[p], particle.s[1], particle.x)
end

model_KPF = SMCModel(M_KPF!, logG_KPF, n, MVRFloat64Particle{d}, Nothing)
smcio_KPF = SMCIO{model_KPF.particle, model_KPF.pScratch}(2^10, n, 1, true, 0.5)

smc!(model_KPF, smcio_KPF)
SequentialMonteCarlo.V(smcio_KPF, (x) -> 1, true, false, n)

smcio_KPF.logZhats[end]


reps = 200
res = Matrix{Float64}(undef,reps,2)

for i in 1:reps
  smc!(model_BPF, smcio_BPF)
  smc!(model_KPF, smcio_KPF)
  res[i,:] = [smcio_BPF.logZhats[end], smcio_KPF.logZhats[end]]
end


std(res[:,1]), std(res[:,2])


logsumexp(res[:,1]) - log(reps)
logsumexp(res[:,2]) - log(reps)

SequentialMonteCarlo.V(smcio_BPF, (x) -> 1, true, false, n)
SequentialMonteCarlo.V(smcio_KPF, (x) -> 1, true, false, n)




# student = MvTdistr(ν, zeros(d), Σ)

# st_test = [samplescalenormalmix(latentscale, latentgauss, Random.GLOBAL_RNG)[1] for _ in 1:100000]
# st_comp = [rand(student)[1] for _ in 1:100000]

# mean(st_test), mean(st_comp)
# var(st_test), var(st_comp)
# quantile(st_test, 0.25), quantile(st_comp, 0.25)
# quantile(st_test, 0.5), quantile(st_comp, 0.5)
# quantile(st_test, 0.75), quantile(st_comp, 0.75)    