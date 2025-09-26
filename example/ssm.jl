using LinearAlgebra
using Distributions
using SequentialMonteCarlo
using StaticArrays
using Random

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

dist(dist::MvNormalScaledDensityProduct, s₁::Float64, s₂::Float64) = MvNormalCanon( (dist.h₁ / s₁) + (dist.h₂ / s₂), (dist.Ω₁ / s₁) + (dist.Ω₂ / s₂))
dist(dist::MvNormalScaledDensityProduct, s₁::Float64) = dist(d, s₁, 1.0)

dist(dist::MvNormalScaledDensityProduct, s₁::Float64, x₁::MVector{d, Float64}) where d = MvNormalCanon( ((dist.h₁ + (dist.Σ₁ \ x₁))/ s₁) + (dist.h₂ / s₂), (dist.Ω₁ / s₁) + (dist.Ω₂ / s₂))

logconstant(dist::MvNormalScaledDensityProduct, s₁::Float64, s₂::Float64) = begin
    constMvN = MvNormal(dist.Δμ, (dist.Σ₁ * s₁) + (dist.Σ₂ * s₂)) # constant multiplier
    return logpdf(constMvN, zeros(length(dist.Δμ)))
end
logconstant(d::MvNormalScaledDensityProduct, s₁::Float64) = logconstant(d, s₁, 1.0)

logconstant(dist::MvNormalScaledDensityProduct, s₁::Float64, x₁::MVector{d, Float64}) = begin
    constMvN = MvNormal(x₁ + dist.Δμ, (dist.Σ₁ * s₁) + dist.Σ₂) # constant multiplier
    return logpdf(constMvN, zeros(length(dist.Δμ)))
end

### CHECK x_1 version AGAINST DENSITY PRODUCT

# dist(MvNormalScaledDensityProduct(latentgauss,obsgauss), 1.0) # should be the same as density_product(latentgauss, obsgauss)
# dist(MvNormalScaledDensityProduct(MvNormal([1.0], Σ),obsgauss), 2.0) 
# density_product(MvNormal([1.0], 2.0*Σ), obsgauss)
# logconstant(MvNormalScaledDensityProduct(MvNormal([1.0], Σ),obsgauss), 2.0)

# logpdf(MvNormal([1.0], 2.0*Σ), [1.0]) + logpdf(obsgauss, [1.0]) - 
# logpdf(dist(MvNormalScaledDensityProduct(MvNormal([1.0], Σ),obsgauss), 2.0), [1.0]) -
# logconstant(MvNormalScaledDensityProduct(MvNormal([1.0], Σ),obsgauss), 2.0)

d = 1
Σ = diagm(ones(d)) # latent noise matrix
Ω = diagm(ones(d)) # observation noise matrix
ν = 4.0

latentgauss = MvNormal(zeros(d), Σ)
latentscale = Chisq(ν) # latent scale = multivariate t with ν degrees of freedom
obsgauss = MvNormal(zeros(d), Ω)

function samplescalenormalmix(scale::Distribution{Univariate}, mvn::MvNormal, rng)
    # zero mean
    z = rand(rng, mvn)
    s = sqrt(scale.ν / rand(rng, scale))
    return z * s
end

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
    x[t] = samplescalenormalmix(latentscale, latentgauss, Random.GLOBAL_RNG)
  else
    x[t] = f(x[t-1], t) .+ samplescalenormalmix(latentscale, latentgauss, Random.GLOBAL_RNG)
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
    newParticle.x .= samplescalenormalmix(latentscale, latentgauss, rng)
  else
    newParticle.x .= f(particle.x, p) .+ samplescalenormalmix(latentscale, latentgauss, rng)
  end
end

# potential function
function logG_BPF(p::Int64, particle::MVFloat64Particle{d}, ::Nothing) where d
  return logpdf(obsgauss, particle.x - y[p])
end

model_BPF = SMCModel(M_BPF!, logG_BPF, n, MVFloat64Particle{d}, Nothing)
smcio_BPF = SMCIO{model_BPF.particle, model_BPF.pScratch}(2^10, n, 1, true, 0.5)

smc!(model_BPF, smcio_BPF)

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
    newParticle.s .= sqrt(latentscale.ν / rand(rng, latentscale))
    newParticle.x .= zeros(d) # dummy for use at time 2
  else
    newParticle.s .= sqrt(latentscale.ν / rand(rng, latentscale))
    newParticle.x .= f(rand(dist(twistedgauss[p-1], particle.s, particle.x), rng), p) # use old scale!
  end
end

function logG_KPF(p::Int64, particle::MVRFloat64Particle{d}, ::Nothing) where d
  return logconstant(twistedgauss[p], particle.s, particle.x)
end


# student = MvTDist(ν, zeros(d), Σ)

# st_test = [samplescalenormalmix(latentscale, latentgauss, Random.GLOBAL_RNG)[1] for _ in 1:100000]
# st_comp = [rand(student)[1] for _ in 1:100000]

# mean(st_test), mean(st_comp)
# var(st_test), var(st_comp)
# quantile(st_test, 0.25), quantile(st_comp, 0.25)
# quantile(st_test, 0.5), quantile(st_comp, 0.5)
# quantile(st_test, 0.75), quantile(st_comp, 0.75)    