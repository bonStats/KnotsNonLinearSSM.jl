using LinearAlgebra
using Distributions
using SequentialMonteCarlo
using StaticArrays
using Random

d = 1
Σ = diagm(ones(d)) # latent noise matrix
Ω = diagm(ones(d)) # observation noise matrix
ν = 4.0

latentgauss = MvNormal(zeros(d), Σ)
latentscale = Chisq(ν)
obsgauss = MvNormal(zeros(d), Ω)

function samplemvstudent(scale::Distribution{Univariate}, mvn::MvNormal, rng)
    # zero mean
    z = rand(rng, mvn)
    s = sqrt(latentscale.ν / rand(rng, scale))
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
    x[t] = samplemvstudent(latentscale, latentgauss, Random.GLOBAL_RNG)
  else
    x[t] = f(x[t-1], t) .+ samplemvstudent(latentscale, latentgauss, Random.GLOBAL_RNG)
  end
  y[t] = x[t] .+ rand(obsgauss)
end

struct MVFloat64Particle{d}
  x::MVector{d, Float64}
  MVFloat64Particle{d}() where d = new(MVector{d, Float64}(undef))
end

function Base.:(==)(x::MVFloat64Particle{d}, y::MVFloat64Particle{d}) where d
  return x.x == y.x
end

function M!(newParticle::MVFloat64Particle{d}, rng, p::Int64, particle::MVFloat64Particle{d}, ::Nothing) where d
  if p == 1
    newParticle.x .= samplemvstudent(latentscale, latentgauss, rng)
  else
    newParticle.x .= f(particle.x, p) .+ samplemvstudent(latentscale, latentgauss, rng)
  end
end

# potential function
function logG(p::Int64, particle::MVFloat64Particle{d}, ::Nothing) where d
  return logpdf(obsgauss, particle.x - y[p])
end

model = SMCModel(M!, logG, n, MVFloat64Particle{d}, Nothing)
smcio = SMCIO{model.particle, model.pScratch}(2^10, n, 1, true, 0.5)

smc!(model, smcio)

smcio.resample
smcio.esses


# student = MvTDist(ν, zeros(d), Σ)

# st_test = [samplemvstudent(latentscale, latentgauss, Random.GLOBAL_RNG)[1] for _ in 1:100000]
# st_comp = [rand(student)[1] for _ in 1:100000]

# mean(st_test), mean(st_comp)
# var(st_test), var(st_comp)
# quantile(st_test, 0.25), quantile(st_comp, 0.25)
# quantile(st_test, 0.5), quantile(st_comp, 0.5)
# quantile(st_test, 0.75), quantile(st_comp, 0.75)    