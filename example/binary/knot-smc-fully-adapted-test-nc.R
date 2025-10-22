## Comparison of asymptotic variances for normalizing constant
## based on the counter example in 
## "A note on auxiliary particle filters"
## AM Johansen, A Doucet - Statistics & Probability Letters, 2008

library(plotly)
library(ggplot2)
library(dplyr)

adaptedknot_col <- "#648FFF"
fulladapt_col <- "#DC267F"
bpf_col <- "#FFB000"

N_reps <- 50000

# functions to calculate asymptotic variance of canonical Feynman-Kac model

sample01 <- function(N, p1){
  n1 <- rbinom(1, size = N, prob = p1)
  n0 <- N - n1
  setNames(object = c(n0,n1), nm = c("0","1"))
}

resample_prob1 <- function(n01, p1){
  p1*n01["1"] / (p1*n01["1"] + (1-p1)*n01["0"])
}

mutate <- function(prt, p_nc){ # p no change
  n11 <-rbinom(1, size = prt["1"], prob = p_nc)
  n10 <-prt["1"] - n11
  n00 <-rbinom(1, size = prt["0"], prob = p_nc)
  n01 <-prt["0"] - n00
  setNames(object = c(n00+n10,n11+n01), nm = c("0","1"))
}

mutate2 <- function(prt, p_nc0, p_nc1){ # p no change
  n11 <-rbinom(1, size = prt["1"], prob = p_nc1)
  n10 <-prt["1"] - n11
  n00 <-rbinom(1, size = prt["0"], prob = p_nc0)
  n01 <-prt["0"] - n00
  setNames(object = c(n00+n10,n11+n01), nm = c("0","1"))
}

### 

bpf_update <- new.env()

with(bpf_update,{
  
  
  gamma0_1 <- function(delta,epsilon){
    1
  }
  
  gamma1_1 <- function(delta,epsilon){
    1/2
  }
  
  gamma1_G1 <- function(delta,epsilon){
    (1-delta) * epsilon * (1-epsilon) + (delta/2)*(epsilon^2 + (1-epsilon)^2)
  }
  
  eta1_G1 <- function(delta,epsilon){
    gamma1_G1(delta,epsilon) / gamma1_1(delta,epsilon)
  }
  
  gamma0Q01_G1 <- function(delta,epsilon){
    0.5 * ((1-epsilon)^2) * ((delta*(1-epsilon) + (1-delta)*epsilon)^2) + 
      0.5 * (epsilon^2) * (((1-delta)*(1-epsilon) + delta*epsilon)^2)
  }
  
  gamma1Q11_G1 <- function(delta,epsilon){
    0.5 * (epsilon^2) * ( (1-delta)*(1-epsilon) + delta*epsilon ) + 
      0.5 * ((1-epsilon)^2) * (delta*(1-epsilon) + (1-delta)*epsilon)
      
  }
  
  
  v01 <- function(delta,epsilon){
    
    (gamma0_1(delta,epsilon)*gamma0Q01_G1(delta,epsilon) - (gamma1_G1(delta,epsilon)^2) ) / (gamma1_1(delta,epsilon)^2)
    
  }
  
  v11 <- function(delta,epsilon){
    
    (gamma1_1(delta,epsilon)*gamma1Q11_G1(delta,epsilon) - (gamma1_G1(delta,epsilon)^2) ) / (gamma1_1(delta,epsilon)^2) 
    
  }
  
  # hat sigma2
  sigma2 <- function(delta,epsilon){
    (v01(delta,epsilon) + v11(delta,epsilon)) / (eta1_G1(delta,epsilon)^2)
  }
  
  nc_estimate <- function(delta, epsilon, prt0, prt1, N){
    logG0 <- log((1-epsilon)*prt0["0"] + epsilon*prt0["1"]) - log(N)
    logG1 <- log(epsilon*prt1["0"] + (1 - epsilon)*prt1["1"]) - log(N)
    exp(logG0 + logG1)
  }
  
  normal_nc_estimate <- function(delta, epsilon, prt0, prt1, N){
    # normalise to correspond asymptotic variance calculation  
    nc_estimate(delta, epsilon, prt0, prt1, N) / gamma1_G1(delta,epsilon)
  }
  
  sampler_exp <- function(N, delta, epsilon){
    
    prt0 <- sample01(N, p1 = 1/2) # initial
    p0_1 <- resample_prob1(prt0, epsilon) # weight prob x=1 t=0
    prt0_hat <- sample01(N, p1 = p0_1) # resample 0
    prt1 <- mutate(prt0_hat, 1-delta) # mutate t=1

    normal_nc_estimate(delta, epsilon, prt0, prt1, N)
  }
  
  est_asy_var <- function(N, delta, epsilon){
    N * var(replicate(N_reps, {sampler_exp(N = N, delta, epsilon)}))
  }
  
})

full_adapt_update <- new.env()

with(full_adapt_update,{
  
  
  gamma0_1 <- function(delta,epsilon){
    1
  }
  
  gamma1_1 <- function(delta,epsilon){
    (1-delta) * epsilon * (1-epsilon) + (delta/2)*(epsilon^2 + (1-epsilon)^2)
  }
  
  gamma1_G1 <- function(delta,epsilon){
    gamma1_1(delta,epsilon)
  }
  
  eta1_G1 <- function(delta,epsilon){
    gamma1_G1(delta,epsilon) / gamma1_1(delta,epsilon)
  }
  
  gamma0Q01_G1 <- function(delta,epsilon){
    0.25 * ( (1-epsilon)*( (delta*(1-epsilon) + (1-delta)*epsilon)^2) + epsilon*( (delta*epsilon + (1-delta)*(1-epsilon) )^2) )
  }
  
  gamma1Q11_G1 <- function(delta,epsilon){
    gamma1_G1(delta,epsilon)  
  }
  
  
  v01 <- function(delta,epsilon){
    
    (gamma0_1(delta,epsilon)*gamma0Q01_G1(delta,epsilon) - gamma1_G1(delta,epsilon)^2)/(gamma1_1(delta,epsilon)^2)
    
  }
  
  v11 <- function(delta,epsilon){
    
    (gamma1_1(delta,epsilon)*gamma1Q11_G1(delta,epsilon) - gamma1_G1(delta,epsilon)^2)/(gamma1_1(delta,epsilon)^2)
    
  }
  
  # hat sigma
  sigma2 <- function(delta,epsilon){
    ( v01(delta,epsilon) +  v11(delta,epsilon) ) / (eta1_G1(delta,epsilon)^2)
  }
  
  G0_F <- function(delta, epsilon, value){
    if(value == 0){
      0.5 * ( delta * (1-epsilon) + (1-delta)*epsilon )
    } else if (value == 1) {
      0.5 * ( (1-delta)*(1-epsilon) + delta*epsilon )
    } else {
      NA
    }
  }
  
  nc_estimate <- function(delta, epsilon, prt0, prt1, N){
    logG0 <- log(G0_F(delta, epsilon, 0)*prt0["0"] + G0_F(delta, epsilon, 1)*prt0["1"]) - log(N)
    logG1 <- 0
    exp(logG0 + logG1)
  }
  
  normal_nc_estimate <- function(delta, epsilon, prt0, prt1, N){
    # normalise to correspond asymptotic variance calculation  
    nc_estimate(delta, epsilon, prt0, prt1, N) / gamma1_G1(delta,epsilon)
  }
  
  sampler_exp <- function(N, delta, epsilon){
    
    prt0 <- sample01(N, p1 = epsilon) # initial
    p0_1 <- resample_prob1(prt0, p1 = (1-epsilon)*(1-delta) + epsilon*delta) # weight prob x=1 t=0
    prt0_hat <- sample01(N, p1 = p0_1) # resample 0
    p_nc0 = (1-delta)*epsilon / ( (1-delta)*epsilon + delta*(1-epsilon) )
    p_nc1 = (1-delta)*(1-epsilon) / ((1-delta)*(1-epsilon) + delta*epsilon) 
    prt1 <- mutate2(prt0_hat, p_nc0 = p_nc0, p_nc1 = p_nc1) # mutate t=1
    
    normal_nc_estimate(delta, epsilon, prt0, prt1, N) # note: prt1 not used
  }
  
  est_asy_var <- function(N, delta, epsilon){
    N * var(replicate(N_reps, {sampler_exp(N = N, delta, epsilon)}))
  }
  
})

adapted_knotset_update <- new.env() # almost fully adapted

with(adapted_knotset_update,{
  
  
  gamma0_1 <- function(delta,epsilon){
    1
  }

  gamma1_1 <- function(delta,epsilon){
    0.5
  }
  
  gamma1_G1 <- function(delta,epsilon){
    (1-delta) * epsilon * (1-epsilon) + (delta/2)*(epsilon^2 + (1-epsilon)^2)
  }
  
  eta1_G1 <- function(delta,epsilon){
    gamma1_G1(delta,epsilon) / gamma1_1(delta,epsilon)
  }
  
  gamma0Q01_G1 <- function(delta,epsilon){
    gamma1_G1(delta,epsilon)^2
  }
  
  gamma1Q11_G1 <- function(delta,epsilon){
    0.5 * ( (1-epsilon)*( (delta*(1-epsilon) + (1-delta)*epsilon)^2) + epsilon*( (delta*epsilon + (1-delta)*(1-epsilon) )^2) )
  }
  
  
  v01 <- function(delta,epsilon){
    
    (gamma0_1(delta,epsilon)*gamma0Q01_G1(delta,epsilon) - gamma1_G1(delta,epsilon)^2)/(gamma1_1(delta,epsilon)^2)
    
  }
  
  v11 <- function(delta,epsilon){
    
    (gamma1_1(delta,epsilon)*gamma1Q11_G1(delta,epsilon) - gamma1_G1(delta,epsilon)^2)/(gamma1_1(delta,epsilon)^2)
    
  }
  
  sigma2 <- function(delta,epsilon){
    ( v01(delta,epsilon) +  v11(delta,epsilon) ) / ( eta1_G1(delta,epsilon)^2 )
  }
  
  G1_ast <- function(delta, epsilon, value){
    if(value == 0){
      delta * (1-epsilon) + (1-delta)*epsilon
    } else if (value == 1) {
      (1-delta)*(1-epsilon) + delta*epsilon
    } else {
      NA
    }
  }
  
  nc_estimate <- function(delta, epsilon, prt0, prt1, N){
    logG0 <- log(0.5) # constant
    logG1 <- log(G1_ast(delta, epsilon, 0)*prt0["0"] + G1_ast(delta, epsilon, 1)*prt0["1"]) - log(N)
    exp(logG0 + logG1)
  }
  
  normal_nc_estimate <- function(delta, epsilon, prt0, prt1, N){
    # normalise to correspond asymptotic variance calculation  
    nc_estimate(delta, epsilon, prt0, prt1, N) / gamma1_G1(delta,epsilon)
  }
  
  sampler_exp <- function(N, delta, epsilon){
    
    prt0 <- sample01(N, p1 = epsilon) # initial
    prt1 <- mutate(prt0, p_nc = 1-delta) # mutate t=1
    #p1_1 <- resample_prob1(prt1, 1-epsilon) # weight x=1 t=1
    
    normal_nc_estimate(delta, epsilon, prt0, prt1, N)
    
  }
  
  est_asy_var <- function(N, delta, epsilon){
    N * var(replicate(N_reps, {sampler_exp(N = N, delta, epsilon)}))
  }
  
})

# Targets the same probability:

N <- 100000
dval <- 0.22
eval <- 0.18

bpf_update$est_asy_var(N, dval, eval)
bpf_update$sigma2(dval,eval)

full_adapt_update$est_asy_var(N, dval, eval)
full_adapt_update$sigma2(dval,eval)

adapted_knotset_update$est_asy_var(N, dval, eval)
adapted_knotset_update$sigma2(dval,eval)

full_adapt_update$v01(dval, eval)
full_adapt_update$v11(dval, eval)


# theory
setNames(
  c(adapted_knotset_update$gamma1_G1(dval,eval),
   full_adapt_update$gamma1_G1(dval,eval),
   bpf_update$gamma1_G1(dval,eval)),
  c("Terminal adapted knotset","'Full' adaptation","Bootstrap"))

# large sample
setNames(
  c(adapted_knotset_update$sampler_exp(N,dval,eval),
   full_adapt_update$sampler_exp(N,dval,eval),
   bpf_update$sampler_exp(N,dval,eval)),
  c("Terminal adapted knotset","'Full' adaptation","Bootstrap"))  


# Test empirical versus theoretical asymptotic variance

N <- 100000
eval <- 0.25
dvals <- c(0.01,0.05, seq(0.1,0.9,length.out = 9), 0.95, 0.99)

bpf_est_asy_var <- data.frame(
  y=sapply(dvals, function(d) bpf_update$est_asy_var(N, d, eval)), 
  x=dvals,
  pf = "Bootstrap",
  N = N)

full_adapt_est_asy_var <- data.frame(
  y=sapply(dvals, function(d) full_adapt_update$est_asy_var(N, d, eval)), 
  x=dvals,
  pf = "'Full' adaptation",
  N = N)

adapted_knotset_est_asy_var <- data.frame(
  y=sapply(dvals, function(d) adapted_knotset_update$est_asy_var(N, d, eval)), 
  x=dvals,
  pf = "Terminal adapted knotset",
  N = N)

est_asy_var <- bind_rows(bpf_est_asy_var, full_adapt_est_asy_var, adapted_knotset_est_asy_var)

ggplot(aes(x=x,y=y,colour=pf),data=est_asy_var) +
  geom_point() + 
  geom_function(fun = function(d) bpf_update$sigma2(delta = d,epsilon = eval), colour = bpf_col) +
  geom_function(fun = function(d) full_adapt_update$sigma2(delta = d,epsilon = eval), colour = fulladapt_col) +
  geom_function(fun = function(d) adapted_knotset_update$sigma2(delta = d,epsilon = eval), colour = adaptedknot_col ) +
  scale_x_continuous("delta", breaks = seq(0,1,length.out = 11)) +
  scale_color_manual("Particle filter type", values = c(Bootstrap = bpf_col, `'Full' adaptation` = fulladapt_col, `Terminal adapted knotset` = adaptedknot_col)) + 
  ggtitle(paste0("Performance of particle filters, epsilon = ",eval))  +
  ylab("Asymptotic variance") + 
  theme_bw()

## Figure 4 paper: export 3.25 x 5 inches landscape

pf_label_order <- function(x) ordered(x, levels = c("Bootstrap", "'Full' adaptation", "Terminal adapted knotset"))

est_asy_var$pf <- pf_label_order(est_asy_var$pf)

Nval <- unique(est_asy_var$N)
eval <- 0.25
ggplot(aes(x=x,y=y), data=est_asy_var) +
  geom_point(aes(shape = pf)) + 
  geom_function(aes(linetype = pf_label_order("Bootstrap")), fun = function(d) bpf_update$sigma2(delta = d,epsilon = eval)) +
  geom_function(aes(linetype = pf_label_order("'Full' adaptation")), fun = function(d) full_adapt_update$sigma2(delta = d,epsilon = eval)) +
  geom_function(aes(linetype = pf_label_order("Terminal adapted knotset")), fun = function(d) adapted_knotset_update$sigma2(delta = d,epsilon = eval)) +
  scale_x_continuous(expression(delta), breaks = seq(0,1,length.out = 11), labels = scales::label_number(drop0trailing=TRUE)) +
  scale_shape_discrete(paste0("N = ",format(Nval,scientific = F, big.mark = ",")), solid = FALSE) + 
  scale_linetype_discrete(expression(N %->% infinity)) + 
  ylab("Variance") + 
  theme_bw()

