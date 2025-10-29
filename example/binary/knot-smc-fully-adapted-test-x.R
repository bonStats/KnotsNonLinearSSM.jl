## Comparison of asymptotic variances for test function x
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
  
  hat_gamma1_1 <- function(delta,epsilon){
    (1-delta) * epsilon * (1-epsilon) + (delta/2)*(epsilon^2 + (1-epsilon)^2)
  }
  
  gamma1_G1 <- function(delta,epsilon){
    hat_gamma1_1(delta,epsilon)
  }
  
  # phi = \bar{\phi}
  hat_gamma1_phi <- function(delta,epsilon){
    0.5*((1-delta) * epsilon + delta*(1-epsilon)) * (1-epsilon)
  }
  
  hat_eta1_phi <- function(delta,epsilon){
    hat_gamma1_phi(delta,epsilon) / hat_gamma1_1(delta,epsilon)
  }
  
  
  gamma0Q01_barphi <- function(delta,epsilon){
    0.5* ( ((1-epsilon)^2)*delta*(1-hat_eta1_phi(delta,epsilon)) - (1-epsilon)*(1-delta)*epsilon*hat_eta1_phi(delta,epsilon) )^2 + 
      0.5* (epsilon*(1-delta)*(1-epsilon)*(1-hat_eta1_phi(delta,epsilon)) - (epsilon^2)*delta*hat_eta1_phi(delta,epsilon) )^2
  }
  
  gamma1Q11_barphi <- function(delta,epsilon){
    0.5*((1-delta)*(1-epsilon) + delta*epsilon) * (epsilon^2) * (hat_eta1_phi(delta,epsilon)^2) +
      0.5*((1-delta)*epsilon + delta*(1-epsilon)) * ((1-epsilon)^2)*((1-hat_eta1_phi(delta,epsilon))^2)
  }
  
  gamma1_barphi <- function(delta,epsilon){
    0.5*((1-delta)*epsilon + delta*(1-epsilon)) * (1-epsilon) * (1-hat_eta1_phi(delta,epsilon)) -
      0.5*((1-delta)*(1-epsilon) + delta*epsilon) * epsilon * hat_eta1_phi(delta,epsilon) 
  }
  
  
  v01 <- function(delta,epsilon){
    
    (gamma0_1(delta,epsilon)*gamma0Q01_barphi(delta,epsilon) - gamma1_barphi(delta,epsilon)^2)/(gamma1_G1(delta,epsilon)^2)
    
  }
  
  v11 <- function(delta,epsilon){
    
    (gamma1_1(delta,epsilon)*gamma1Q11_barphi(delta,epsilon) - gamma1_barphi(delta,epsilon)^2)/(gamma1_G1(delta,epsilon)^2)
    
  }
  
  sigma2 <- function(delta,epsilon){
    v01(delta,epsilon) +  v11(delta,epsilon)
  }
  
  sampler_exp <- function(N, delta, epsilon){
    
    prt0 <- sample01(N, p1 = 1/2) # initial
    p0_1 <- resample_prob1(prt0, epsilon) # weight prob x=1 t=0
    prt0_hat <- sample01(N, p1 = p0_1) # resample 0
    prt1 <- mutate(prt0_hat, 1-delta) # mutate t=1
    p1_1 <- resample_prob1(prt1, 1-epsilon) # weight x=1 t=1
    p1_1
  }
  
  est_asy_var <- function(N, delta, epsilon){
    N * var(replicate(N_reps, {sampler_exp(N = N, delta, epsilon)}))
  }
  
})

full_adapt_update <- new.env() # same as apf_predict since G_1 = 1

with(full_adapt_update,{
  
  
  gamma0_1 <- function(delta,epsilon){
    1
  }
  
  gamma1_1 <- function(delta,epsilon){
    (1-delta) * epsilon * (1-epsilon) + (delta/2)*(epsilon^2 + (1-epsilon)^2)
  }
  
  gamma1_phi <- function(delta,epsilon){
    0.5*((1-delta) * epsilon + delta*(1-epsilon)) * (1-epsilon)
  }
  
  eta1_phi <- function(delta,epsilon){
    gamma1_phi(delta,epsilon) / gamma1_1(delta,epsilon)
  }
  
  gamma1_phiprime <- function(delta,epsilon){
    0.5*((1-delta) * epsilon + delta*(1-epsilon)) * (1-epsilon) * (1-eta1_phi(delta,epsilon)) - 
      0.5*((1-delta)*(1-epsilon) + delta*epsilon) * epsilon * eta1_phi(delta,epsilon)
  }
  
  
  gamma0Q01_phiprime <- function(delta,epsilon){
    0.25 * (1-epsilon) * (delta*(1-epsilon)*(1-eta1_phi(delta,epsilon)) - (1-delta)*epsilon*eta1_phi(delta,epsilon) )^2 + 
      0.25 * epsilon * ((1-delta)*(1-epsilon)*(1-eta1_phi(delta,epsilon)) - delta*epsilon*eta1_phi(delta,epsilon))^2
  }
  
  gamma1Q11_phiprime <- function(delta,epsilon){
    0.5*((1-delta)*(1-epsilon) + delta*epsilon) * epsilon * (eta1_phi(delta,epsilon)^2) +
      0.5*((1-delta) * epsilon + delta*(1-epsilon)) * (1-epsilon) * ((1-eta1_phi(delta,epsilon))^2)
  }
  
  
  v01 <- function(delta,epsilon){
    
    (gamma0_1(delta,epsilon)*gamma0Q01_phiprime(delta,epsilon) - gamma1_phiprime(delta,epsilon)^2)/(gamma1_1(delta,epsilon)^2)
    
  }
  
  v11 <- function(delta,epsilon){
    
    (gamma1_1(delta,epsilon)*gamma1Q11_phiprime(delta,epsilon) - gamma1_phiprime(delta,epsilon)^2)/(gamma1_1(delta,epsilon)^2)
    
  }
  
  sigma2 <- function(delta,epsilon){
    v01(delta,epsilon) +  v11(delta,epsilon)
  }
  
  sampler_exp <- function(N, delta, epsilon){
    
    prt0 <- sample01(N, p1 = epsilon) # initial
    p0_1 <- resample_prob1(prt0, p1 = (1-epsilon)*(1-delta) + epsilon*delta) # weight prob x=1 t=0
    prt0_hat <- sample01(N, p1 = p0_1) # resample 0
    p_nc0 = (1-delta)*epsilon / ( (1-delta)*epsilon + delta*(1-epsilon) )
    p_nc1 = (1-delta)*(1-epsilon) / ((1-delta)*(1-epsilon) + delta*epsilon) 
    prt1 <- mutate2(prt0_hat, p_nc0 = p_nc0, p_nc1 = p_nc1) # mutate t=1
    p1_1 <- resample_prob1(prt1, 1/2) # weight x=1 t=1
    p1_1
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
    1/2
  }
  
  hat_gamma1_1 <- function(delta,epsilon){
    (1-delta) * epsilon * (1-epsilon) + (delta/2)*(epsilon^2 + (1-epsilon)^2)
  }
  
  gamma1_G1 <- function(delta,epsilon){
    hat_gamma1_1(delta,epsilon)
  }
  
  # phi = \bar{\phi}
  hat_gamma1_phi <- function(delta,epsilon){
    0.5*((1-delta) * epsilon + delta*(1-epsilon)) * (1-epsilon)
  }
  
  hat_eta1_phi <- function(delta,epsilon){
    hat_gamma1_phi(delta,epsilon) / hat_gamma1_1(delta,epsilon)
  }
  
  
  gamma0Q01_barphi <- function(delta,epsilon){
    (0.5*((1-delta) * epsilon + delta*(1-epsilon)) * (1-epsilon) * (1-hat_eta1_phi(delta,epsilon)) - 
       0.5*((1-delta)*(1-epsilon) + delta*epsilon) * epsilon * hat_eta1_phi(delta,epsilon) )^2
  }
  
  gamma1Q11_barphi <- function(delta,epsilon){
    0.5*((1-delta)*(1-epsilon) + delta*epsilon) * (epsilon^2) * (hat_eta1_phi(delta,epsilon)^2) +
      0.5*((1-delta)*epsilon + delta*(1-epsilon)) * ((1-epsilon)^2)*((1-hat_eta1_phi(delta,epsilon))^2)
  }
  
  gamma1_barphi <- function(delta,epsilon){
    0.5*((1-delta)*epsilon + delta*(1-epsilon)) * (1-epsilon) * (1-hat_eta1_phi(delta,epsilon)) -
      0.5*((1-delta)*(1-epsilon) + delta*epsilon) * epsilon * hat_eta1_phi(delta,epsilon) 
  }
  
  
  v01 <- function(delta,epsilon){
    
    (gamma0_1(delta,epsilon)*gamma0Q01_barphi(delta,epsilon) - gamma1_barphi(delta,epsilon)^2)/(gamma1_G1(delta,epsilon)^2)
    
  }
  
  v11 <- function(delta,epsilon){
    
    (gamma1_1(delta,epsilon)*gamma1Q11_barphi(delta,epsilon) - gamma1_barphi(delta,epsilon)^2)/(gamma1_G1(delta,epsilon)^2)
    
  }
  
  sigma2 <- function(delta,epsilon){
    v01(delta,epsilon) +  v11(delta,epsilon)
  }
  
  sampler_exp <- function(N, delta, epsilon){
    
    prt0 <- sample01(N, p1 = epsilon) # initial
    prt1 <- mutate(prt0, p_nc = 1-delta) # mutate t=1
    p1_1 <- resample_prob1(prt1, 1-epsilon) # weight x=1 t=1
    p1_1
  }
  
  est_asy_var <- function(N, delta, epsilon){
    N * var(replicate(N_reps, {sampler_exp(N = N, delta, epsilon)}))
  }
  
})

# Targets the same probability:

N <- 100000
dval <- 0.22
eval <- 0.18

# theory
setNames(
  c(adapted_knotset_update$hat_eta1_phi(dval,eval),
   full_adapt_update$eta1_phi(dval,eval),
   bpf_update$hat_eta1_phi(dval,eval)),
  c("Adapted knotset","'Full' adaptation","Bootstrap"))

# large sample
setNames(
  c(adapted_knotset_update$sampler_exp(N,dval,eval),
   full_adapt_update$sampler_exp(N,dval,eval),
   bpf_update$sampler_exp(N,dval,eval)),
  c("Adapted knotset","'Full' adaptation","Bootstrap"))  


# Recreate Figure 2 from AM Johansen, A Doucet (2008)

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
  pf = "Adapted knotset",
  N = N)

est_asy_var_jd2008 <- bind_rows(bpf_est_asy_var, full_adapt_est_asy_var, adapted_knotset_est_asy_var)

ggplot(aes(x=x,y=y,colour=pf),data=est_asy_var_jd2008) +
  geom_point() + 
  geom_function(fun = function(d) bpf_update$sigma2(delta = d,epsilon = eval), colour = bpf_col) +
  geom_function(fun = function(d) full_adapt_update$sigma2(delta = d,epsilon = eval), colour = fulladapt_col) +
  geom_function(fun = function(d) adapted_knotset_update$sigma2(delta = d,epsilon = eval), colour = adaptedknot_col ) +
  scale_x_continuous("delta", breaks = seq(0,1,length.out = 11)) +
  scale_color_manual("Particle filter type", values = c(Bootstrap = bpf_col, `'Full' adaptation` = fulladapt_col, `Adapted knotset` = adaptedknot_col)) + 
  ggtitle(paste0("Performance of particle filters, epsilon = ",eval))  +
  ylab("Asymptotic variance") + 
  theme_bw()

# Recreate Figure 1 from AM Johansen, A Doucet (2008)

eval <- seq(from=0.05,to=0.5,length.out = 500)
dval <- seq(from=0.05,to=0.5,length.out = 500) # it appears the delta axis is labelled from 0 to 1 (instead of 0.05 to 0.5)

zval <- outer(dval,eval,function(delta,epsilon) full_adapt_update$sigma2(delta,epsilon) - bpf_update$sigma2(delta,epsilon))
plot_ly(x = dval, y = eval, z = zval) %>% add_surface()



# Look at differences: 'Full' adaptation vs BPF

eval <- seq(from=0.05,to=0.99,length.out = 500)
dval <- seq(from=0.05,to=0.99,length.out = 500)

zval <- outer(dval,eval,function(delta,epsilon) full_adapt_update$sigma2(delta,epsilon) - bpf_update$sigma2(delta,epsilon))
plot_ly(x = dval, y = eval, z = zval) %>% add_surface()

# Look at differences: Adapted knotset vs BPF

eval <- seq(from=0.01,to=0.99,length.out = 500)
dval <- seq(from=0.01,to=0.99,length.out = 500)

zval <- outer(dval,eval,function(delta,epsilon) adapted_knotset_update$sigma2(delta,epsilon) - bpf_update$sigma2(delta,epsilon))
plot_ly(x = dval, y = eval, z = zval) %>% add_surface()

# Look at differences: Adapted knotset vs 'Full' adaptation

eval <- seq(from=0.01,to=0.99,length.out = 500)
dval <- seq(from=0.01,to=0.99,length.out = 500)

zval <- outer(dval,eval,function(delta,epsilon) adapted_knotset_update$sigma2(delta,epsilon) - full_adapt_update$sigma2(delta,epsilon))
plot_ly(x = dval, y = eval, z = zval) %>% add_surface()



# Redux Figure 2 from AM Johansen, A Doucet (2008) ***epsilon = 0.01***

N <- 100000
eval <- 0.01
dvals <- c(0.01,0.05, seq(0.1,0.9,length.out = 9), 0.95, 0.99)

bpf_est_asy_var <- data.frame(
  y=sapply(dvals, function(d) bpf_update$est_asy_var(N, d, eval)), 
  x=dvals,
  pf = "Bootstrap")

full_adapt_est_asy_var <- data.frame(
  y=sapply(dvals, function(d) full_adapt_update$est_asy_var(N, d, eval)), 
  x=dvals,
  pf = "'Full' adaptation")

adapted_knotset_est_asy_var <- data.frame(
  y=sapply(dvals, function(d) adapted_knotset_update$est_asy_var(N, d, eval)), 
  x=dvals,
  pf = "Adapted knotset")

est_asy_var <- bind_rows(bpf_est_asy_var, full_adapt_est_asy_var, adapted_knotset_est_asy_var)

ggplot(aes(x=x,y=y,colour=pf),data=est_asy_var) +
  geom_point() + 
  geom_function(fun = function(d) bpf_update$sigma2(delta = d,epsilon = eval), colour = bpf_col ) +
  geom_function(fun = function(d) full_adapt_update$sigma2(delta = d,epsilon = eval), colour = fulladapt_col ) +
  geom_function(fun = function(d) adapted_knotset_update$sigma2(delta = d,epsilon = eval), colour = adaptedknot_col ) +
  scale_x_continuous("delta", breaks = seq(0,1,length.out = 11)) +
  scale_y_continuous("Asymptotic variance") +
  scale_color_manual("Particle filter type", values = c(Bootstrap = bpf_col, `'Full' adaptation` = fulladapt_col, `Adapted knotset` = adaptedknot_col)) + 
  ggtitle(paste0("Performance of particle filters, epsilon = ",eval))  +
  theme_bw()

# log scale
ggplot(aes(x=x,y=y,colour=pf),data=est_asy_var) +
  geom_point() + 
  geom_function(fun = function(d) bpf_update$sigma2(delta = d,epsilon = eval), colour = bpf_col ) +
  geom_function(fun = function(d) full_adapt_update$sigma2(delta = d,epsilon = eval), colour = fulladapt_col ) +
  geom_function(fun = function(d) adapted_knotset_update$sigma2(delta = d,epsilon = eval), colour = adaptedknot_col ) +
  scale_x_continuous("delta", breaks = seq(0,1,length.out = 11)) +
  scale_y_continuous("Asymptotic variance (log scale)", transform = "log") +
  scale_color_manual("Particle filter type", values = c(Bootstrap = bpf_col, `'Full' adaptation` = fulladapt_col, `Adapted knotset` = adaptedknot_col)) + 
  ggtitle(paste0("Performance of particle filters, epsilon = ",eval))  +
  theme_bw()

# log scale delta <= 0.3
ggplot(aes(x=x,y=y,colour=pf),data=filter(est_asy_var,x < 0.35)) +
         geom_point() + 
         geom_function(fun = function(d) bpf_update$sigma2(delta = d,epsilon = eval), colour = bpf_col ) +
         geom_function(fun = function(d) full_adapt_update$sigma2(delta = d,epsilon = eval), colour = fulladapt_col ) +
         geom_function(fun = function(d) adapted_knotset_update$sigma2(delta = d,epsilon = eval), colour = adaptedknot_col ) +
         scale_x_continuous("delta", breaks = seq(0,1,length.out = 11)) +
         scale_y_continuous("Asymptotic variance (log scale)", transform = "log") +
         scale_color_manual("Particle filter type", values = c(Bootstrap = bpf_col, `'Full' adaptation` = fulladapt_col, `Adapted knotset` = adaptedknot_col)) + 
         ggtitle(paste0("Performance of particle filters, epsilon = ",eval))  +
         theme_bw()


# Theory says that all v_n,n will be equal...

adapted_knotset_update$v11(0.2,0.2)
bpf_update$v11(0.2,0.2)


# other values
eval = 0.5
ggplot(aes(x=x,y=y,colour=pf),data=est_asy_var) +
  geom_function(fun = function(d) bpf_update$sigma2(delta = d,epsilon = eval), colour = bpf_col) +
  geom_function(fun = function(d) full_adapt_update$sigma2(delta = d,epsilon = eval), colour = fulladapt_col ) +
  geom_function(fun = function(d) adapted_knotset_update$sigma2(delta = d,epsilon = eval), colour = adaptedknot_col ) +
  scale_x_continuous("delta", breaks = seq(0,1,length.out = 11)) +
  scale_y_continuous("Asymptotic variance") +
  ggtitle(paste0("Performance of particle filters, epsilon = ",eval))  +
  theme_bw()


## Figure 2 paper (Figure 2 in J&D): export 4 x 6 inches landscape (3.25 * 6 journal)

pf_label_order <- function(x) ordered(x, levels = c("Bootstrap", "'Full' adaptation", "Adapted knotset"))

est_asy_var_jd2008$pf <- pf_label_order(est_asy_var_jd2008$pf)

Nval <- unique(est_asy_var_jd2008$N)
eval <- 0.25
ggplot(aes(x=x,y=y), data=est_asy_var_jd2008) +
  geom_point(aes(shape = pf)) + 
  geom_function(aes(linetype = pf_label_order("Bootstrap")), fun = function(d) bpf_update$sigma2(delta = d,epsilon = eval)) +
  geom_function(aes(linetype = pf_label_order("'Full' adaptation")), fun = function(d) full_adapt_update$sigma2(delta = d,epsilon = eval)) +
  geom_function(aes(linetype = pf_label_order("Adapted knotset")), fun = function(d) adapted_knotset_update$sigma2(delta = d,epsilon = eval)) +
  scale_x_continuous(expression(delta), breaks = seq(0,1,length.out = 11), labels = scales::label_number(drop0trailing=TRUE)) +
  scale_shape_discrete(paste0("N = ",format(Nval,scientific = F, big.mark = ",")), solid = FALSE) + 
  scale_linetype_discrete(expression(N %->% infinity)) + 
  ylab("Variance") + 
  theme_bw()

eval <- 0.5
ggplot(aes(x=x,y=y), data=est_asy_var) +
  geom_function(aes(linetype = "'Full' adaptation"), fun = function(d) full_adapt_update$sigma2(delta = d,epsilon = eval) - bpf_update$sigma2(delta = d,epsilon = eval)) +
  geom_function(aes(linetype = "Adapted knotset"), fun = function(d) adapted_knotset_update$sigma2(delta = d,epsilon = eval) - bpf_update$sigma2(delta = d,epsilon = eval)) +
  scale_x_continuous(expression(delta), breaks = seq(0,1,length.out = 11), labels = scales::label_number(drop0trailing=TRUE)) +
  scale_shape_discrete(paste0("N = ",format(Nval,scientific = F, big.mark = ","))) + 
  scale_linetype_discrete(expression(N %->% infinity)) + 
  ylab("Variance (relative to BPF)") + 
  theme_bw()
  
  
# Figure 3: 2.5*8 inches

  eps_label <- function(eps){
      as.expression(bquote(epsilon~"="~.(eps)))
  }
  
  eps_label_variable <- function(eps, all_eps){
    
    ordered(eps, levels = all_eps, labels = sapply(all_eps, eps_label))
    
  }
  
  evals <- c(0.05,0.1,0.2,0.4,0.5)
  
  gg_geom_funs_rel <- function(epsilons){
    gg <- ggplot(aes(x=x), data = data.frame(x = range(est_asy_var$x))) 
    
    for(eps in epsilons){
      gg <- gg + 
        geom_function(aes(linetype = pf_label_order("'Full' adaptation")), 
                      fun = function(d,e) full_adapt_update$sigma2(delta = d,epsilon = e) -  bpf_update$sigma2(delta = d,epsilon = e), 
                      args = list(e=eps), 
                      data = data.frame(x = range(est_asy_var$x), eps = eps_label_variable(eps, epsilons))) +
        geom_function(aes(linetype = pf_label_order("Adapted knotset")), 
                      fun = function(d,e) adapted_knotset_update$sigma2(delta = d,epsilon = e) -  bpf_update$sigma2(delta = d,epsilon = e), 
                      args = list(e=eps), 
                      data = data.frame(x = range(est_asy_var$x), eps = eps_label_variable(eps, epsilons)))
    }
    
    return(gg)
  }
  

  gg_geom_funs_rel(evals) +
    scale_x_continuous(expression(delta), breaks = seq(0,1,length.out = 6), labels = scales::label_number(drop0trailing=TRUE)) +
    scale_shape_discrete(paste0("N = ",format(Nval,scientific = F, big.mark = ","))) + 
    scale_linetype_discrete(expression(N %->% infinity)) + 
    ylab("Excess Variance") + 
    facet_grid(rows=~eps, labeller = label_parsed) +
    theme_bw() +
    theme(legend.position="right") +
    guides(linetype = guide_legend(title.position = "top", title.hjust = 0.5))
