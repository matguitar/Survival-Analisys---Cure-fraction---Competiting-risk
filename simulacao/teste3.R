library(survival)
library(nlme)     
library(MASS)     
library(stats)    
library(ggplot2)  
library(tidyr)    
library(extraDistr) 

set.seed(42)

# =================================================================
# SECTION 0: PARÂMETROS - CENÁRIO 2
# =================================================================

logit <- function(p) log(p / (1 - p))

PARAM_FIXOS_NOVO <- list(
    beta_sigma_int = 0.2,          # log(sigma)
    beta_mu_int    = 5,            # Intercepto mu (constante)
    beta_p0_int    = logit(0.20),  # Intercepto p0 (z=1)
    beta_p0_z2     = logit(0.35),  # p0 para nível z=2
    beta_p0_z3     = logit(0.50),  # p0 para nível z=3
    beta_phi_int   = 0.8           # log(phi)
)

THETA_TRUE <- unlist(PARAM_FIXOS_NOVO)
N_PARAM    <- length(THETA_TRUE)
names(THETA_TRUE) <- names(PARAM_FIXOS_NOVO)

SAMPLE_SIZES  <- c(100,500)
N_MONTE_CARLO <- 1000
P_SUSC_TARGET <- 0.05

# =================================================================
# SECTION 1: FUNÇÕES DE SOBREVIVÊNCIA E BUSCA DO TAU
# =================================================================

S_latente <- function(w, mu, sigma) {
    # P(W >= w) sincronizado com ceiling(exp) - 1
    1 - pnorm((log(pmax(w, 1e-7)) - mu) / sigma)
}

prob_censura_estavel <- function(tau, mu, sigma, n_size) {
    tau_int <- floor(tau)
    if (tau_int < 1) return(1)
    y_vals <- 1:tau_int
    sy <- S_latente(y_vals, mu, sigma)
    # Log-sum-exp para evitar underflow com expoente N
    p_cens <- (1 / tau_int) * sum(exp(n_size * log(pmax(sy, 1e-100))))
    return(p_cens)
}

find_tau_robust <- function(mu, sigma, p_susc, n_size) {
    res <- tryCatch({
        uniroot(function(t) prob_censura_estavel(t, mu, sigma, n_size) - p_susc,
                interval = c(1, 15000), extendInt = "yes", tol = 1e-5)$root
    }, error = function(e) 2000)
    return(res)
}

# =================================================================
# SECTION 2: GERAÇÃO DE DADOS (CENÁRIO 2 - COVARIÁVEL Z)
# =================================================================

gen_data_scenario2 <- function(N, params, p_susc) {
    # Covariável z com 3 níveis (Referência z=1)
    z <- sample(c(1, 2, 3), size = N, replace = TRUE)
    z2 <- as.numeric(z == 2)
    z3 <- as.numeric(z == 3)
    
    sigma <- exp(params[1])
    mu    <- rep(params[2], N) # Mu constante
    
    # Fração de cura p0 depende de z
    lin_p0 <- params[3] + params[4]*z2 + params[5]*z3
    p0 <- 1 / (1 + exp(-lin_p0))
    
    phi <- exp(params[6])
    eta <- (pmax(p0, 1e-7)^(-phi) - 1) / phi
    
    # Riscos iniciais M ~ BN
    M <- rnbinom(N, size = 1/phi, prob = 1/(1 + phi*eta))
    
    W <- rep(Inf, N)
    idx_susc <- which(M > 0)
    for (i in idx_susc) {
        U <- runif(M[i])
        # Lógica rlnd_disc: teto do quantil menos 1
        W[i] <- ceiling(exp(mu[i] + sigma * qnorm(1 - U))) - 1
    }
    W <- pmax(0, W)
    
    # Tau calibrado para a média de mu
    tau_val <- find_tau_robust(params[2], sigma, p_susc, N)
    C <- rdunif(N, 1, max(1, round(tau_val)))
    
    Y_obs <- pmin(W, C)
    Delta <- as.numeric(W <= C & W < Inf)
    Y_obs[Y_obs == Inf] <- C[Y_obs == Inf]
    
    return(data.frame(y = Y_obs, status = Delta, z2 = z2, z3 = z3))
}

# =================================================================
# SECTION 3: VEROSSIMILHANÇA E OTIMIZAÇÃO
# =================================================================

fvero <- function(theta, data) {
    sigma <- exp(theta[1])
    mu    <- theta[2]
    
    lin_p0 <- theta[3] + theta[4]*data$z2 + theta[5]*data$z3
    p0 <- 1 / (1 + exp(-lin_p0))
    p0 <- pmin(pmax(p0, 1e-6), 0.9999)
    
    phi <- exp(theta[6])
    eta <- (p0^(-phi) - 1) / phi
    
    # Sobrevivência latente (P >= y e P >= y+1)
    Sy1 <- S_latente(data$y, mu, sigma)
    Sy2 <- S_latente(data$y + 1, mu, sigma)
    
    # Sobrevivência populacional (BN)
    Spop1 <- (1 + phi * eta * (1 - Sy1))^(-1/phi)
    Spop2 <- (1 + phi * eta * (1 - Sy2))^(-1/phi)
    
    fpop <- pmax(Spop1 - Spop2, 1e-16)
    
    # Contribuição: status=1 (fpop), status=0 (Spop1)
    loglik <- sum(data$status * log(fpop) + (1 - data$status) * log(pmax(Spop1, 1e-16)))
    
    if(!is.finite(loglik)) return(-1e20)
    return(loglik)
}

ml_optim <- function(theta_init, data) {
    # Tentativa com BFGS (rápido e preciso para mu constante)
    mgg <- optim(theta_init, fvero, data = data, method = "BFGS",
                 control = list(fnscale = -1, maxit = 1000, reltol = 1e-12))
    
    # Se falhar, tenta SANN como fallback
    if (mgg$convergence != 0) {
        mgg <- optim(theta_init, fvero, data = data, method = "SANN",
                     control = list(fnscale = -1, maxit = 500))
        mgg <- optim(mgg$par, fvero, data = data, method = "BFGS",
                     control = list(fnscale = -1, maxit = 1000))
    }
    
    if (mgg$convergence != 0) return(list(converged = FALSE))
    
    h <- tryCatch(fdHess(mgg$par, fvero, data = data)$Hessian, error = function(e) NA)
    if(any(is.na(h))) return(list(converged = FALSE))
    
    covmat <- tryCatch(ginv(-h), error = function(e) matrix(NA, N_PARAM, N_PARAM))
    setheta <- sqrt(pmax(0, diag(covmat)))
    
    if(any(is.na(setheta)) || any(setheta > 15)) return(list(converged = FALSE))
    
    return(list(par = mgg$par, se = setheta, converged = TRUE))
}

# =================================================================
# SECTION 4: LOOP DE SIMULAÇÃO MONTE CARLO
# =================================================================

resultados_lista <- list()

for (N in SAMPLE_SIZES) {
    cat(sprintf("\n--- Iniciando N = %d (%d reps) ---\n", N, N_MONTE_CARLO))
    
    est_rep <- matrix(NA, nrow = N_MONTE_CARLO, ncol = N_PARAM)
    se_rep  <- matrix(NA, nrow = N_MONTE_CARLO, ncol = N_PARAM)
    cens_obs <- numeric(N_MONTE_CARLO)
    valid_runs <- 0
    
    for (run in 1:N_MONTE_CARLO) {
        sim_data <- gen_data_scenario2(N, THETA_TRUE, P_SUSC_TARGET)
        cens_obs[run] <- 1 - mean(sim_data$status)
        
        # Usamos o valor real como chute inicial para acelerar a simulação
        res <- ml_optim(THETA_TRUE, sim_data)
        
        if (res$converged) {
            valid_runs <- valid_runs + 1
            est_rep[valid_runs, ] <- res$par
            se_rep[valid_runs, ]  <- res$se
        }
        
        if (run %% 250 == 0) {
            cat(sprintf("  Progresso: %d/%d | Válidas: %d | Censura: %.1f%%\n", 
                        run, N_MONTE_CARLO, valid_runs, mean(cens_obs[1:run])*100))
        }
    }
    
    # Cálculo das métricas para as rodadas válidas
    est_clean <- est_rep[1:valid_runs, ]
    se_clean  <- se_rep[1:valid_runs, ]
    
    bias   <- colMeans(est_clean) - THETA_TRUE
    rmse   <- sqrt(colMeans(sweep(est_clean, 2, THETA_TRUE, "-")^2))
    avg_se <- colMeans(se_clean)
    aw     <- 2 * qnorm(0.975) * avg_se
    
    lower_ci <- est_clean - qnorm(0.975) * se_clean
    upper_ci <- est_clean + qnorm(0.975) * se_clean
    cp <- colMeans(sweep(lower_ci, 2, THETA_TRUE, "<") & sweep(upper_ci, 2, THETA_TRUE, ">"))
    
    resultados_lista[[as.character(N)]] <- data.frame(
        N = N,
        Parameter = names(THETA_TRUE),
        True_Value = THETA_TRUE,
        Bias = bias,
        RMSE = rmse,
        Avg_SE = avg_se,
        AW = aw,
        CP = cp,
        Valid_Runs = valid_runs,
        Avg_Censura = mean(cens_obs)
    )
}

# =================================================================
# SECTION 5: RESULTADOS FINAIS E TABELA
# =================================================================

df_final <- do.call(rbind, resultados_lista)
rownames(df_final) <- NULL

print("--- RESULTADOS FINAIS CENÁRIO 2 (6 PARÂMETROS) ---")
print(df_final)

# Opcional: Salvar em CSV
# write.csv(df_final, "resultados_cenario2.csv", row.names = FALSE)
