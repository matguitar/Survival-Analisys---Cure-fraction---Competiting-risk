import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

# =================================================================
# SECTION 1: PARÂMETROS E DEFINIÇÕES
# =================================================================
PARAM_FIXOS = {
    "beta_sigma_int": 0.577998,
    "beta_mu_int": 2.753863,
    "beta_mu_x1": 1.937027,
    "beta_mu_x2": -0.887691,
    "beta_p0_int": 1.688387,
    "beta_p0_x1": -1.206960,
    "beta_p0_x2": 0.420279,
    "beta_phi_int": 2.297517
}

# --- Exemplo de Uso ---
# Ordem: [log_sigma, b_mu_int, b_mu_x1, b_mu_x2, b_p0_int, b_p0_x1, b_p0_x2, log_phi]
THETA_FIXO = np.array([
    0.577998, 2.753863,1.937027, -0.887691, 1.688387, -1.206960, 0.420279, 2.297517
])

N_FIXO = 300

N = N_FIXO

P_SUSC_TARGET_FIXO = 0.05

# =================================================================
# SECTION 2: FUNÇÕES MATEMÁTICAS
# =================================================================

def s_latente(w, mu, sigma):
    """Sobrevivência da Log-Normal Discreta"""
    # z = (log(w) - mu) / sigma
    return 1 - norm.cdf((np.log(w) - mu) / sigma)

def p_obs_susc_func(tau, mu, sigma, n_size):
    """
    Calcula a probabilidade observada de ser susceptível dado um tau.
    Implementação vetorizada para performance.
    """
    tau_floor = int(np.floor(tau))
    if tau_floor < 1:
        return 0.0
    
    # Valores de y de 1 até floor(tau)
    y_vals = np.arange(1, tau_floor + 1)
    
    # Cálculo de S(y) para todos os y
    s_y = s_latente(y_vals, mu, sigma)
    
    # Evita log de zero
    s_y = np.maximum(s_y, 1e-100)
    
    # Probabilidade de censura: média de S(y)^n
    # log_s_n = n * log(s_y) -> exp(log_s_n) = s_y^n
    p_censura = np.mean(np.power(s_y, n_size))
    
    return p_censura

# =================================================================
# SECTION 3: BUSCA DA RAIZ (SOLVER)
# =================================================================

def find_tau_python(mu_i, sigma_i, target, n_size):
    # Função objetivo: onde a diferença é zero
    def objective(tau):
        return p_obs_susc_func(tau, mu_i, sigma_i, n_size) - target

    try:
        # brentq precisa de um intervalo onde o sinal mude [f(a) e f(b) opostos]
        # Testamos de 1 até 1.000.000 (ajustável se necessário)
        tau_root = brentq(objective, 1.0, 1000000.0, xtol=1e-5)
        return tau_root
    except ValueError:
        # Caso o sinal não mude no intervalo, retornamos um valor de fallback
        return 50000.0



import numpy as np
import pandas as pd
from scipy.stats import norm, nbinom, uniform
from scipy.optimize import brentq

def rlnd_disc(size, mu, sigma):
    """Gera tempos log-normais discretos: ceil(exp(N(mu, sigma)))"""
    z = np.random.normal(mu, sigma, size)
    return np.ceil(np.exp(z))

def gen_data_lnd_nb(N, THETA_TRUE, P_SUSC_TARGET, N_REF):
    # 0. Gerar Covariáveis
    x1 = np.random.normal(0, 1, N)
    x2 = np.random.normal(0, 1, N)
    Xe = np.column_stack([np.ones(N), x1, x2])

    # 1. Parâmetros (Link Functions)
    # THETA_TRUE: [log_sigma, b_mu0, b_mu1, b_mu2, b_p0, b_p1, b_p2, log_phi]
    sigma = np.exp(THETA_TRUE[0])
    mu = Xe @ THETA_TRUE[1:4]
    
    # p0 (fração de cura) via link logit
    logits_p0 = Xe @ THETA_TRUE[4:7]
    p0 = 1 / (1 + np.exp(-logits_p0))
    
    phi = np.exp(THETA_TRUE[7])
    
    # eta derivado da probabilidade de cura p0 para o modelo Binomial Negativa
    # p0 = (1 + phi * eta)^(-1/phi)
    eta = (np.power(p0, -phi) - 1) / phi
    eta = np.maximum(eta, 0)
    
    # 2. Geração de Riscos (M) - Quantidade de causas competitivas
    # No scipy, nbinom.rvs usa (n, p). 
    # n = size = 1/phi, p = 1/(1 + phi*eta)
    n_size_nb = 1 / phi
    prob_bn = 1 / (1 + phi * eta)
    M = nbinom.rvs(n_size_nb, prob_bn, size=N)
    
    # 3. Geração de Tempos Latentes (W)
    W = np.full(N, np.inf)
    idx_susc = np.where(M > 0)[0]
    
    for i in idx_susc:
        # O tempo observado é o mínimo dos M tempos de falha competitivos
        latentes = rlnd_disc(M[i], mu[i], sigma)
        W[i] = np.min(latentes)
    
    # 4. Cálculo do Tau individual e Censura (C)
    # Usando a função de busca de raiz definida anteriormente
    C = np.zeros(N)
    for i in range(N):
        tau_i = find_tau_python(mu[i], sigma, P_SUSC_TARGET, N_REF)
        # Censura Uniforme Discreta entre 1 e tau_i
        C[i] = np.ceil(uniform.rvs(loc=0, scale=tau_i))
    
    # 5. Observação (Y) e Status (Delta)
    Y = np.minimum(W, C)
    delta = (W <= C).astype(int)
    
    # Ajuste para infinitos (indivíduos curados que nunca falham)
    # Se W é inf, Y assume o valor da censura e delta é 0
    mask_inf = np.isinf(W)
    Y[mask_inf] = C[mask_inf]
    delta[mask_inf] = 0
    
    return pd.DataFrame({
        'y': Y,
        'status': delta,
        'x1': x1,
        'x2': x2
    })


df_resultado = gen_data_lnd_nb(N=N_FIXO, THETA_TRUE=THETA_FIXO, P_SUSC_TARGET=P_SUSC_TARGET_FIXO, N_REF=N_FIXO)
print(df_resultado.head())

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
import scipy.linalg as la

# -----------------------------------------------------------------
# 1. Preparação das Matrizes de Covariáveis
# -----------------------------------------------------------------
# Usando o df_resultado gerado anteriormente
dados = df_resultado.copy()

# Xe_sigma (apenas intercepto)
Xe_sigma = np.ones((len(dados), 1))
# Xe_mu (intercepto + x1 + x2)
Xe_mu = np.column_stack([np.ones(len(dados)), dados['x1'], dados['x2']])
# Xe_p0 (intercepto + x1 + x2)
Xe_p0 = np.column_stack([np.ones(len(dados)), dados['x1'], dados['x2']])
# Xe_phi (apenas intercepto)
Xe_phi = np.ones((len(dados), 1))

y = dados['y'].values
status = dados['status'].values

# -----------------------------------------------------------------
# 2. Função de Log-Verossimilhança (Negativa para o minimize)
# -----------------------------------------------------------------
def fvero(theta):
    n_sigma = Xe_sigma.shape[1]
    n_mu = Xe_mu.shape[1]
    n_p0 = Xe_p0.shape[1]
    n_phi = Xe_phi.shape[1]

    # Fatiamento dos parâmetros
    idx = 0
    beta_sigma = theta[idx : idx + n_sigma]; idx += n_sigma
    beta_mu = theta[idx : idx + n_mu]; idx += n_mu
    beta_p0 = theta[idx : idx + n_p0]; idx += n_p0
    beta_phi = theta[idx : idx + n_phi]

    # Transformações (Links)
    sigma = np.exp(Xe_sigma @ beta_sigma).flatten()
    mu = (Xe_mu @ beta_mu).flatten()
    p0 = 1 / (1 + np.exp(-(Xe_p0 @ beta_p0))).flatten()
    phi = np.exp(Xe_phi @ beta_phi).flatten()
    
    # Prevenção de divisão por zero/phi muito pequeno
    phi = np.maximum(phi, 1e-6)
    eta = (np.power(p0, -phi) - 1) / phi
    eta = np.maximum(eta, 0)

    # Função de sobrevivência Lognormal Discreta S(y) = P(W > y)
    def S_LND(time, s, m):
        # Evitar log(0)
        time = np.maximum(time, 1e-8)
        return 1 - norm.cdf((np.log(time) - m) / s)

    # vF1 = P(W <= y)
    vF1 = 1 - S_LND(y, sigma, mu)
    # Sobrevivência populacional no tempo y
    Spop1 = np.power(1 + phi * eta * vF1, -1/phi)

    # vF2 = P(W <= y + 1)
    vF2 = 1 - S_LND(y + 1, sigma, mu)
    # Sobrevivência populacional no tempo y+1
    Spop2 = np.power(1 + phi * eta * vF2, -1/phi)

    # fpop(y) = P(Y = y) = S(y) - S(y+1) para dados discretos
    fpop = Spop1 - Spop2
    
    # Proteção numérica para o log
    fpop = np.maximum(fpop, 1e-10)
    Spop1 = np.maximum(Spop1, 1e-10)

    # Log-lik (Negativa para minimizar)
    loglik = np.sum(status * np.log(fpop) + (1 - status) * np.log(Spop1))
    
    return -loglik

# -----------------------------------------------------------------
# 3. Estimação (MLE)
# -----------------------------------------------------------------
n_total = Xe_sigma.shape[1] + Xe_mu.shape[1] + Xe_p0.shape[1] + Xe_phi.shape[1]
theta0 = np.array([
    0.2,             # b_sigma
    2, 1, 0,   # b_mu
    1.3, -1, 0,   # b_p0
    2              # b_phi
])

# BFGS para refinamento e obtenção da Hessiana
print("Refinando com BFGS...")
res = minimize(fvero,theta0, method='BFGS', options={'maxiter': 10})


# -----------------------------------------------------------------
# 4. Erros Padrão e Tabela de Resultados
# -----------------------------------------------------------------
# Calculando a Hessiana numericamente se o BFGS não fornecer uma boa inversão
import numdifftools as nd

# Calculamos a matriz de informação observada (Hessiana da log-lik negativa)
hess_func = nd.Hessian(fvero)
obs_inf = hess_func(res.x)

try:
    cov_mat = la.inv(obs_inf)
    se_theta = np.sqrt(np.maximum(0, np.diag(cov_mat)))
except:
    print("Erro ao inverter a Hessiana. Usando pseudo-inversa.")
    cov_mat = la.pinv(obs_inf)
    se_theta = np.sqrt(np.maximum(0, np.diag(cov_mat)))

t_vals = res.x / se_theta
p_vals = 2 * (1 - norm.cdf(np.abs(t_vals)))

# Nomes dos parâmetros
nomes = (["b_sigma_int"] + 
         ["b_mu_int", "b_mu_x1", "b_mu_x2"] + 
         ["b_p0_int", "b_p0_x1", "b_p0_x2"] + 
         ["b_phi_int"])

mest = pd.DataFrame({
    'Estimate': res.x,
    'Std. Error': se_theta,
    '|z|': np.abs(t_vals),
    'p-value': p_vals
}, index=nomes)

print("\n--- Resultado da Estimação ---")
print(mest.round(6))

import numpy as np
import pandas as pd

N_FIXO_LIST = (100, 200, 300, 400,500,700,1000)
M_MC = 500

theta_true = THETA_FIXO

resultados_mc = []

for N_FIXO in N_FIXO_LIST:

    estimates = []
    ci_lower = []
    ci_upper = []

    for m in range(M_MC):

        N = N_FIXO

        df_resultado = gen_data_lnd_nb(
            N=N_FIXO,
            THETA_TRUE=THETA_FIXO,
            P_SUSC_TARGET=P_SUSC_TARGET_FIXO,
            N_REF=N_FIXO
        )

        dados = df_resultado.copy()

        Xe_sigma = np.ones((len(dados), 1))
        Xe_mu = np.column_stack([np.ones(len(dados)), dados['x1'], dados['x2']])
        Xe_p0 = np.column_stack([np.ones(len(dados)), dados['x1'], dados['x2']])
        Xe_phi = np.ones((len(dados), 1))

        y = dados['y'].values
        status = dados['status'].values

        res = minimize(fvero, theta0, method='BFGS', options={'maxiter': 100})

        hess_func = nd.Hessian(fvero)
        obs_inf = hess_func(res.x)

        try:
            cov_mat = la.inv(obs_inf)
        except:
            cov_mat = la.pinv(obs_inf)

        se_theta = np.sqrt(np.maximum(0, np.diag(cov_mat)))

        estimates.append(res.x)

        ci_lower.append(res.x - 1.96 * se_theta)
        ci_upper.append(res.x + 1.96 * se_theta)

    estimates = np.array(estimates)
    ci_lower = np.array(ci_lower)
    ci_upper = np.array(ci_upper)

    bias = np.mean(estimates - theta_true, axis=0)
    rmse = np.sqrt(np.mean((estimates - theta_true) ** 2, axis=0))
    aw = np.mean(ci_upper - ci_lower, axis=0)
    cp = np.mean(
        (theta_true >= ci_lower) & (theta_true <= ci_upper),
        axis=0
    )

    for i, nome in enumerate(nomes):
        resultados_mc.append({
            "N": N_FIXO,
            "Parametro": nome,
            "Bias": bias[i],
            "RMSE": rmse[i],
            "AW": aw[i],
            "CP": cp[i]
        })

tabela_resultados = pd.DataFrame(resultados_mc)

print("\n================ RESULTADOS MONTE CARLO ================\n")
print(tabela_resultados.round(4))

tabela_resultados.to_csv("resultados_monte_carlo.csv", index=False)
