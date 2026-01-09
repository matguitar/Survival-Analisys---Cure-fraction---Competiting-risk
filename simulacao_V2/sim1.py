import numpy as np
import pandas as pd
from scipy.stats import norm, nbinom
from scipy.optimize import minimize, brentq
import scipy.linalg as la

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

# Ordem: [beta_sigma_int, b_mu_int, b_mu_x1, b_mu_x2, b_p0_int, b_p0_x1, b_p0_x2, beta_phi_int]
THETA_FIXO = np.array([
    0.577998, 2.753863, 1.937027, -0.887691, 1.688387, -1.206960, 0.420279, 2.297517
])

N_FIXO = 100
P_SUSC_TARGET_FIXO = 0.05

# =================================================================
# SECTION 2: FUNÇÕES MATEMÁTICAS (LND E CENSURA)
# =================================================================

def s_latente(w, mu, sigma):
    """Sobrevivência da Log-Normal Discreta S(w) = P(W > w)"""
    # w deve ser >= 0. Se w=0, S(0)=1.
    # Usando a definição: 1 - Phi((log(w) - mu)/sigma)
    w_adj = np.where(w <= 0, 1e-10, w)
    surv = 1 - norm.cdf((np.log(w_adj) - mu) / sigma)
    return np.where(w <= 0, 1.0, surv)

def p_obs_susc_func(tau, mu, sigma, n_causes):
    """Calcula a probabilidade média de censura para busca do tau"""
    tau_floor = int(np.floor(tau))
    if tau_floor < 1: return 0.0
    y_vals = np.arange(1, tau_floor + 1)
    # S(y)^m pois o tempo observado Y é o min(W1...Wm)
    s_y = s_latente(y_vals - 1, mu, sigma)
    p_censura = np.mean(np.power(np.maximum(s_y, 1e-100), n_causes))
    return p_censura

def find_tau(mu_i, sigma_i, target, n_causes):
    def objective(tau):
        return p_obs_susc_func(tau, mu_i, sigma_i, n_causes) - target
    try:
        return brentq(objective, 1.1, 10000.0, xtol=1e-5)
    except ValueError:
        return 10000.0

# =================================================================
# SECTION 3: GERAÇÃO DE DADOS
# =================================================================

def rlnd_disc(size, mu, sigma):
    """Gera tempos latentes: W = floor(exp(mu + sigma*Z)) + 1"""
    Z = np.random.normal(0, 1, size)
    W_cont = np.exp(mu + sigma * Z)
    return np.floor(W_cont).astype(int) + 1

def gen_data_lnd_nb(N, THETA_TRUE, P_SUSC_TARGET):
    # 0. Covariáveis
    x1 = np.random.binomial(1, 0.41, N)
    x2 = np.random.binomial(1, 0.63, N)
    Xe = np.column_stack([np.ones(N), x1, x2])
    
    # 1. Parâmetros (Link Functions)
    sigma = np.exp(THETA_TRUE[0]) 
    mu = Xe @ THETA_TRUE[1:4]
    p0 = 1 / (1 + np.exp(Xe @ THETA_TRUE[4:7]))
    phi = np.exp(THETA_TRUE[7])
    
    eta = (np.power(p0, -phi) - 1) / phi
    
    # 2. Número de causas (M)
    n_size_nb = 1 / phi
    prob_bn = 1 / (1 + phi * eta)
    M = nbinom.rvs(n_size_nb, prob_bn, size=N)
    M = np.minimum(M,10)
    
    # 3. Tempos Latentes (W)
    W = np.full(N, np.inf)
    for i in range(N):
        if M[i] > 0:
            latentes = rlnd_disc(M[i], mu[i], sigma)
            W[i] = np.min(latentes)
    
    # 4. Censura (C)
    C = np.zeros(N)
    for i in range(N):
        if M[i] > 0:
            tau_i = find_tau(mu[i], sigma, P_SUSC_TARGET, M[i])
            C[i] = np.random.randint(1, max(2, int(np.ceil(tau_i))))
        else:
            C[i] = 1000.0 # Valor arbitrário para curados
            
    # 5. Observação (Y) e Status
    Y = np.minimum(W, C)
    delta = (W <= C).astype(int)
    
    # Ajuste para infinitos
    mask_inf = np.isinf(W)
    Y[mask_inf] = C[mask_inf]
    delta[mask_inf] = 0
    
    return pd.DataFrame({'y': Y, 'status': delta, 'x1': x1, 'x2': x2})

# =================================================================
# SECTION 4: ESTIMAÇÃO POR MÁXIMA VEROSSIMILHANÇA
# =================================================================

def fvero(theta, y, status, Xe_mu, Xe_p0):
    # Fatiamento e Links
    sigma = np.exp(theta[0])
    mu = (Xe_mu @ theta[1:4]).flatten()
    p0 = 1 / (1 + np.exp(Xe_p0 @ theta[4:7])).flatten()
    phi = np.exp(theta[7])
    
    phi = np.maximum(phi, 1e-6)
    eta = (np.power(p0, -phi) - 1) / phi
    
    def get_Spop(t):
        # Conforme eq 3.12: Spop = (1 + phi*eta*F(t))^(-1/phi)
        # F(t) é a CDF da LND: Phi((log(t)-mu)/sigma)
        t_adj = np.where(t <= 0, 1e-10, t)
        Fw = norm.cdf((np.log(t_adj) - mu) / sigma)
        Fw = np.where(t <= 0, 0, Fw)
        return np.power(1 + phi * eta * Fw, -1/phi)

    Spop1 = get_Spop(y)
    Spop2 = get_Spop(y + 1)
    
    fpop = np.maximum(Spop1 - Spop2, 1e-12)
    Spop1 = np.maximum(Spop1, 1e-12)
    
    loglik = np.sum(status * np.log(fpop) + (1 - status) * np.log(Spop1))
    return -loglik

# =================================================================
# EXECUÇÃO E RESULTADOS
# =================================================================

df_resultado = gen_data_lnd_nb(N_FIXO, THETA_FIXO, P_SUSC_TARGET_FIXO)

# Matrizes
Xe = np.column_stack([np.ones(len(df_resultado)), df_resultado['x1'], df_resultado['x2']])
y_vals = df_resultado['y'].values
delta_vals = df_resultado['status'].values

# Otimização
theta0 = THETA_FIXO * 0.8 # Chute inicial
res = minimize(fvero, theta0, args=(y_vals, delta_vals, Xe, Xe), method='BFGS')

# Erros Padrão (Hessiana)
from scipy.optimize import approx_fprime
def hessian_scipy(f, x, args, eps=1e-4):
    n = len(x)
    hess = np.zeros((n, n))
    g0 = approx_fprime(x, f, eps, *args)
    for i in range(n):
        x_perturbed = np.copy(x)
        x_perturbed[i] += eps
        g_perturbed = approx_fprime(x_perturbed, f, eps, *args)
        hess[i, :] = (g_perturbed - g0) / eps
    return (hess + hess.T) / 2

obs_inf = hessian_scipy(fvero, res.x, (y_vals, delta_vals, Xe, Xe))
cov_mat = la.pinv(obs_inf)
se_theta = np.sqrt(np.maximum(0, np.diag(cov_mat)))

# Tabela
nomes = ["b_sigma_int", "b_mu_int", "b_mu_x1", "b_mu_x2", "b_p0_int", "b_p0_x1", "b_p0_x2", "b_phi_int"]
mest = pd.DataFrame({
    'True': THETA_FIXO,
    'Estimate': res.x,
    'Std.Error': se_theta,
    'p-value': 2 * (1 - norm.cdf(np.abs(res.x / se_theta)))
}, index=nomes)

print("\n--- Resultado da Estimação ---")
print(mest.round(4))
