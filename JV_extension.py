import numpy as np
from scipy.special import i0
import warnings

def vonmisespdf(x, mu, K):
    """
    Von Mises 分布的概率密度函数（pdf）。
    
    参数：
    x  -- 需要计算的角度值（以弧度为单位）
    mu -- Von Mises 分布的均值（以弧度为单位）
    K  -- 浓度参数
    
    返回：
    p  -- 在 x 处的概率密度值
    """
    p = np.exp(K * np.cos(x - mu)) / (2 * np.pi * i0(K))
    return p

def wrap(Y, bound=np.pi):
    """
    将值映射到圆周空间 [-bound, bound)。
    """
    return np.mod(Y + bound, 2 * bound) - bound

def A1inv(R):
    """
    逆函数，用于估计 Von Mises 分布的 K 值。
    """
    if 0 <= R < 0.53:
        return 2 * R + R**3 + (5 * R**5) / 6
    elif R < 0.85:
        return -0.4 + 1.39 * R + 0.43 / (1 - R)
    else:
        return 1 / (R**3 - 4 * R**2 + 3 * R)


def JV10_function(X, T, NT, B_start):
    """
    JV10_function 辅助函数，用于计算 Von Mises 分布的参数估计。
    """
    
    # 输入检查
    if X.ndim > 1 or T.ndim > 1 or X.shape[0] != T.shape[0] or (NT.shape[0] != X.shape[0] or NT.shape[0] != T.shape[0]):
        raise ValueError("Input is not correctly dimensioned")
    
    if (B_start[0] < 0 or 
        any(b < 0 for b in B_start[1:4]) or 
        any(b > 1 for b in B_start[1:4]) or 
        abs(sum(B_start[1:4]) - 1) > 1e-6):
        raise ValueError('Invalid initial parameters')

    MaxIter = 10**4
    MaxdLL = 10**-4
    
    n = X.shape[0]
    nn = NT.shape[0] if NT is not None else 0

    # 初始参数
    if B_start is None:
        K = 5
        Pt = 0.5
        Pn = 0.3 if nn > 0 else 0
        Pu = 1 - Pt - Pn
    else:
        K, Pt, Pn, Pu = B_start

    E = wrap(X - T)
    NE = wrap(np.repeat(X[:, np.newaxis], nn, axis=1) - NT) if nn > 0 else np.zeros_like(X)

    LL = np.nan
    dLL = np.nan
    iter = 0

    while True:
        iter += 1
        
        Wt = Pt * vonmisespdf(E, 0, K)
        Wg = Pu * np.ones(n) / (2 * np.pi)

        if nn == 0:
            Wn = np.zeros_like(Wg)
        else:
            Wn = Pn/nn * vonmisespdf(NE, 0, K)
        
        W = np.sum(np.column_stack((Wt, Wn, Wg)), axis=1)
        
        dLL = LL - np.sum(np.log(W))
        LL = np.sum(np.log(W))
        if np.abs(dLL) < MaxdLL or iter > MaxIter:
            break
        
        Pt = np.sum(Wt / W) / n
        Pn = np.sum(np.sum(Wn, axis=1) / W) / n
        Pu = np.sum(Wg / W) / n
        
        rw = np.column_stack((Wt / W, Wn / np.repeat(W[:, np.newaxis], nn, axis=1)))
        
        S = np.column_stack((np.sin(E), np.sin(NE)))
        C = np.column_stack((np.cos(E), np.cos(NE)))
        r = np.array([np.sum(np.sum(S * rw)), np.sum(np.sum(C * rw))])
        
        if np.sum(np.sum(rw)) == 0:
            K = 0
        else:
            R = np.sqrt(np.sum(r**2)) / np.sum(np.sum(rw))
            K = A1inv(R)
        
        if n <= 15:
            if K < 2:
                K = max(K - 2 / (n * K), 0)
            else:
                K = K * (n - 1)**3 / (n**3 + n)

    if iter > MaxIter:
        print("Warning: Maximum iteration limit exceeded.")
        B = [np.nan, np.nan, np.nan, np.nan]
        LL = np.nan
    else:
        B = [K, Pt, Pn, Pu]

    return B, LL


def JV10_error(X, T=None):
    """
    Returns precision (P) and bias (B) measures for circular recall data.
    Inputs X and T are (nx1) vectors of responses and target values
    respectively, in the range -PI <= X < PI. If T is not specified,
    the default target value is 0.

    Ref: Bays PM, Catalao RFG & Husain M. The precision of visual working 
    memory is set by allocation of a shared resource. Journal of Vision 
    9(10): 7, 1-11 (2009)
    """

    # If T is not specified, set it to a zero vector of the same size as X
    if T is None:
        T = np.zeros_like(X)

    # Ensure inputs are within the range -PI to PI
    if np.any(np.abs(X) > np.pi) or np.any(np.abs(T) > np.pi):
        raise ValueError('Input values must be in radians, range -PI to PI')

    # Ensure inputs have the same dimensions
    if X.shape != T.shape:
        raise ValueError('Inputs must have the same dimensions')

    # Convert row vectors to column vectors if necessary
    if X.ndim == 1:
        X = X[:, np.newaxis]
        T = T[:, np.newaxis]

    E = wrap(X - T)

    # Precision calculation
    N = X.shape[0]
    x = np.logspace(-2, 2, 100)

    ##这里是对什么量的假设？
    P0 = np.trapz(N / (np.sqrt(x) * np.exp(x + N * np.exp(-x))), x)  # Expected precision under uniform distribution
    P = 1.0 / np.std(E, ddof=1) - P0  # Corrected precision

    # Bias calculation
    B = np.angle(np.mean(np.exp(1j * E)))

    return P, B



def JV10_fit(X, T, NT=None):
    """
    返回混合模型的最大似然参数 B，用于描述回忆响应 X 与目标 T、非目标 NT 和均匀响应的关系。
    输入应为弧度，范围 -PI <= X < PI。
    
    B = JV10_fit(X, T, NT) 返回向量 [K pT pN pU]，其中 K 是 Von Mises 分布的浓度参数，
    pT 是以目标值响应的概率，pN 是以非目标值响应的概率，pU 是随机响应的概率。

    [B, LL] = JV10_fit(X, T, NT) 还返回对数似然 LL。
    """
    
    if NT is None:
        NT = np.zeros((X.shape[0], 0))
        nn = 0
    else:
        nn = NT.shape[0]

    if (X.ndim > 1 or T.ndim > 1 or X.shape[0] != T.shape[0] or 
        (nn > 0 and (NT.shape[0] != X.shape[0] or NT.shape[0] != T.shape[0]))):
        raise ValueError('Input is not correctly dimensioned')

    n = X.shape[0]

    # 初始参数
    K_values = [1, 10, 100]
    N_values = [0.01, 0.1, 0.4]
    U_values = [0.01, 0.1, 0.4]

    if nn == 0:
        N_values = [0]

    LL = -np.inf
    B = [np.nan, np.nan, np.nan, np.nan]

    warnings.filterwarnings("ignore", category=UserWarning, message="JV10_function:MaxIter")

    # 参数估计
    for K in K_values:
        for N in N_values:
            for U in U_values:
                b, ll = JV10_function(X, T, NT, [K, 1-N-U, N, U])
                if ll > LL:
                    LL = ll
                    B = b

    warnings.filterwarnings("default", category=UserWarning, message="JV10_function:MaxIter")

    return B, LL
