#Part 1 - binary classification
#objective, gradient and hessian vector product for sigmoid and atan
#E(w) = sum_n [phi*(u_n) - y_n u_n] + (lam/2) w^T C w   where u = Xw
#bias is the LAST column of X so C just zeros out last entry
import numpy as np
from collections import namedtuple



# sigmoid stuff
# all of these only ever call exp on numbers <= 0 so nothing overflows (and no clipping)
def softplus(u):
    #phi*(u) = log(1 + e^u)
    return np.maximum(u, 0.0) + np.log1p(np.exp( -np.abs(u) ))


def sigmoid(u):

    e = np.exp(-np.abs(u))
    return np.where(u >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def dsigmoid(u):
    #same as sigma(u) * sigma(-u)
    #dont use p(1-p) because 1-p turns into exactly 0 once u is bigger than ~37
    e = np.exp(-np.abs(u))
    return e / (1.0 + e) ** 2



# atan stuff
# sigma(u) = 1/2 + arctan(u)/pi
def sigma_atan(u):
    #written with atan2 so it doesnt cancel out to 0 when u is really negative
    return np.arctan2(1.0, -u) / np.pi


def dsigma_atan(u):
    return 1.0 / (np.pi * (1.0 + u*u))


def phi_star_atan(u):
    #integral of sigma from 0 to u
    #the spec says integrate from -inf but that blows up for atan
    return u * sigma_atan(u) - np.log1p(u*u) / (2.0 * np.pi)



#bundle each activation together so the class can just swap them
Activation = namedtuple("Activation", "name sigma dsigma phi_star log_sigma")

SIGMOID = Activation("sigmoid", sigmoid, dsigmoid, softplus, lambda u: -softplus(-u))
ATAN = Activation("atan", sigma_atan, dsigma_atan, phi_star_atan, lambda u: np.log( sigma_atan(u) ))



#main objective class
class BinaryObjective:
    def __init__(self, X, y, lam, activation = SIGMOID):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.lam = float(lam)
        self.act = activation

        # counters for the spec (objective / gradient / HVP calls)
        self.n_obj = 0
        self.n_grad = 0
        self.n_hvp = 0



# helper for the regularizer
    @staticmethod
    def without_bias(v):
        #this is C v, just a copy with the bias (last spot) set to 0
        v = np.array(v, dtype=float)
        v[-1] = 0.0
        return v



# the 3 functions newton-cg needs
    def objective(self, w):

        self.n_obj += 1
        u = self.X @ w
        w_reg = self.without_bias(w)

        return np.sum( self.act.phi_star(u) - self.y * u ) + 0.5 * self.lam * (w_reg @ w_reg)



    def gradient_and_curvature(self, w):

        self.n_grad += 1
        u = self.X @ w

        # g = X^T (sigma(u) - y) + lam C w
        g = self.X.T @ (self.act.sigma(u) - self.y) + self.lam * self.without_bias(w)

        #curvature = sigma'(u), get it once here and reuse it for every HVP at this w
        return g, self.act.dsigma(u)



    def hessian_vector_product(self, v, curvature):

        self.n_hvp += 1
        #H v = X^T (curvature * (X v)) + lam C v  ->  never actually builds H
        return self.X.T @ (curvature * (self.X @ v)) + self.lam * self.without_bias(v)



#evaluation
def log_loss(X, y, w, act = SIGMOID):
    #uses 1 - sigma(u) = sigma(-u) so we never have to clip anything
    u = X @ w
    return -np.mean( y * act.log_sigma(u) + (1 - y) * act.log_sigma(-u) )


def accuracy(X, y, w):
    #predict 1 when u >= 0 (same as sigma >= 0.5)
    return np.mean( (X @ w >= 0) == y )
