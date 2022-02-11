from urllib import response
import numpy as np
from tinychain.ml.linalg import matmul, identity
np.random.seed(123)

import json

import tinychain as tc
from tinychain.decorators import closure, post_op
from tinychain.ref import After, If, While
from tinychain.value import UInt, F32, Number, Int, Value
from tinychain.state import Map, Tuple
from tinychain.collection.tensor import Tensor, Dense, einsum

HOST = tc.host.Host('http://127.0.0.1:8702')
ENDPOINT = '/transact/hypothetical'

@post_op
def svd2(cxt, A: Tensor, l=UInt(0), epsilon=F32(1e-5), max_iter=UInt(1000)):
    #cxt.max_iter = max_iter
    #cxt.epsilon = epsilon
    cxt.shape = A.shape
    cxt.n_orig, cxt.m_orig = [UInt(dim) for dim in cxt.shape.unpack(2)]
    k = Int(If(l == UInt(0), Value.min(Int(cxt.n_orig), Int(cxt.m_orig)), l))
    A_orig=Tensor(A).copy()

    cxt.A1, n, m = Tuple(If(
        UInt(cxt.n_orig) > UInt(cxt.m_orig), 
        Tuple([Tensor(matmul(Tensor(A).transpose(), A)), Number(Tensor(A).shape[1]), Number(Tensor(A).shape[1])]),
        Tuple(If(
            UInt(cxt.n_orig) < UInt(cxt.m_orig),
            Tuple([Tensor(matmul(A, Tensor(A).transpose())), Number(Tensor(A).shape[0]), Number(Tensor(A).shape[0])]),
            Tuple([A, cxt.n_orig, cxt.m_orig])
        )),
    )).unpack(3)

    Q = Dense.random_uniform([n, k]).abs()
    cxt.qr = tc.linalg.qr
    Q, R = cxt.qr(x=Q).unpack(2)
    #Q_prev = Tensor(Q).copy()

    @closure(cxt.qr, cxt.A1)
    @post_op
    def step(i: UInt, Q_prev: Tensor, Q: Tensor, R: Tensor, err: F32):
        Z = Tensor(matmul(Tensor(cxt.A1), Tensor(Q)))#Tensor(matmul(Tensor(A1), Tensor(Q)))#einsum('ij,jk->ik', [Tensor(A1), Tensor(Q)]))
        _Q, _R = cxt.qr(x=Z).unpack(2)
        #can use other stopping criteria as well
        _err = F32(Tensor(_Q).sub(Q_prev).pow(2).sum())
        _Q_prev = Tensor(_Q).copy()
        return Map(i=i+1, Q_prev=_Q_prev, Q=Tensor(_Q), R=Tensor(_R), err=_err)

    @closure(epsilon, max_iter)
    @post_op
    def cond(i: UInt, err: F32):
        return (err > epsilon).logical_and(i < max_iter)

    result_loop = Map(While(cond, step, Map(
        i=UInt(0),Q_prev=Tensor(Q).copy(),
        Q=Tensor(Q).copy(),
        R=Tensor(R),
        err=F32(1.0)
        )))
    Q, R = result_loop['Q'], result_loop['R']

    singular_values = Tensor(Tensor(tc.linalg.diagonal(R)).pow(0.5))
    
    #return singular_values
    cxt.eye = identity(singular_values.shape[0], F32).as_dense().copy()
    cxt.inv_matrix = (cxt.eye * singular_values.pow(-1))
    cxt.Q_T = Tensor(Q).transpose()
    cxt.vec_sing_values_upd = Map(If(cxt.n_orig==cxt.m_orig, 
                            Map(left_vecs=cxt.Q_T, 
                                right_vecs=cxt.Q_T, 
                                singular_values=(singular_values).pow(2)),
                            Map(left_vecs=einsum('ij,jk->ik', [einsum('ij,jk->ik', [A_orig, Q]), cxt.inv_matrix]), 
                                right_vecs=cxt.Q_T, 
                                singular_values=singular_values)))
    vec_sing_values = Map(If(cxt.n_orig < cxt.m_orig, 
                        Map(left_vecs= cxt.Q_T, 
                            right_vecs= einsum('ij,jk->ik', [einsum('ij,jk->ik', [cxt.inv_matrix, Q]), A_orig]), 
                            singular_values=singular_values), 
                        cxt.vec_sing_values_upd))

    return vec_sing_values['left_vecs'], vec_sing_values['singular_values'], vec_sing_values['right_vecs']





def svd_simultaneous_power_iteration(A, k, epsilon=0.00001):
    #source http://mlwiki.org/index.php/Power_Iteration
    #adjusted to work with n<m and n>m matrices
    n_orig, m_orig = A.shape
    if k is None:
        k=min(n_orig,m_orig)
        
    A_orig=A.copy()
    if n_orig > m_orig:
        A = A.T @ A
        n, m = A.shape
    elif n_orig < m_orig:
        A = A @ A.T
        n, m = A.shape
    else:
        n,m=n_orig, m_orig
        
    Q = np.random.rand(n, k)
    Q, _ = np.linalg.qr(Q)
    Q_prev = Q
 
    for i in range(1000):
        Z = A @ Q
        Q, R = np.linalg.qr(Z)
        # can use other stopping criteria as well 
        err = ((Q - Q_prev) ** 2).sum()
        Q_prev = Q
        if err < epsilon:
            break
            
    singular_values=np.sqrt(np.diag(R))    
    if n_orig < m_orig: 
        left_vecs=Q.T
        #use property Values @ V = U.T@A => V=inv(Values)@U.T@A
        right_vecs=np.linalg.inv(np.diag(singular_values))@left_vecs.T@A_orig
    elif n_orig==m_orig:
        left_vecs=Q.T
        right_vecs=left_vecs
        singular_values=np.square(singular_values)
    else:
        right_vecs=Q.T
        #use property Values @ V = U.T@A => U=A@V@inv(Values)
        left_vecs=A_orig@ right_vecs.T @np.linalg.inv(np.diag(singular_values))

    return left_vecs, singular_values, right_vecs



# if n_orig < m_orig: 
#         left_vecs=Q.T
#         #use property Values @ V = U.T@A => V=inv(Values)@U.T@A
#         right_vecs=np.linalg.inv(np.diag(singular_values))@left_vecs.T@A_orig
#     elif n_orig==m_orig:
#         left_vecs=Q.T
#         right_vecs=left_vecs
#         singular_values=np.square(singular_values)
#     else:
#         right_vecs=Q.T
#         #use property Values @ V = U.T@A => U=A@V@inv(Values)
#         left_vecs=A_orig@ right_vecs.T @np.linalg.inv(np.diag(singular_values))


def main():
    # Data
    data = np.random.rand(4,3)
    eps = 1e-6
    #data = [1.0, 3.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    shape = [4, 3]
    # TinyChain
    cxt = tc.Context()
    cxt.A = tc.tensor.Dense.load(shape, tc.F32, data.flatten().tolist())
    cxt.svd2 = svd2
    cxt.result = cxt.svd2(A=cxt.A, l=2, epsilon=F32(1e-5), max_iter=1000)
    response = HOST.post(ENDPOINT, cxt)
    print(svd_simultaneous_power_iteration(data, k=None))
    res_shape_1 = response[0]['/state/collection/tensor/dense'][0][0]
    res_values_1 = np.array(response[0]['/state/collection/tensor/dense'][1]).reshape(res_shape_1)

    res_shape_2 = response[1]['/state/collection/tensor/dense'][0][0]
    res_values_2 = np.array(response[1]['/state/collection/tensor/dense'][1]).reshape(res_shape_2)

    res_shape_3 = response[2]['/state/collection/tensor/dense'][0][0]
    res_values_3 = np.array(response[2]['/state/collection/tensor/dense'][1]).reshape(res_shape_3)

    print(res_values_1 @ (np.eye(2, 2)*res_values_2) @ res_values_3)
    print(data)
    print(HOST.post(ENDPOINT, cxt))

if __name__ == '__main__':
    main()