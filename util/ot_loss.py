import numpy as np
import torch
import pdb

def sinkhorn_knopp(a, b, M, reg, numItermax=1000,
             stopThr=1e-9, verbose=False, log=False, **kwargs):
    """
    Solve the entropic regularization optimal transport problem and return the OT matrix.
    """
    # Not allow gradient descent.
    with torch.no_grad():
        if len(a) == 0:
            a = torch.ones(M.size(0)) / M.size(0)
        if len(b) == 0:
            b = torch.ones(M.size(1)) / M.size(1)

        # init data
        Nini = len(a)
        Nfin = len(b)

        if len(b.size()) > 1:
            nbb = b.size(1)
        else:
            nbb = 0

        if log:
            log = {'err': []}

        # we assume that no distances are null except those of the diagonal of
        # distances
        if nbb:
            u = torch.ones(Nini, nbb) / Nini
            v = torch.ones(Nfin, nbb) / Nfin
        else:
            u = torch.ones(Nini) / Nini
            v = torch.ones(Nfin) / Nfin

        K = torch.exp(-M/reg)

        Kp = (1 / a).view(-1, 1) * K
        cpt = 0
        err = 1
        while (err > stopThr and cpt < numItermax):
            uprev = u
            vprev = v

            KtransposeU = torch.mm(K.transpose(0,1), u)
            v = b / KtransposeU
            u = 1. / torch.mm(Kp, v)

            if ((KtransposeU == 0).sum()>0 or
                    (torch.isnan(u)).sum()>0 or (torch.isnan(v)).sum()>0 or
                    (torch.isinf(u)).sum()>0 or (torch.isinf(v)).sum()>0):
                # we have reached the machine precision
                # come back to previous solution and quit loop
                print('Warning: numerical errors at iteration', cpt)
                u = uprev
                v = vprev
                break
            if cpt % 10 == 0:
                # we can speed up the process by checking for the error only all
                # the 10th iterations
                if nbb:
                    err = torch.sum((u - uprev) ** 2) / torch.sum((u) ** 2) + \
                          torch.sum((v - vprev) ** 2) / torch.sum((v) ** 2)
                else:
                    # compute right marginal tmp2= (diag(u)Kdiag(v))^T1
                    tmp2 = torch.einsum('i,ij,j->j', u, K, v)
                    err = torch.norm(tmp2 - b) ** 2  # violation of marginal
                if log:
                    log['err'].append(err)

                if verbose:
                    if cpt % 200 == 0:
                        print(
                            '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                    print('{:5d}|{:8e}|'.format(cpt, err))
            cpt = cpt + 1
        if log:
            log['u'] = u
            log['v'] = v


        if nbb:  # return only loss
            res = torch.einsum('ik,ij,jk,ij->k', u, K, v, M)
            T = u.view((-1, 1)) * K * v.view((1, -1))
            if log:
                log['wd'] = res
                return T, log
            else:
                return T

        else:  # return OT matrix

            if log:
                return u.view((-1, 1)) * K * v.view((1, -1)), log
            else:
                return u.view((-1, 1)) * K * v.view((1, -1))



def update_UV(a, b, nbb, M, reg, numItermax=1000,
             stopThr=1e-9, verbose=False, log=False):

    # init data
    Nini = len(a)
    Nfin = len(b)

    if log:
        log = {'err': []}


    u = torch.ones(Nini, nbb) / Nini
    v = torch.ones(Nfin, nbb) / Nfin

    K = torch.exp(-M / reg)
    Kp = (1 / a).view(-1, 1) * K
    cpt = 0
    err = 1
    while (err > stopThr and cpt < numItermax):
        uprev = u
        vprev = v

        KtransposeU = torch.mm(K.transpose(0, 1), u)
        v = b / KtransposeU
        u = 1. / torch.mm(Kp, v)

        if ((KtransposeU == 0).sum() > 0 or
                (torch.isnan(u)).sum() > 0 or (torch.isnan(v)).sum() > 0 or
                (torch.isinf(u)).sum() > 0 or (torch.isinf(v)).sum() > 0):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration', cpt)
            u = uprev
            v = vprev
            break
        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = torch.sum((u - uprev) ** 2) / torch.sum((u) ** 2) + \
                    torch.sum((v - vprev) ** 2) / torch.sum((v) ** 2)
            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))
        cpt = cpt + 1

    # transport matrix
    T = u.view((-1, 1)) * K * v.view((1, -1))
    if log:
        log['u'] = u
        log['v'] = v
        log['T'] = T
        return u, v, log
    else:
        return u, v


def sinkhorn_wasserstein_loss(a, b, M, reg, numItermax, cuda,
             stopThr=1e-9, verbose=False, log=False, **kwargs):
    """
    Solve the entropic regularization optimal transport problem and return the OT matrix.
    """

    if len(a) == 0:
        a = torch.ones(M.size(0)) / M.size(0)
    if len(b) == 0:
        b = torch.ones(M.size(1)) / M.size(1)
    
    if True:
        a = a.cuda(cuda)
        b = b.cuda(cuda)
        M = M.cuda(cuda)

    if len(b.size()) > 1:
        nbb = b.size(1)
    else:
        assert False, 'a and b must be with shape(n,1).'

    if log:
        u, v, log = update_UV(a, b, nbb, M.detach(), reg, numItermax, stopThr, verbose, log)
    else:
        u, v = update_UV(a, b, nbb, M.detach(), reg, numItermax, stopThr, verbose, log)

    K = torch.exp(-M / reg)

    w_loss = torch.einsum('ik,ij,jk,ij->k', u, K, v, M)

    if log:
        return w_loss, log
    else:
        return w_loss


def sinkhorn_stabilized(a, b, M, reg, numItermax, cuda, tau=1e3, stopThr=1e-9,
                        warmstart=None, verbose=False, print_period=20,
                        log=False, **kwargs):

    if len(a) == 0:
        a = torch.ones((M.shape[0],)) / M.shape[0]
    if len(b) == 0:
        b = torch.ones((M.shape[1],)) / M.shape[1]

    # test if multiple target
    if len(b.shape) > 1:
        nbb = b.shape[1]
        a = a.unsqueeze(1)
    else:
        nbb = 0

    # init data
    na = len(a)
    nb = len(b)

    cpt = 0
    if log:
        log = {'err': []}

    # we assume that no distances are null except those of the diagonal of distances
    if warmstart is None:
        alpha, beta = torch.zeros(na), torch.zeros(nb)
    else:
        alpha, beta = warmstart

    if nbb:
        u = torch.ones(na, nbb) / na
        v = torch.ones(nb, nbb) / nb
    else:
        u = torch.ones(na) / na
        v = torch.ones(nb) / nb

    if True: #cuda:
        u, v = u.cuda(cuda), v.cuda(cuda)
        alpha, beta = alpha.cuda(cuda), beta.cuda(cuda)

    def get_K(alpha, beta):
        """log space computation"""
        return torch.exp(-(M - alpha.reshape(na, 1).cuda(cuda) - beta.reshape(1, nb).cuda(cuda)) / reg)

    def get_Gamma(alpha, beta, u, v):
        """log space gamma computation"""
        return torch.exp(-(M - alpha.reshape(na, 1).cuda(cuda) - beta.reshape(1, nb).cuda(cuda)) / reg + torch.log(u.reshape(na, 1).cuda(cuda)) + torch.log(v.reshape(1, nb).cuda(cuda)))

    K = get_K(alpha, beta)
    transp = K
    loop = 1
    cpt = 0
    err = 1

    while loop:
        uprev = u
        vprev = v

        # sinkhorn update
        v = b / (torch.mv(torch.t(K), u) + 1e-16)
        u = a / (torch.mv(K, v) + 1e-16)
        
        # remove numerical problems and store them in K
        if torch.abs(u).max() > tau or torch.abs(v).max() > tau:
            if nbb:
                alpha, beta = alpha + reg * \
                    torch.max(torch.log(u), 1), beta + reg * torch.max(torch.log(v))
            else:
                alpha, beta = alpha + reg * torch.log(u), beta + reg * torch.log(v)
                if nbb:
                    u, v = torch.ones((na, nbb)) / na, torch.ones((nb, nbb)) / nb
                else:
                    u, v = torch.ones(na) / na, torch.ones(nb) / nb
                if True: #cuda:
                    u, v = u.cuda(cuda), v.cuda(cuda)
            K = get_K(alpha, beta)

        if cpt % print_period == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            if nbb:
                err = torch.sum((u - uprev)**2) / torch.sum((u)**2) + \
                    torch.sum((v - vprev)**2) / torch.sum((v)**2)
            else:
                transp = get_Gamma(alpha, beta, u, v)
                err = torch.norm((torch.sum(transp, dim=0) - b))**2
            if log:
                log['err'].append(err)

            if verbose:
                if cpt % (print_period * 20) == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

        if err <= stopThr:
            loop = False

        if cpt >= numItermax:
            loop = False

        if torch.sum((u != u) == 1) > 0 or torch.sum((v != v) == 1) > 0:
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration', cpt)
            u = uprev
            v = vprev
            break

        cpt = cpt + 1

    # print('err=',err,' cpt=',cpt)
    if log:
        log['logu'] = alpha / reg + torch.log(u)
        log['logv'] = beta / reg + torch.log(v)
        log['alpha'] = alpha + reg * torch.log(u)
        log['beta'] = beta + reg * torch.log(v)
        log['warmstart'] = (log['alpha'], log['beta'])
        if nbb:
            res = torch.zeros((nbb))
            for i in range(nbb):
                res[i] = torch.sum(get_Gamma(alpha, beta, u[:, i], v[:, i])
                                   * M)
            return res, log

        else:
            return get_Gamma(alpha, beta, u, v), log
    else:
        if nbb:
            res = torch.zeros((nbb))
            for i in range(nbb):
                res[i] = torch.sum(get_Gamma(alpha, beta, u[:, i], v[:, i])
                                   * M)
            return res
        else:
            return get_Gamma(alpha, beta, u, v)



def sinkhorn_stabilized_normalized(x, y, epsilon, n, niter, cuda):

    Wxy = stabilized_sinkhorn_loss(x, y, epsilon, n, niter, cuda)
    Wxx = stabilized_sinkhorn_loss(x, x, epsilon, n, niter, cuda)
    Wyy = stabilized_sinkhorn_loss(y, y, epsilon, n, niter, cuda)
    return 2 * Wxy - Wxx - Wyy

def stabilized_sinkhorn_loss(x, y, reg, n, numItermax, cuda0):
    tau=1e3
    stopThr=1e-9
    warmstart=None
    verbose=False
    print_period=20
    log=False
    # both marginals are fixed with equal weights
    a = 1. / n * torch.ones(n)
    b = 1. / n * torch.ones(n)

    if True:
        a = a.cuda(cuda0)
        b = b.cuda(cuda0)

    M = cost_matrix(x, y)  # cost function

    # init data
    na = a.size(0)
    nb = b.size(0)

    cpt = 0
    if log:
        log = {'err': []}

    # we assume that no distances are null except those of the diagonal of distances
    if warmstart is None:
        alpha, beta = M.new(na).zero_(), M.new(nb).zero_()
    else:
        alpha, beta = warmstart
    u, v = M.new(na).fill_(1 / na), M.new(nb).fill_(1 / nb)
    uprev, vprev = M.new(na).zero_(), M.new(nb).zero_()

    def get_K(alpha, beta):
        """log space computation"""
        return torch.exp(-(M - alpha[:, None].expand_as(M) - beta[None, :].expand_as(M)) / reg)

    def get_Gamma(alpha, beta, u, v):
        """log space gamma computation"""
        return torch.exp(
            -(M - alpha[:, None].expand_as(M) - beta[None, :].expand_as(M)) / reg + torch.log(u)[:, None].expand_as(
                M) + torch.log(v)[None, :].expand_as(M))

    K = get_K(alpha, beta)
    transp = K
    loop = True
    cpt = 0
    err = 1
    while loop:

        if u.abs().max() > tau or v.abs().max() > tau:
            alpha, beta = alpha + reg * torch.log(u), beta + reg * torch.log(v)
            u, v = M.new(na).fill_(1 / na), M.new(nb).fill_(1 / nb)
            K = get_K(alpha, beta)

        uprev = u
        vprev = v

        Kt_dot_u = torch.mv(K.t(), u)
        v = b / Kt_dot_u
        u = a / torch.mv(K, v)

        if cpt % print_period == 0:
            # we can speed up the process by checking for the error only all the 10th iterations
            transp = get_Gamma(alpha, beta, u, v)
            err = torch.dist(transp.sum(0), b) ** 2
            if log:
                log['err'].append(err)

            if verbose:
                if cpt % (print_period * 20) == 0:
                    print('{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

        if err <= stopThr:
            loop = False

        if cpt >= numItermax:
            loop = False

        if (Kt_dot_u == 0).sum() > 0 or (u != u).sum() > 0 or (v != v).sum() > 0:  # u!=u is a test for NaN...
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errrors')
            if cpt != 0:
                u = uprev
                v = vprev
            break

        cpt = cpt + 1


    Gamma = get_Gamma(alpha, beta, u, v)

    if log:
        log['logu'] = alpha / reg + torch.log(u)
        log['logv'] = beta / reg + torch.log(v)
        log['alpha'] = alpha + reg * torch.log(u)
        log['beta'] = beta + reg * torch.log(v)
        log['warmstart'] = (log['alpha'], log['beta'])
        return torch.sum(torch.mul(Gamma, M)), log
    else:
        return torch.sum(torch.mul(Gamma, M))


def sinkhorn_normalized(x, y, epsilon, n, niter, cuda):

    Wxy = sinkhorn_loss(x, y, epsilon, n, niter, cuda)
    Wxx = sinkhorn_loss(x, x, epsilon, n, niter, cuda)
    Wyy = sinkhorn_loss(y, y, epsilon, n, niter, cuda)
    return 2 * Wxy - Wxx - Wyy


def sinkhorn_loss(x, y, epsilon, n, niter, cuda0):
    """
    Given two emprical measures with n points each with locations x and y
    outputs an approximation of the OT cost with regularization parameter epsilon
    niter is the max. number of steps in sinkhorn loop
    """

    # The Sinkhorn algorithm takes as input three variables :
    C = cost_matrix(x, y) # Wasserstein cost function

    # both marginals are fixed with equal weights
    mu = 1. / n * torch.ones(n)
    nu = 1. / n * torch.ones(n)
    if True:
        mu = mu.cuda(cuda0)
        nu = nu.cuda(cuda0)
        # mu = mu.to(cuda)
        # nu = nu.to(cuda)

    # Parameters of the Sinkhorn algorithm.
    rho = 1  # (.5) **2          # unbalanced transport
    tau = -.8  # nesterov-like acceleration
    lam = rho / (rho + epsilon)  # Update exponent
    thresh = 10**(-1)  # stopping criterion

    # Elementary operations .....................................................................
    def ave(u, u1):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1

    def M(u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(1) + v.unsqueeze(0)) / epsilon

    def lse(A):
        "log-sum-exp"
        return torch.log(torch.exp(A).sum(1, keepdim=True) + 1e-6)  # add 10^-6 to prevent NaN

    # Actual Sinkhorn loop ......................................................................
    u, v, err = 0. * mu, 0. * nu, 0.
    actual_nits = 0  # to check if algorithm terminates because of threshold or max iterations reached

    for i in range(niter):
        u1 = u  # useful to check the update
        u = epsilon * (torch.log(mu) - lse(M(u, v)).squeeze()) + u
        v = epsilon * (torch.log(nu) - lse(M(u, v).t()).squeeze()) + v
        # accelerated unbalanced iterations
        # u = ave( u, lam * ( epsilon * ( torch.log(mu) - lse(M(u,v)).squeeze()   ) + u ) )
        # v = ave( v, lam * ( epsilon * ( torch.log(nu) - lse(M(u,v).t()).squeeze() ) + v ) )
        err = (u - u1).abs().sum()

        actual_nits += 1
        if err.item() < thresh:
            break
    U, V = u, v
    pi = torch.exp(M(U, V))  # Transport plan pi = diag(a)*K*diag(b)
    cost = torch.sum(pi * C)  # Sinkhorn cost

    return cost


def normalized_wasserstein_loss(x, y, C, epsilon, n, niter, cuda):
    _, Wxy = wasserstein_loss(x, y, C, epsilon, n, niter, cuda)
    _, Wxx = wasserstein_loss(x, x, C, epsilon, n, niter, cuda)
    _, Wyy = wasserstein_loss(y, y, C, epsilon, n, niter, cuda)
    return 2 * Wxy - Wxx - Wyy
    

def wasserstein_loss(a, b, C, epsilon, n, niter, cuda0):
    """
    Given two emprical measures with n points each with locations x and y
    outputs an approximation of the OT cost with regularization parameter epsilon
    niter is the max. number of steps in sinkhorn loop
    """

    # The Sinkhorn algorithm takes as input three variables :

    if cuda0:
        mu = a.squeeze().cuda(cuda0)
        nu = b.squeeze().cuda(cuda0)
    else:
        mu = a.squeeze()
        nu = b.squeeze()
    # Parameters of the Sinkhorn algorithm.
    rho = 1  # (.5) **2          # unbalanced transport
    tau = -.8  # nesterov-like acceleration
    lam = rho / (rho + epsilon)  # Update exponent
    thresh = 10**(-1)  # stopping criterion

    # Elementary operations .....................................................................
    def ave(u, u1):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1

    def M(u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(1) + v.unsqueeze(0)) / epsilon

    def lse(A):
        "log-sum-exp"
        return torch.log(torch.exp(A).sum(1, keepdim=True) + 1e-10)  # add 10^-10 to prevent NaN

    # Actual Sinkhorn loop ......................................................................
    u, v, err = 0. * mu, 0. * nu, 0.
    actual_nits = 0  # to check if algorithm terminates because of threshold or max iterations reached

    for i in range(niter):
        u1 = u  # useful to check the update
        # pdb.set_trace()
        u = epsilon * (torch.log(mu + 1e-10) - lse(M(u, v)).squeeze()) + u
        v = epsilon * (torch.log(nu + 1e-10) - lse(M(u, v).t()).squeeze()) + v
        # accelerated unbalanced iterations
        # u = ave( u, lam * ( epsilon * ( torch.log(mu) - lse(M(u,v)).squeeze()   ) + u ) )
        # v = ave( v, lam * ( epsilon * ( torch.log(nu) - lse(M(u,v).t()).squeeze() ) + v ) )
        err = (u - u1).abs().sum()

        # actual_nits += 1
        # if err.item() < thresh:
        #     break
    U, V = u, v

    pi = torch.exp(M(U, V))  # Transport plan pi = diag(a)*K*diag(b)
    cost = torch.sum(pi * C)  # Sinkhorn cost

    return pi, cost

def normalized_parallel_wasserstein_loss(a, b, C, epsilon, n, niter, devices):

    lock1 = False
    lock2 = False
    lock3 = False

    a_1 = a.clone().to(devices[1])
    b_1 = b.clone().to(devices[1])
    C_1 = C.clone().to(devices[1])
    Wxy, lock1 = parallel_wasserstein_loss(a_1, b_1, C_1, epsilon, n, niter, devices[1], lock1)

    a_2 = a.clone().to(devices[2])
    C_2 = C.clone().to(devices[2])
    Wxx, lock2 = parallel_wasserstein_loss(a_2, a_2, C_2, epsilon, n, niter, devices[2], lock2)

    b_3 = b.clone().to(devices[3])
    C_3 = C.clone().to(devices[3])
    Wyy, lock3 = parallel_wasserstein_loss(b_3, b_3, C_3, epsilon, n, niter, devices[3], lock3)
    

    if lock1 and lock2 and lock3:
        Wxy_0 = Wxy.clone().to(devices[0])
        Wxx_0 = Wxx.clone().to(devices[0])
        Wyy_0 = Wyy.clone().to(devices[0])
        return 2 * Wxy_0 - Wxx_0 - Wyy_0

def parallel_wasserstein_loss(a, b, C, epsilon, n, niter, cuda0, lock=False):
    """
    Given two emprical measures with n points each with locations x and y
    outputs an approximation of the OT cost with regularization parameter epsilon
    niter is the max. number of steps in sinkhorn loop
    """
    # num_classes = label.size(1)
    # a: batchsize * num_classes
    # b: batchsize * num_classes
    # C: batchsize * num_classes * num_classes

    if cuda0:
        mu = a.cuda(cuda0)
        nu = b.cuda(cuda0)
    else:
        mu = a
        nu = b

    # Parameters of the Sinkhorn algorithm.
    rho = 1  # (.5) **2          # unbalanced transport
    tau = -.8  # nesterov-like acceleration
    lam = rho / (rho + epsilon)  # Update exponent
    thresh = 10**(-1)  # stopping criterion

    # Elementary operations .....................................................................
    def M(u, v): 
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(2) + v.unsqueeze(1)) / epsilon # batchsize * num_classes * num_classes

    def lse(A):
        "log-sum-exp"
        return torch.log(torch.exp(A).sum(2) + 1e-10) # batchsize * num_classes

    # Actual Sinkhorn loop ......................................................................
    u, v, err = 0. * mu, 0. * nu, 0.
    actual_nits = 0  # to check if algorithm terminates because of threshold or max iterations reached

    for i in range(niter):
        # pdb.set_trace()
        u1 = u  # useful to check the update
        u = epsilon * (torch.log(mu + 1e-10) - lse(M(u, v))) + u
        v = epsilon * (torch.log(nu + 1e-10) - lse(M(u, v).transpose(1,2))) + v
        # err = (u - u1).abs().sum() / u.size(0)
        err = (u-u1).abs().max()
        # actual_nits += 1
        # if err.item() < thresh:
        #     break
        
    U, V = u, v
    # sum(0) is to summarize all the dim of batchsize
    pi = torch.exp(M(U, V)).sum(0)  # Transport plan pi = diag(a)*K*diag(b) 
    # cost = torch.sum(pi * C)  # Sinkhorn cost

    lock = True

    return pi, lock


def normalized_parallel_wasserstein_loss_one_gpu(a, b, C, epsilon, n, niter, cuda0):

    Wxy = parallel_wasserstein_loss_one_gpu(a, b, C, epsilon, n, niter, cuda0)
    Wxx = parallel_wasserstein_loss_one_gpu(a, a, C, epsilon, n, niter, cuda0)
    Wyy = parallel_wasserstein_loss_one_gpu(b, b, C, epsilon, n, niter, cuda0)

    return 2 * Wxy - Wxx - Wyy



def parallel_wasserstein_loss_one_gpu(a, b, C, epsilon, n, niter, cuda0):
    """
    Given two emprical measures with n points each with locations x and y
    outputs an approximation of the OT cost with regularization parameter epsilon
    niter is the max. number of steps in sinkhorn loop
    """
    # num_classes = label.size(1)
    # a: batchsize * num_classes
    # b: batchsize * num_classes
    # C: batchsize * num_classes * num_classes

    if cuda0:
        mu = a.cuda(cuda0)
        nu = b.cuda(cuda0)
    else:
        mu = a
        nu = b

    # Parameters of the Sinkhorn algorithm.
    rho = 1  # (.5) **2          # unbalanced transport
    tau = -.8  # nesterov-like acceleration
    lam = rho / (rho + epsilon)  # Update exponent
    thresh = 10**(-1)  # stopping criterion

    # Elementary operations .....................................................................
    def M(u, v): 
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(2) + v.unsqueeze(1)) / epsilon # batchsize * num_classes * num_classes

    def lse(A):
        "log-sum-exp"
        return torch.log(torch.exp(A).sum(2) + 1e-10) # batchsize * num_classes

    # Actual Sinkhorn loop ......................................................................
    u, v, err = 0. * mu, 0. * nu, 0.
    actual_nits = 0  # to check if algorithm terminates because of threshold or max iterations reached

    for i in range(niter):
        # pdb.set_trace()
        u1 = u  # useful to check the update
        u = epsilon * (torch.log(mu + 1e-10) - lse(M(u, v))) + u
        v = epsilon * (torch.log(nu + 1e-10) - lse(M(u, v).transpose(1,2))) + v
        # err = (u - u1).abs().sum() / u.size(0)
        err = (u-u1).abs().max()
        # actual_nits += 1
        # if err.item() < thresh:
        #     break
        
    U, V = u, v
    # sum(0) is to summarize all the dim of batchsize
    pi = torch.exp(M(U, V)).sum(0)  # Transport plan pi = diag(a)*K*diag(b) 
    # cost = torch.sum(pi * C)  # Sinkhorn cost

    return pi


def cost_matrix(x, y, p=1):
    "Returns the matrix of $|x_i-y_j|^p$."
    x_col = x.unsqueeze(1)
    y_lin = y.unsqueeze(0)
    C = torch.sum((torch.abs(x_col - y_lin)) ** p, 2)
    C /= C.max()
    # c = torch.mm(x, y.t())
    
    return C


def euclidean_distances(X, Y, squared=False):
    """
    Considering the rows of X (and Y=X) as vectors, compute the
    distance matrix between each pair of vectors.
    Parameters
    ----------
    X : {array-like}, shape (n_samples_1, n_features)
    Y : {array-like}, shape (n_samples_2, n_features)
    squared : boolean, optional
        Return squared Euclidean distances.
    Returns
    -------
    distances : {array}, shape (n_samples_1, n_samples_2)
    """
    XX = np.einsum('ij,ij->i', X, X)[:, np.newaxis]
    YY = np.einsum('ij,ij->i', Y, Y)[np.newaxis, :]
    distances = np.dot(X, Y.T)
    distances *= -2
    distances += XX
    distances += YY
    np.maximum(distances, 0, out=distances)
    if X is Y:
        # Ensure that distances between vectors and themselves are set to 0.0.
        # This may not be the case due to floating point rounding errors.
        distances.flat[::distances.shape[0] + 1] = 0.0
    return distances if squared else np.sqrt(distances, out=distances)


def dist(x1, x2=None, metric='sqeuclidean'):
    """Compute distance between samples in x1 and x2 using function scipy.spatial.distance.cdist

    Parameters
    ----------

    x1 : ndarray, shape (n1,d)
        matrix with n1 samples of size d
    x2 : array, shape (n2,d), optional
        matrix with n2 samples of size d (if None then x2=x1)
    metric : str | callable, optional
        Name of the metric to be computed (full list in the doc of scipy),  If a string,
        the distance function can be 'braycurtis', 'canberra', 'chebyshev', 'cityblock',
        'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski',
        'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
        'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'.


    Returns
    -------

    M : np.array (n1,n2)
        distance matrix computed with given metric

    """
    if x2 is None:
        x2 = x1
    if metric == "sqeuclidean":
        return euclidean_distances(x1, x2, squared=True)
    return cdist(x1, x2, metric=metric)


def dist0(n, method='lin_square'):
    """Compute standard cost matrices of size (n, n) for OT problems

    Parameters
    ----------
    n : int
        Size of the cost matrix.
    method : str, optional
        Type of loss matrix chosen from:

        * 'lin_square' : linear sampling between 0 and n-1, quadratic loss

    Returns
    -------
    M : ndarray, shape (n1,n2)
        Distance matrix computed with given metric.
    """
    res = 0
    if method == 'lin_square':
        x = np.arange(n, dtype=np.float64).reshape((n, 1))
        res = dist(x, x)
    return res


def geometricBar(weights, alldistribT):
    """return the weighted geometric mean of distributions"""
    assert(len(weights) == alldistribT.shape[1])
    return np.exp(np.dot(np.log(alldistribT), weights.T))


def geometricMean(alldistribT):
    """return the  geometric mean of distributions"""
    return np.exp(np.mean(np.log(alldistribT), axis=1))

# default numItermax=1000
def barycenter(A, M, reg, weights=None, numItermax=100,
               stopThr=1e-4, verbose=False, log=False):
    r"""Compute the entropic regularized wasserstein barycenter of distributions A

     The function solves the following optimization problem:

    .. math::
       \mathbf{a} = arg\min_\mathbf{a} \sum_i W_{reg}(\mathbf{a},\mathbf{a}_i)

    where :

    - :math:`W_{reg}(\cdot,\cdot)` is the entropic regularized Wasserstein distance (see ot.bregman.sinkhorn)
    - :math:`\mathbf{a}_i` are training distributions in the columns of matrix :math:`\mathbf{A}`
    - reg and :math:`\mathbf{M}` are respectively the regularization term and the cost matrix for OT

    Parameters
    ----------
    A : ndarray, shape (d,n)
        n training distributions a_i of size d
    M : ndarray, shape (d,d)
        loss matrix   for OT
    reg : float
        Regularization term >0
    weights : ndarray, shape (n,)
        Weights of each histogram a_i on the simplex (barycentric coodinates)
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True


    Returns
    -------
    a : (d,) ndarray
        Wasserstein barycenter
    log : dict
        log dictionary return only if log==True in parameters

    """

    if weights is None:
        weights = np.ones(A.shape[1]) / A.shape[1]
    else:
        assert(len(weights) == A.shape[1])

    if log:
        log = {'err': []}

    # M = M/np.median(M) # suggested by G. Peyre
    K = np.exp(-M / reg)

    cpt = 0
    err = 1

    UKv = np.dot(K, np.divide(A.T, np.sum(K, axis=0)).T)
    u = (geometricMean(UKv) / UKv.T).T

    while (err > stopThr and cpt < numItermax):
        cpt = cpt + 1
        UKv = u * np.dot(K, np.divide(A, np.dot(K, u)))
        u = (u.T * geometricBar(weights, UKv)).T / UKv

        if cpt % 10 == 1:
            err = np.sum(np.std(UKv, axis=1))

            # log and verbose print
            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

    if log:
        log['niter'] = cpt
        return geometricBar(weights, UKv), log
    else:
        return geometricBar(weights, UKv)


def get_mid_distribution(initial_distribution, targets, num_step=7):
    batchsize = targets.size(0)
    num_classes = targets.size(1)

    targets_0 = torch.zeros(batchsize, num_classes)
    targets_1 = torch.zeros(batchsize, num_classes)
    targets_2 = torch.zeros(batchsize, num_classes)
    targets_3 = torch.zeros(batchsize, num_classes)
    targets_4 = torch.zeros(batchsize, num_classes)

    for i in range(0, batchsize):
        init_dist = initial_distribution[i, :].cpu().numpy().reshape(num_classes)
        tag = targets[i, :].cpu().numpy().reshape(num_classes)
        
        A = np.vstack((init_dist, tag)).T
        n = A.shape[0]
        n_distributions = A.shape[1]
        
        M = dist0(n)
        M /= M.max()

        alpha_list = np.linspace(0, 1, num_step)
        reg = 1e-3
        B_wass = np.zeros((n, num_step))
        
        for j in range(0, num_step):
            alpha = alpha_list[j]
            weights = np.array([1-alpha, alpha])
            B_wass[:, j] = barycenter(A, M, reg, weights)

        # the B[:, 0] is the initial_distribution
        targets_0[i, :] = torch.from_numpy(B_wass[:, 1]).float()
        targets_1[i, :] = torch.from_numpy(B_wass[:, 2]).float()
        targets_2[i, :] = torch.from_numpy(B_wass[:, 3]).float()
        targets_3[i, :] = torch.from_numpy(B_wass[:, 4]).float()
        targets_4[i, :] = torch.from_numpy(B_wass[:, 5]).float()

    if initial_distribution.is_cuda:
        targets_0 = targets_0.float().cuda()
        targets_1 = targets_1.float().cuda()
        targets_2 = targets_2.float().cuda()
        targets_3 = targets_3.float().cuda()
        targets_4 = targets_4.float().cuda()

    return targets_0, targets_1, targets_2, targets_3, targets_4