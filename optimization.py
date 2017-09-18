import numpy as np
from collections import defaultdict, deque  # Use this for effective implementation of L-BFGS
from utils import get_line_search_tool
from datetime import datetime
import scipy
from scipy import sparse
from scipy import optimize


def conjugate_gradients(matvec, b, x_0, tolerance=1e-4, max_iter=None, trace=False, display=False):
    """
    Solves system Ax=b using Conjugate Gradients method.

    Parameters
    ----------
    matvec : function
        Implement matrix-vector product of matrix A and arbitrary vector x
    b : 1-dimensional np.array
        Vector b for the system.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
        Stop optimization procedure and return x_k when:
         ||Ax_k - b||_2 <= tolerance * ||b||_2
    max_iter : int, or None
        Maximum number of iterations. if max_iter=None, set max_iter to n, where n is
        the dimension of the space
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display:  bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['residual_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    x_k = np.copy(x_0).astype(float)
    # TODO: Implement Conjugate Gradients method.
    
    
    t0 = datetime.now()
    
    if max_iter == None:
        max_iter = x_k.size
    
    for it in range(max_iter + 1):
        # Oracle
        if it > 0:
            Ax_k = Ax_k + alpha * Ad_k
            g_k_prev = g_k.copy()
        else:
            Ax_k = matvec(x_k)
        
        g_k = Ax_k - b    
        
        # Fill trace data
        if display:
            print(it)
        if trace:
            history['time'].append((datetime.now() - t0).total_seconds())
            history['residual_norm'].append(np.linalg.norm(g_k))
            if x_k.size < 3:
                history['x'].append(x_k)
        
        # Criterium
        if np.linalg.norm(Ax_k - b) <= tolerance * np.linalg.norm(b):
            break
        
        if it >= max_iter:
            return x_k, 'iterations_exceeded', history
        
        # Direction
        if it > 0:
            d_k = - g_k + (np.dot(g_k, g_k) / np.dot(g_k_prev, g_k_prev)) * d_k
        else:
            d_k = - g_k
        
        # line search
        Ad_k = matvec(d_k)
        alpha = np.dot(g_k, g_k) / np.dot(d_k, Ad_k)
        
        x_k = x_k + alpha * d_k
        
    
    return x_k, 'success', history

def conjugate_gradients_call(oracle, x_0, tolerance=1e-4, max_iter=500, display=False, trace=False):

    matvec = lambda x: oracle.A.dot(x)
    return conjugate_gradients (matvec, oracle.b, x_0, tolerance, max_iter, trace, display)
    

def nonlinear_conjugate_gradients(oracle, x_0, tolerance=1e-4, max_iter=500,
          line_search_options={'method': 'Wolfe', 'c1': 1e-4, 'c2': 0.2},
          display=False, trace=False):
    """
    Nonlinear conjugate gradient method for optimization (Polak--Ribiere version).

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func() and .grad() methods implemented for computing
        function value and its gradient respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to the student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidean norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    
    t0 = datetime.now()
    grad_x_0 = oracle.grad(x_0)
    norm_grad_x_0 = np.linalg.norm(grad_x_0)
    
    for it in range(max_iter + 1):
        
        # Oracle
        if it > 0:
            grad_x_k_prev = grad_x_k.copy()
            norm_grad_x_k_prev = norm_grad_x_k            
        
        f_x_k = oracle.func(x_k)
        grad_x_k = oracle.grad(x_k)        
        norm_grad_x_k = np.linalg.norm(grad_x_k)
            
        # Debug info
        if display:
            print(it)
        
        # Fill trace data
        if trace:
            history['time'].append((datetime.now() - t0).total_seconds())
            history['grad_norm'].append(norm_grad_x_k)
            history['func'].append(f_x_k)
            if x_k.size < 3:
                history['x'].append(x_k)
        
        # Criterium
        if norm_grad_x_k * norm_grad_x_k <= tolerance * norm_grad_x_0 * norm_grad_x_0:
            break
        
        if it >= max_iter:
            return x_k, 'iterations_exceeded', history
        
        # Direction
        if it > 0:
            betta = np.dot(grad_x_k, (grad_x_k - grad_x_k_prev)) / (norm_grad_x_k_prev * norm_grad_x_k_prev)
            #betta = (norm_grad_x_k * norm_grad_x_k) / (norm_grad_x_k_prev * norm_grad_x_k_prev)
            
            
            d_k = -grad_x_k + betta * d_k
        else:
            d_k = -grad_x_k

        # Line search
        alpha = line_search_tool.line_search (oracle, x_k, d_k)

        x_k = x_k + alpha * d_k
    
    
    return x_k, 'success', history


def lbfgs(oracle, x_0, tolerance=1e-4, max_iter=500, memory_size=10, 
          line_search_options=None, display=False, trace=False):
    """
    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func() and .grad() methods implemented for computing
        function value and its gradient respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    memory_size : int
        The length of directions history in L-BFGS method.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    t0 = datetime.now()
    
    x_k = np.copy(x_0)
    grad_x_0 = oracle.grad(x_0)
    norm_grad_x_0 = np.linalg.norm(grad_x_0)
    
    memory = []
    
    for it in range(max_iter + 1):
        f_x_k = oracle.func(x_k)
        
        if it > 0:
            grad_x_k_prev = grad_x_k.copy ()
            grad_x_k = oracle.grad(x_k)
            
            s_k_prev = x_k - x_k_prev
            y_k_prev = grad_x_k - grad_x_k_prev
            
            chop_size = np.minimum(memory_size - 1, len(memory))
            if chop_size < len(memory):
                memory = memory[len(memory) - chop_size:]
                memory = memory[len(memory) - chop_size:]
            
            memory.append((s_k_prev, y_k_prev))
            
        else:
            grad_x_k = oracle.grad(x_k)
    
        norm_grad_x_k = np.linalg.norm(grad_x_k)

        #Fill trace data
        if display:
            print(it)
        if trace:
            history['time'].append((datetime.now() - t0).total_seconds())
            history['grad_norm'].append(norm_grad_x_k)
            history['func'].append(f_x_k)
            if x_k.size < 3:
                history['x'].append(x_k)
        
        if norm_grad_x_k * norm_grad_x_k <= tolerance * norm_grad_x_0 * norm_grad_x_0:
            break
        
        if it >= max_iter:
            return x_k, 'iterations_exceeded', history
                
        d_k = -grad_x_k
        if it > 0:
            mu = np.zeros(len(memory))
            for index in range(len(memory)):
                i = len(memory) - index - 1
                s_i, y_i = memory[i]
                mu[i] = np.dot(s_i, d_k) / np.dot(s_i, y_i)
                d_k = d_k - mu[i] * y_i
            s_k_prev, y_k_prev = memory[len(memory) - 1]
            d_k = (np.dot(s_k_prev, y_k_prev) / np.dot(y_k_prev, y_k_prev)) * d_k
            for i in range(len(memory)):
                s_i, y_i = memory[i]
                betta = np.dot (y_i, d_k) / np.dot (s_i, y_i)
                d_k = d_k + (mu[i] - betta) * s_i
        
        alpha = line_search_tool.line_search(oracle, x_k, d_k)
        
        x_k_prev = x_k.copy ()
        x_k = x_k + alpha * d_k

    
    return x_k, 'success', history


def hessian_free_newton(oracle, x_0, tolerance=1e-4, max_iter=500, 
                        line_search_options=None, display=False, trace=False):
    """
    Hessian Free method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess_vec() methods implemented for computing
        function value, its gradient and matrix product of the Hessian times vector respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    
    t0 = datetime.now()
    
    grad_x_0 = oracle.grad(x_0)
    norm_grad_x_0 = np.linalg.norm(grad_x_0)

    for it in range(max_iter + 1):
        # Oracle
        f_x_k = oracle.func(x_k)
        grad_x_k = oracle.grad(x_k)
        norm_grad_x_k = np.linalg.norm(grad_x_k)
        
        # Fill trace data
        if display:
            print(it)
        if trace:
            history['time'].append((datetime.now() - t0).total_seconds())
            history['grad_norm'].append(norm_grad_x_k)
            history['func'].append(f_x_k)
            if x_k.size < 3:
                history['x'].append(x_k)
        
        # Criterium
        if norm_grad_x_k * norm_grad_x_k <= tolerance * norm_grad_x_0 * norm_grad_x_0:
            break
        
        if it >= max_iter:
            return x_k, 'iterations_exceeded', history
        
        # Direction
        eta_k = np.minimum (0.5, np.sqrt(norm_grad_x_k))
        d_k = -grad_x_k 
        matvec = lambda v: oracle.hess_vec (x_k, v)
        
        condition=False
        while condition == False:
            d_k, msg_cg, history_cg = conjugate_gradients(matvec, -grad_x_k, d_k, tolerance = eta_k)
            if np.dot(d_k, grad_x_k) < 0:
                condition = True
            eta_k = eta_k / 10.
                
        # Line Search
        alpha = line_search_tool.line_search(oracle, x_k, d_k)
        
        x_k = x_k + alpha * d_k

    return x_k, 'success', history

def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    """
    Gradien descent optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively.
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format and is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = gradient_descent(oracle, np.zeros(5), line_search_options={'method': 'Armijo', 'c1': 1e-4})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    
    t0=datetime.now()
    norm_grad0=np.linalg.norm(oracle.grad(x_0))    
    
    for iteration in range(max_iter + 1):
        #Oracle
        grad_k=oracle.grad(x_k)
        norm_grad_k=np.linalg.norm (grad_k)
        
        #Fill trace data
        if trace:
            history['time'].append((datetime.now() - t0).total_seconds())
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(norm_grad_k)
            if x_k.size < 3:
                history['x'].append(x_k)
        
        if display==True:
            print(u"debug info")
        
        #Criterium
        if norm_grad_k*norm_grad_k <= tolerance * norm_grad0*norm_grad0:
            break;
        if iteration == max_iter:
            return x_k, 'iterations_exceeded', history
        
        #Compute direction
        d_k = -grad_k
        
        #Line search
        if iteration == 0:
            alpha = line_search_tool.line_search (oracle, x_k, d_k)
        else:
            alpha = line_search_tool.line_search (oracle, x_k, d_k, 2. * alpha)
        
        if alpha == None:
            return x_k, 'computational_error', history
        
        #Update x_k
        x_k = x_k + alpha * d_k

    return x_k, 'success', history

def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    """
    Newton's optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively. If the Hessian
        returned by the oracle is not positive-definite method stops with message="newton_direction_error"
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'newton_direction_error': in case of failure of solving linear system with Hessian matrix (e.g. non-invertible matrix).
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = newton(oracle, np.zeros(5), line_search_options={'method': 'Constant', 'c': 1.0})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
                
    t0=datetime.now()
    norm_grad0=np.linalg.norm(oracle.grad(x_0))    
    
    for iteration in range(max_iter + 1):
        #Oracle
        grad_k=oracle.grad(x_k)
        norm_grad_k=np.linalg.norm (grad_k)
        hess_k=oracle.hess(x_k)
        
        #Fill trace data
        if trace:
            history['time'].append((datetime.now() - t0).total_seconds())
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(norm_grad_k)
            if x_k.size < 3:
                history['x'].append(x_k)
        
        if display==True:
            print(u"debug info")
        
        #Criterium
        if norm_grad_k*norm_grad_k <= tolerance * norm_grad0*norm_grad0:
            break;
        if iteration == max_iter:
            return x_k, 'iterations_exceeded', history
        
        #Compute direction
        try:
            L=scipy.linalg.cho_factor(hess_k, lower=True)
            d_k=scipy.linalg.cho_solve(L,-grad_k)
        except:
            return x_k, 'computational_error', history
        
        #Line search
        alpha = line_search_tool.line_search (oracle, x_k, d_k)
        
        if alpha == None:
            return x_k, 'computational_error', history
        
        #Update x_k
        x_k = x_k + alpha * d_k

    return x_k, 'success', history
