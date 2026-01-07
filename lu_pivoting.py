import numpy as np
from algo3 import lu_pivoting

if __name__ == "__main__":
    np.random.seed(42)
    n_tests = 10; p = 0.2; n = 1000; success = 0
    import time
    start_time = time.time()    
    for _ in range(n_tests):
        A_rand = np.random.rand(n,n)
        res = lu_pivoting(A_rand)
        if (res and np.allclose(res[0] @ A_rand @ res[1], res[2] @ res[3]) 
            and np.allclose(res[2], np.tril(res[2])) 
            and np.allclose(res[3], np.triu(res[3]))): success += 1
    end_time = time.time()
    print(f"  Time taken for {n_tests} tests: {end_time - start_time:.4f} seconds")            
    print(f"PASS: {success}/{n_tests} tests passed.")