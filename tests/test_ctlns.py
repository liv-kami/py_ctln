# Import the package for testing

from src.py_ctln import CTLN

# ─────────────────────────── The Tests ────────────────────────────

def test_set_params():
    CTLN.set_params(epsilon=0.26,delta=0.51)
    assert CTLN.epsilon == 0.26
    assert CTLN.delta == 0.51
    CTLN.set_params(epsilon=0.25,delta=0.5)

def test_w_mat():
    sA = [[0,0,1],[1,0,0],[0,1,0]]
    W = CTLN.get_w_mat(sA)
    W_ideal = [[0,-1.5,-0.75],[-0.75,0,-1.5],[-1.5,-0.75,0]]
    assert (W == W_ideal).all()

def test_check_fp():
    sA = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    sig1 = [0,1,2]
    sig2 = [0,1]

    is_fp1, x_fp1 = CTLN.check_fp(sA,sig1)
    is_fp2, x_fp2 = CTLN.check_fp(sA,sig2)

    assert is_fp1
    assert not is_fp2
    assert (x_fp2==[[4],[-2],[0]]).all()

def test_check_stability():
    sA = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    sig1 = [0,1,2]
    sig2 = [0,1]
    stable1,eigvals1 = CTLN.check_stability(sA,sig1)
    stable2,eigvals2 = CTLN.check_stability(sA,sig2)
    print(eigvals1)
    print(eigvals2)
    print(stable1,stable2)

    assert (eigvals1 == -1).all()
    assert (eigvals2 == -1).all()
    assert stable1
    assert stable2

def test_get_fp():
    sA = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    fixpts, supports, stability = CTLN.get_fp(sA)
    assert (supports == [[1,2,3]])

def test_uid():
    sA = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    assert CTLN.is_uid(sA)
    sA2 = [[0, 0, 1], [0, 0, 0], [0, 1, 0]]
    assert not CTLN.is_uid(sA2)

def test_uod():
    sA = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    assert CTLN.is_uod(sA)
    sA2 = [[0, 0, 1], [0, 0, 0], [0, 1, 0]]
    assert not CTLN.is_uod(sA2)

def test_is_core():
    sA = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    assert CTLN.is_core(sA)
    sA2 = [[0, 0, 1], [0, 0, 0], [0, 1, 0]]
    assert not CTLN.is_core(sA2)

def test_is_permitted():
    sA = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    assert CTLN.is_permitted(sA)
    sA2 = [[0, 0, 1], [0, 0, 1], [1, 1, 0]]
    assert CTLN.is_permitted(sA2)
    sA3 = [[0, 0, 1], [0, 0, 0], [0, 1, 0]]
    assert not CTLN.is_permitted(sA3)

def test_domination():
    sA = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    assert len(
        CTLN.find_graphical_domination(
            sA,
            types_to_look_for=['outside-in','inside-in']
        )[1]
    )==6

def test_strongly_connected():
    pass

def test_weakly_connected():
    pass

def test_strongly_core():
    strongly_core_fives = [a for a in CTLN.collections.core_n(5) if
                           CTLN.is_strongly_core(a)]
    assert len(strongly_core_fives) == 26

def test_is_hamiltonian():
    a = [
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
    ]
    b = [
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ]
    c = [
        [0, 0, 1],
        [1, 0, 1],
        [1, 0, 0]
    ]
    assert CTLN.is_hamiltonian(a)[0] == True
    assert CTLN.is_hamiltonian(b)[0] == True
    assert CTLN.is_hamiltonian(c)[0] == False

    assert CTLN.is_hamiltonian(a)[1] == [[1,2,3],[2,3,1],[3,1,2]]
    assert CTLN.is_hamiltonian(b)[1] == [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
    assert CTLN.is_hamiltonian(c)[1] == []

def test_parallel_run():
    """Test the parallel_run method to ensure it works correctly."""
    
    # Create a set of test matrices
    matrices = [
        [[0, 0, 1], [1, 0, 0], [0, 1, 0]],  # core motif
        [[0, 0, 1], [0, 0, 0], [0, 1, 0]],  # not core
        [[0, 1, 1], [1, 0, 1], [1, 1, 0]],  # complete graph (core)
        [[0, 0, 1], [1, 0, 1], [0, 0, 0]],  # not strongly connected
    ]
    
    # Test 1: parallel_run produces same results as sequential execution
    # for is_core method
    parallel_results = CTLN.parallel_run(matrices, CTLN.is_core)
    sequential_results = [CTLN.is_core(m) for m in matrices]
    assert parallel_results == sequential_results, \
        "Parallel results don't match sequential results for is_core"
    
    # Test 2: Verify specific results for is_core
    assert parallel_results == [True, False, True, False], \
        "is_core results are incorrect"
    
    # Test 3: Test with num_processes parameter
    parallel_single = CTLN.parallel_run(matrices, CTLN.is_core, num_processes=1)
    assert parallel_single == parallel_results, \
        "Results differ when using num_processes=1"
    
    # Test 4: Test with complex return values (get_fp returns nested lists)
    parallel_fp = CTLN.parallel_run(matrices, CTLN.get_fp)
    assert len(parallel_fp) == len(matrices), \
        "Number of results doesn't match number of matrices"
    assert all(len(result) == 3 for result in parallel_fp), \
        "get_fp results should have 3 elements (fixpts, supports, stability)"
    
    # Test 5: Order preservation - results should be in same order as input
    sequential_fp = [CTLN.get_fp(m) for m in matrices]
    for i, (parallel, sequential) in enumerate(zip(parallel_fp, sequential_fp)):
        assert parallel[1] == sequential[1], \
            f"Support order not preserved at matrix index {i}"
    
    # Test 6: Single matrix
    single_result = CTLN.parallel_run([matrices[0]], CTLN.is_core)
    assert single_result == [True], \
        "Single matrix parallel execution failed"

def test_is_circulant():
    sA = [
        [0,0,0,0,1],
        [1,0,0,0,0],
        [0,1,0,0,0],
        [0,0,1,0,0],
        [0,0,0,1,0]
    ]

    assert CTLN.is_circulant(sA)[0] == True
    assert CTLN.is_circulant(sA)[1] == [1,2,3,4,5]
    assert CTLN.is_circulant(sA)[2] == [1]

    sA = [
        [0,0,0,1,1],
        [1,0,0,0,1],
        [1,1,0,0,0],
        [0,1,1,0,0],
        [0,0,1,1,0]
    ]
    
    assert CTLN.is_circulant(sA)[0] == True
    assert CTLN.is_circulant(sA)[1] == [1,2,3,4,5]
    assert CTLN.is_circulant(sA)[2] == [1,2]

    sA = [
        [0,0,0,0,1],
        [1,0,0,0,0],
        [0,1,0,0,0],
        [0,0,1,0,1],
        [0,0,0,0,0]
    ]

    assert CTLN.is_circulant(sA)[0] == False
    assert CTLN.is_circulant(sA)[1] == []
    assert CTLN.is_circulant(sA)[2] == []

def test_is_clique_union():
    sA = [
        [0,1,1],
        [1,0,1],
        [1,1,0]
    ]

    assert CTLN.is_clique_union(sA,[[0],[1],[2]]) == True
    assert CTLN.is_clique_union(sA,[[0,1],[2]]) == True

    sA = [
        [0,1,1],
        [1,0,0],
        [1,1,0]
    ]

    assert CTLN.is_clique_union(sA,[[0],[1],[2]]) == False
    assert CTLN.is_clique_union(sA,[[0,1],[2]]) == False
    assert CTLN.is_clique_union(sA,[[1,2],[0]]) == True

