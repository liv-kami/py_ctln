# Import the package for testing

from src.py_ctln import CTLN

# ─────────────────────────── The Tests ────────────────────────────

#TODO: Create testing functions

#TODO: add testing to workflow on github (add hatch test)

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
    pass

#TODO: Finish the testing functions