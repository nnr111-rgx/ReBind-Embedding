MSG_LEN = 100
CC_K = 3
CC_G = (0o5, 0o7)
CODE_BITS = 204
T = 142
NP = 7
Q = 4
PAD_QARY = 4
L_MAX = 2
QARY_MAX_LEN = T * (1 + L_MAX)
REBIND_MAX_LEN = 2 * QARY_MAX_LEN
TRAIN_SAMPLES = 120000
VAL_SAMPLES = 12000
TEST_SAMPLES = 12000
STAGES = {
    "stage1": {"p_ins": 0.01, "p_del": 0.01},
    "stage2": {"p_ins": 0.02, "p_del": 0.02},
    "stage3": {"p_ins": 0.03, "p_del": 0.03},
}
P_SUB_MIN = 0.01
P_SUB_MAX = 0.05
