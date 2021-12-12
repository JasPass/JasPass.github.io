"""
Code to compute coupling-coefficients between different SO(3) irreps,
using the approach of: http://arxiv.org/abs/1006.2875v1
"""

from sympy import sqrt, Rational, Matrix
# from scipy import linalg
from time import time


def MatrixElements(J, M, s):
    """Matrix-elements for J_p / J_m"""

    if s == 1:
        return sqrt((J - M) * (J + M + 1))
    elif s == -1:
        return sqrt((J + M) * (J - M + 1))


def IsValid(M1, M2, M):
    """Condition for M_1 x M_2 --> M for SO(2)"""

    if M1 + M2 == M:
        return True
    else:
        return False


def Indexing(J1, J2, J):
    """Gives the order of coupling-coefficients,
       as returned by CouplingCoefficients()"""

    ordering = []

    for N1 in range(2 * J1 + 1):
        for N2 in range(2 * J2 + 1):
            for N in range(2 * J + 1):

                if IsValid(N1 - J1, N2 - J2, N - J):
                    ordering.append([N1 - J1, N2 - J2, N - J])

    return ordering


def Coupling_Conditions(J1, M1, J2, M2, J, M, M_S, M_T):
    """All relations for given M1 and M2 in the lattice

    We first choose a set of three dummy SO(2) irrep labels: (N1, N2, N),
    given the following branching conditions:

    J1 --> N1 , J2 --> N2 , J --> N

    Secondly, we choose SO(3) irrep labels (M1_S, M2_S),
    given the following branching conditions:

    M_T x M1_S  --> M1 , M_T x M2_S --> M2

    """

    # List to store relation between reduced
    # coupling-coefficients  (as a row of the matrix)
    relation = []

    # Loop through all the SO(2) irreps in J1, J2, J
    for i in range(2 * J1 + 1):
        for j in range(2 * J2 + 1):
            for k in range(2 * J + 1):

                # Construct the SO(2) irrep labels
                N1 = i - J1
                N2 = j - J2
                N = k - J

                # < J1, M1 | T_(M_T) | ~ < J1, M1 - M_T |
                # Note: we should loop over all M1_s,
                # consistent with: M_T x M1_S  --> M1
                M1_S = M1 - M_T

                # < J2, M2 | T_(M_T) | ~ < J2, M2 - M_T |
                # Note: we should loop over all M2_s,
                # consistent with: M_T x M2_S  --> M2
                M2_S = M2 - M_T

                # Checks if:
                # N1 x N2 --> N
                if IsValid(N1, N2, N):

                    # Checks if the coupling coefficient:
                    # (J1, N1; J2, N2 | J, N),
                    # match the coupling coefficient:
                    # (J1, M1; J2, M2 | J, N_S)
                    if N1 == M1 and N2 == M2 and N == M_S:

                        # The coefficient for the coupling coefficient:
                        # (J1, M1; J2, M2 | J, M_S)
                        # is:
                        # < J, M_S | T_(M_T) | J, M >
                        relation.append(MatrixElements(J, M, M_T))

                    # Checks if the coupling coefficient:
                    # (J1, N1; J2, N2 | J, N),
                    # match the coupling coefficient:
                    # (J1, M1_S; J2, M2 | J, M)
                    elif N1 == M1_S and N2 == M2 and N == M:

                        # The coefficient for the coupling coefficient:
                        # (J1, M1_S; J2, M2 | J, M)
                        # is:
                        # < J1, M1 | T_(M_T) | J1, M1_S >
                        relation.append(-MatrixElements(J1, M1_S, M_T))

                    # Checks if the coupling coefficient:
                    # (J1, N1; J2, N2 | J, N),
                    # match the coupling coefficient:
                    # (J1, M1; J2, M2_S | J, M)
                    elif N1 == M1 and N2 == M2_S and N == M:

                        # The coefficient for the coupling coefficient:
                        # (J1, M1; J2, M2_S | J, M)
                        # is:
                        # < J2, M2 | T_(M_T) | J2, M2_S >
                        relation.append(-MatrixElements(J2, M2_S, M_T))

                    # The coupling coefficient:
                    # (J1, N1; J2, N2 | J, N),
                    # does not match any coupling coefficient
                    # in the current relation
                    else:
                        relation.append(0)

    # return the coefficients of the current relation
    return relation


def Normalization(NullVector, J1, J2, J):
    """Computes the normalization for the coupling-coefficients"""

    total = 0
    ordering = Indexing(J1, J2, J)

    for n in range(len(NullVector)):

        if ordering[n][2] == J:

            total += NullVector[n]**2

    return total


def Linear_System_Matrix(J1, J2, J):
    """Computes the non normalized coupling-coefficients

    We first choose a set of four SO(2) irrep labels: (M1, M2, M, M_s),
    and a generator which is in SO(3) but not in SO(2): T_(M_T), M_T = +1, -1,
    given the following branching conditions:

    J1 --> M1 , J2 --> M2 , J --> M

    M1 x M2 --> M_S , M_T x M --> M_S
    """

    # List to store relations between reduced
    # coupling-coefficients (as a matrix)
    relations = []

    # Loop through all the SO(2) irreps in J1, J2, J
    for i in range(2 * J1 + 1):
        for j in range(2 * J2 + 1):
            for k in range(2 * J + 1):

                # Loop over the (+ / -) for:
                # M x M_T --> M_S
                for M_T in [1, -1]:

                    # Construct the SO(2) irrep labels
                    M1 = i - J1
                    M2 = j - J2
                    M = k - J

                    # T_(M_T) | J, M > ~ | J, M + M_T >
                    # Note: we should loop over all M_s,
                    # consistent with: M_T x M  --> M_S
                    M_S = M + M_T

                    # Checks if:
                    # M1 x M2 --> M_s
                    if IsValid(M1, M2, M_S):

                        # Get the relation between reduced coupling-coefficients
                        # for the SO(2) irrep labels: (M1, M2, M, M_s)
                        relation = Coupling_Conditions(J1, M1, J2, M2, J, M, M_S, M_T)
                        relations.append(relation)

    # Return the list of all relation between reduced
    # coupling-coefficients, as a coefficient matrix
    return relations


def CouplingCoefficients(J1, J2, J):
    """Computes the normalized coupling coefficients"""

    M = Matrix(Linear_System_Matrix(J1, J2, J))

    print('Im done!')

    NullVector = M.nullspace()[0]

    Normalized = NullVector / sqrt(Normalization(NullVector, J1, J2, J))

    return list(Normalized)


# Print out the coupling-coefficients for J1 = 1/2 x J2 = 1/2 --> J = 1
L1 = Rational(1, 2)
L2 = Rational(1, 2)
L = Rational(1, 1)

# NullVector1 = MasterMatrix(L1, L2, L)
# print(NullVector1)
# print(len(NullVector1))

Start_Time = time()

print('The Coupling-Coefficients: ', CouplingCoefficients(L1, L2, L))
print(Linear_System_Matrix(L1, L2, L))

End_Time = time()

# print('The indexing: ', Indexing(L1, L2, L))
# print('The time of computation: ', round(End_Time - Start_Time, 2), 'seconds')
