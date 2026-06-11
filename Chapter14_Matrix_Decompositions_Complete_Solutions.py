# Chapter 14 Complete Solutions - Matrix Decompositions
# Student: Lexie
# Purpose: Worked solutions for LU, QR, and Cholesky matrix decomposition,
#          plus a practical sensor matrix factorization problem.

import numpy as np
from scipy.linalg import lu
from numpy.linalg import qr, cholesky, matrix_rank, det, norm


def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def problem1_lu_decomposition():
    section("Problem 1 - LU Decomposition")

    A = np.array([
        [2, 1, 1],
        [4, -6, 0],
        [-2, 7, 2]
    ], dtype=float)

    P, L, U = lu(A)
    A_reconstructed = P @ L @ U

    print("A =")
    print(A)
    print("\nP =")
    print(P)
    print("\nL =")
    print(L)
    print("\nU =")
    print(U)
    print("\nP @ L @ U =")
    print(A_reconstructed)
    print("\nReconstruction error =", norm(A - A_reconstructed))

    print("\nExplanation:")
    print("LU decomposition factors A into triangular pieces. L is lower triangular,")
    print("U is upper triangular, and P records any row swaps used for numerical stability.")


def problem2_qr_decomposition():
    section("Problem 2 - QR Decomposition")

    A = np.array([
        [1, 2],
        [3, 4],
        [5, 6]
    ], dtype=float)

    Q, R = qr(A, mode='reduced')
    A_reconstructed = Q @ R

    print("A =")
    print(A)
    print("\nQ =")
    print(Q)
    print("\nR =")
    print(R)
    print("\nQ @ R =")
    print(A_reconstructed)
    print("\nReconstruction error =", norm(A - A_reconstructed))

    print("\nExplanation:")
    print("QR decomposition writes A as Q @ R. Q has orthonormal columns and R is")
    print("upper triangular. This is especially useful in least-squares problems.")


def problem3_cholesky_decomposition():
    section("Problem 3 - Cholesky Decomposition")

    A = np.array([
        [4, 2, 2],
        [2, 5, 1],
        [2, 1, 3]
    ], dtype=float)

    L = cholesky(A)
    A_reconstructed = L @ L.T

    print("A =")
    print(A)
    print("\nL =")
    print(L)
    print("\nL @ L.T =")
    print(A_reconstructed)
    print("\nReconstruction error =", norm(A - A_reconstructed))

    print("\nExplanation:")
    print("Cholesky decomposition writes A as L @ L.T. It requires A to be symmetric")
    print("and positive definite. Symmetry is required because L @ L.T is always symmetric.")


def problem4_compare_shapes():
    section("Problem 4 - Compare Shapes")

    A_lu = np.array([
        [2, 1, 1],
        [4, -6, 0],
        [-2, 7, 2]
    ], dtype=float)
    P, L_lu, U = lu(A_lu)

    A_qr = np.array([
        [1, 2],
        [3, 4],
        [5, 6]
    ], dtype=float)
    Q, R = qr(A_qr, mode='reduced')

    A_chol = np.array([
        [4, 2, 2],
        [2, 5, 1],
        [2, 1, 3]
    ], dtype=float)
    L_chol = cholesky(A_chol)

    print("LU input A_lu.shape =", A_lu.shape)
    print("P.shape =", P.shape, "L.shape =", L_lu.shape, "U.shape =", U.shape)

    print("\nQR input A_qr.shape =", A_qr.shape)
    print("Q.shape =", Q.shape, "R.shape =", R.shape)

    print("\nCholesky input A_chol.shape =", A_chol.shape)
    print("L.shape =", L_chol.shape)

    print("\nExplanation:")
    print("LU is normally used for square matrices and produces square factors.")
    print("QR can be used for rectangular matrices. With reduced mode, Q has the")
    print("same number of rows as A but only as many columns as A has features.")
    print("Cholesky is for square symmetric positive definite matrices and returns")
    print("one triangular factor L.")


def part_b_sensor_correlation_compression():
    section("Part B - Practical Problem: Sensor Correlation Compression")

    A = np.array([
        [12, 15, 18],
        [14, 17, 22],
        [16, 20, 25]
    ], dtype=float)

    print("Sensor matrix A =")
    print(A)
    print("Columns: Radar, EO/IR, Acoustic")

    rank_A = matrix_rank(A)
    det_A = det(A)

    print("\nStep 1: rank(A) =", rank_A)
    print("Step 1: det(A) =", det_A)

    print("\nInterpretation:")
    print("The matrix has full rank if rank(A)=3. A determinant far from zero also")
    print("suggests the columns are not exact linear combinations of each other.")

    Q, R = qr(A, mode='reduced')

    print("\nStep 2: Q =")
    print(Q)
    print("\nStep 2: R =")
    print(R)

    print("\nInterpretation of Q and R:")
    print("Q contains orthonormal directions extracted from the original sensor columns.")
    print("R tells how to combine those orthonormal directions to reconstruct A.")

    A_reconstructed = Q @ R
    reconstruction_error = norm(A - A_reconstructed)

    print("\nStep 3: A_reconstructed = Q @ R =")
    print(A_reconstructed)
    print("Reconstruction error =", reconstruction_error)

    print("\nStep 4: Scaling explanation:")
    print("For very large matrices, directly computing inverses can be slow and")
    print("numerically unstable. Factorizations such as QR, LU, Cholesky, and SVD")
    print("break matrices into easier pieces and are often the preferred way to")
    print("solve systems or least-squares problems.")

    print("\nStep 5: Research paragraph:")
    print("Matrix decompositions are fundamental because they transform difficult")
    print("matrix problems into simpler structured problems. Triangular, orthogonal,")
    print("or diagonal factors are easier and more stable to work with than the")
    print("original matrix. This is why decompositions are used in solving linear")
    print("systems, least-squares regression, matrix inverses, determinants, PCA,")
    print("SVD, and many machine-learning algorithms.")


def main():
    problem1_lu_decomposition()
    problem2_qr_decomposition()
    problem3_cholesky_decomposition()
    problem4_compare_shapes()
    part_b_sensor_correlation_compression()


if __name__ == "__main__":
    main()
