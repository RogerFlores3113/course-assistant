# MATH 2210 Problem Set 2: Elimination and LU Factorization

**Due Thursday September 17, 2026 at 11:59 pm.** Total: 60 points. Write up your own solution and list your collaborators. Problems 1 and 3 are adapted from MIT 18.06 Fall 2018 pset 1 (S. G. Johnson) and Strang, section 2.2.

## Problem 1 (15 points)

Consider Gaussian elimination on the following system of equations.

```
2x + 5y +  z = 0
4x + dy +  z = 2
     y  -  z = 3
```

(a) What number d forces you to do a row exchange during elimination, and what non-singular triangular system do you obtain for that d?

(b) What value of d would make this system singular (no third pivot)?

## Problem 2 (15 points)

Let A be the 3 by 3 matrix with rows (1, 6, -3), (-2, 3, 4), (1, 0, -2), and let b1 = (7, 3, 0) and b2 = (0, -2, 1) be column vectors.

```
A = [  1  6 -3 ]     b1 = [ 7 ]     b2 = [  0 ]
    [ -2  3  4 ]          [ 3 ]          [ -2 ]
    [  1  0 -2 ]          [ 0 ]          [  1 ]
```

Solve both A x1 = b1 and A x2 = b2 at once by forming the augmented matrix [A | b1 b2] and eliminating once. Give x1 and x2 explicitly.

## Problem 3 (20 points)

Let A be the 3 by 3 matrix with rows (2, 1, 1), (4, -6, 0), (-2, 7, 2), and let b = (5, -2, 9) as a column vector.

```
A = [  2  1  1 ]     b = [  5 ]
    [  4 -6  0 ]         [ -2 ]
    [ -2  7  2 ]         [  9 ]
```

(a) Find the factorization A = LU, with L lower triangular with ones on the diagonal and U upper triangular.

(b) Use forward substitution on Ly = b and back substitution on Ux = y to find x such that Ax = b.

## Problem 4 (10 points)

A system of linear equations Ax = b cannot have exactly two solutions. If x and y (with y different from x) are two solutions, what is another solution? Explain why this shows that the number of solutions is never exactly two.
