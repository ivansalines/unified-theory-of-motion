# Preserved Finding

## 1. Direct matrix geometry

The actual reduced rotation-time tangent matrices

    B(s) in R^(9 x 9)

were sampled and flattened into 81-dimensional vectors. Direct PCA showed
that their moving geometry is effectively two-dimensional.

For Q=522.25:

- PC1 + PC2 cumulative variance: 0.9999999922777

For Q=550:

- PC1 + PC2 cumulative variance: 0.9999986108182

## 2. Alternating two-stroke affine involution

Using the centered moving matrix

    R(s) = B(s) - C,

the measured relations are

    R(s + P_line)  ~= -R(s)
    R(s + 2P_line) ~=  R(s)

while the dominant tangent direction satisfies

    w(s + P_line)  ~= -w(s)
    w(s + 2P_line) ~=  w(s).

The best active-plane half-step map has determinant +1, trace close to -2,
and rotation angle close to 180 degrees for both tested Q values.

## 3. Continuous rocking generator

The two-stroke trajectory is overwhelmingly first-harmonic and is described
by a nearly constant planar generator:

    dR/ds ~= G R.

After normalization by the measured angular rate,

    J = G / omega,

the audit finds

    J^2 ~= -I.

Consequently,

    exp(P_line G)   ~= -I
    exp(2 P_line G) ~=  I.

The measured angular rate agrees with pi / P_line to about 2.4e-5 to 2.6e-5
relative error.

## 4. Dominant mechanical anatomy

The strongest fundamental matrix couplings are the reciprocal pair

    d(delta v2)/ds     <- delta omega2
    d(delta omega2)/ds <- delta v2.

The next tier couples the rocking core to Delta, y1, y2, and z.

Between Q=522.25 and Q=550, the normalized hierarchy remains nearly the same,
while the strongest absolute fundamental amplitude increases by about 11.34
times.

## 5. Current interpretation

The system does not merely alternate between two static states. Its reduced
tangent law executes a continuous, almost harmonic rotation in an active
two-dimensional affine plane.

A concise interpretation is:

> Rocking is not a loss of equilibrium; it is the equilibrium mechanism in
> motion.
