// CUDA mixed-precision Middle kick and SETTLE velocity constraints.
#ifndef USE_MIXED_PRECISION
#error LangevinMiddle kick SETTLE fusion requires mixed precision
#endif

inline DEVICE mixed4 loadKickSettlePos(GLOBAL const real4* RESTRICT posq,
        GLOBAL const real4* RESTRICT posqCorrection, int index) {
    real4 pos1 = posq[index];
    real4 pos2 = posqCorrection[index];
    return make_mixed4(pos1.x+(mixed)pos2.x, pos1.y+(mixed)pos2.y, pos1.z+(mixed)pos2.z, pos1.w);
}

inline DEVICE mixed kickSettleStoredDouble(mixed value) {
    // Keep the f64 materialization boundary before velocity projection.
    mixed rounded;
    asm volatile("mov.b64 %0, %1;" : "=d"(rounded) : "d"(value));
    return rounded;
}

inline DEVICE mixed4 kickSettleVelocity(int index, int paddedNumAtoms, mixed fscale,
        GLOBAL const mixed4* RESTRICT velm, GLOBAL const mm_long* RESTRICT force) {
    mixed4 velocity = velm[index];
    if (velocity.w != 0.0) {
        velocity.x += fscale*velocity.w*force[index];
        velocity.y += fscale*velocity.w*force[index+paddedNumAtoms];
        velocity.z += fscale*velocity.w*force[index+paddedNumAtoms*2];
        velocity.x = kickSettleStoredDouble(velocity.x);
        velocity.y = kickSettleStoredDouble(velocity.y);
        velocity.z = kickSettleStoredDouble(velocity.z);
    }
    return velocity;
}

KERNEL void integrateLangevinMiddleKickSettle(int numClusters, int paddedNumAtoms,
        GLOBAL const real4* RESTRICT oldPos, GLOBAL const real4* RESTRICT posqCorrection,
        GLOBAL mixed4* RESTRICT velm, GLOBAL const mm_long* RESTRICT force,
        GLOBAL const mixed2* RESTRICT dt, GLOBAL const int4* RESTRICT clusterAtoms,
        int numResidualAtoms, GLOBAL const int* RESTRICT residualAtoms) {
    mixed fscale = dt[0].y/(mixed) 0x100000000;
    for (int index = GLOBAL_ID; index < numClusters; index += GLOBAL_SIZE) {
        int4 atoms = clusterAtoms[index];
        mixed4 v0 = kickSettleVelocity(atoms.x, paddedNumAtoms, fscale, velm, force);
        mixed4 v1 = kickSettleVelocity(atoms.y, paddedNumAtoms, fscale, velm, force);
        mixed4 v2 = kickSettleVelocity(atoms.z, paddedNumAtoms, fscale, velm, force);
        mixed4 apos0 = loadKickSettlePos(oldPos, posqCorrection, atoms.x);
        mixed4 apos1 = loadKickSettlePos(oldPos, posqCorrection, atoms.y);
        mixed4 apos2 = loadKickSettlePos(oldPos, posqCorrection, atoms.z);

        // Compute intermediate quantities: the atom masses, the bond directions, the relative velocities,
        // and the angle cosines and sines.

        computeSettleVelocities(apos0, apos1, apos2, &v0, &v1, &v2);
        velm[atoms.x] = v0;
        velm[atoms.y] = v1;
        velm[atoms.z] = v2;
    }
    // Apply the ordinary kick to the disjoint residual atoms.
    for (int slot = GLOBAL_ID; slot < numResidualAtoms; slot += GLOBAL_SIZE) {
        int index = residualAtoms[slot];
        mixed4 velocity = velm[index];
        if (velocity.w != 0.0) {
            velocity.x += fscale*velocity.w*force[index];
            velocity.y += fscale*velocity.w*force[index+paddedNumAtoms];
            velocity.z += fscale*velocity.w*force[index+paddedNumAtoms*2];
            velm[index] = velocity;
        }
    }
}
