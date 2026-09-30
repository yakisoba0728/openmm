// CUDA mixed-precision SETTLE position constraints and Middle finalization.
#ifndef USE_MIXED_PRECISION
#error LangevinMiddle SETTLE fusion requires mixed precision
#endif

inline DEVICE mixed4 loadPos(GLOBAL const real4* RESTRICT posq, GLOBAL const real4* RESTRICT posqCorrection, int index) {
    real4 pos1 = posq[index];
    real4 pos2 = posqCorrection[index];
    return make_mixed4(pos1.x+(mixed)pos2.x, pos1.y+(mixed)pos2.y, pos1.z+(mixed)pos2.z, pos1.w);
}

inline DEVICE mixed settleMiddleStoredDouble(mixed value) {
    // Keep the f64 materialization boundary before Part3 arithmetic.
    mixed rounded;
    asm volatile("mov.b64 %0, %1;" : "=d"(rounded) : "d"(value));
    return rounded;
}

inline DEVICE void finishSettleMiddleAtom(int index, mixed4 constrained,
#ifdef RELOAD_LANGEVIN_MIDDLE_ORIGINAL_DELTA
        GLOBAL const volatile mixed4* RESTRICT originalDelta,
#else
        mixed3 original,
#endif
        mixed invDt,
        GLOBAL real4* RESTRICT posq, GLOBAL mixed4* RESTRICT velm, GLOBAL real4* RESTRICT posqCorrection) {
    mixed4 delta = make_mixed4(settleMiddleStoredDouble(constrained.x),
            settleMiddleStoredDouble(constrained.y), settleMiddleStoredDouble(constrained.z), 0);
#ifdef RELOAD_LANGEVIN_MIDDLE_ORIGINAL_DELTA
    // SETTLE atoms retain their original deltas; volatile requests a reload.
    mixed3 original;
    original.x = originalDelta[index].x;
    original.y = originalDelta[index].y;
    original.z = originalDelta[index].z;
#endif
    mixed4 velocity = velm[index];
    if (velocity.w != 0.0) {
        velocity.x += (delta.x-original.x)*invDt;
        velocity.y += (delta.y-original.y)*invDt;
        velocity.z += (delta.z-original.z)*invDt;
        velm[index] = velocity;
        // Retain Part3's mixed-position reconstruction and charge component.
        real4 pos1 = posq[index];
        real4 pos2 = posqCorrection[index];
        mixed4 pos = make_mixed4(pos1.x+(mixed)pos2.x, pos1.y+(mixed)pos2.y, pos1.z+(mixed)pos2.z, pos1.w);
        pos.x += delta.x;
        pos.y += delta.y;
        pos.z += delta.z;
        posq[index] = make_real4((real) pos.x, (real) pos.y, (real) pos.z, (real) pos.w);
        posqCorrection[index] = make_real4(pos.x-(real) pos.x, pos.y-(real) pos.y, pos.z-(real) pos.z, 0);
    }
}

KERNEL void applySettleAndLangevinMiddlePart3(int numClusters, mixed tol, GLOBAL real4* RESTRICT oldPos,
        GLOBAL mixed4* RESTRICT posDelta, GLOBAL mixed4* RESTRICT velm, GLOBAL const int4* RESTRICT clusterAtoms,
        GLOBAL const float2* RESTRICT clusterParams
#ifdef USE_MIXED_PRECISION
        , GLOBAL real4* RESTRICT posqCorrection
#endif
        , GLOBAL const mixed2* RESTRICT dt
#ifdef FUSE_LANGEVIN_MIDDLE_RESIDUAL_TAIL
        , int numResidualAtoms, GLOBAL const int* RESTRICT residualAtoms, GLOBAL const mixed4* RESTRICT oldDelta
#endif
        ) {
    mixed invDt = 1/dt[0].y;
#ifndef USE_MIXED_PRECISION
        GLOBAL real4* posqCorrection = 0;
#endif
    int index = GLOBAL_ID;
    while (index < numClusters) {
        // Load the data for this cluster.

        int4 atoms = clusterAtoms[index];
        float2 params = clusterParams[index];
        mixed4 apos0 = loadPos(oldPos, posqCorrection, atoms.x);
        mixed4 xp0 = posDelta[atoms.x];
        mixed4 apos1 = loadPos(oldPos, posqCorrection, atoms.y);
        mixed4 xp1 = posDelta[atoms.y];
        mixed4 apos2 = loadPos(oldPos, posqCorrection, atoms.z);
        mixed4 xp2 = posDelta[atoms.z];
#ifndef RELOAD_LANGEVIN_MIDDLE_ORIGINAL_DELTA
        mixed3 original0 = make_mixed3(xp0.x, xp0.y, xp0.z);
        mixed3 original1 = make_mixed3(xp1.x, xp1.y, xp1.z);
        mixed3 original2 = make_mixed3(xp2.x, xp2.y, xp2.z);
#endif
        mixed m0 = 1/velm[atoms.x].w;
        mixed m1 = 1/velm[atoms.y].w;
        mixed m2 = 1/velm[atoms.z].w;

        // Apply the SETTLE algorithm.

        computeSettlePositions(apos0, apos1, apos2, params, m0, m1, m2, &xp0, &xp1, &xp2);

        // Record the new positions.

        // Materialize constrained deltas before applying Part3.
#ifdef RELOAD_LANGEVIN_MIDDLE_ORIGINAL_DELTA
        finishSettleMiddleAtom(atoms.x, xp0, posDelta, invDt, oldPos, velm, posqCorrection);
        finishSettleMiddleAtom(atoms.y, xp1, posDelta, invDt, oldPos, velm, posqCorrection);
        finishSettleMiddleAtom(atoms.z, xp2, posDelta, invDt, oldPos, velm, posqCorrection);
#else
        finishSettleMiddleAtom(atoms.x, xp0, original0, invDt, oldPos, velm, posqCorrection);
        finishSettleMiddleAtom(atoms.y, xp1, original1, invDt, oldPos, velm, posqCorrection);
        finishSettleMiddleAtom(atoms.z, xp2, original2, invDt, oldPos, velm, posqCorrection);
#endif
        index += GLOBAL_SIZE;
    }
#ifdef FUSE_LANGEVIN_MIDDLE_RESIDUAL_TAIL
    // SHAKE/CCMA completed on the same stream before this launch. The list is
    // disjoint from SETTLE, so no cross-workgroup barrier is needed here.
    for (int residual = GLOBAL_ID; residual < numResidualAtoms; residual += GLOBAL_SIZE) {
        int atom = residualAtoms[residual];
        mixed4 velocity = velm[atom];
        if (velocity.w != 0.0) {
            mixed4 delta = posDelta[atom];
            velocity.x += (delta.x-oldDelta[atom].x)*invDt;
            velocity.y += (delta.y-oldDelta[atom].y)*invDt;
            velocity.z += (delta.z-oldDelta[atom].z)*invDt;
            velm[atom] = velocity;
#ifdef USE_MIXED_PRECISION
            real4 pos1 = oldPos[atom];
            real4 pos2 = posqCorrection[atom];
            mixed4 pos = make_mixed4(pos1.x+(mixed)pos2.x, pos1.y+(mixed)pos2.y, pos1.z+(mixed)pos2.z, pos1.w);
#else
            real4 pos = oldPos[atom];
#endif
            pos.x += delta.x;
            pos.y += delta.y;
            pos.z += delta.z;
#ifdef USE_MIXED_PRECISION
            oldPos[atom] = make_real4((real) pos.x, (real) pos.y, (real) pos.z, (real) pos.w);
            posqCorrection[atom] = make_real4(pos.x-(real) pos.x, pos.y-(real) pos.y, pos.z-(real) pos.z, 0);
#else
            oldPos[atom] = pos;
#endif
        }
    }
#endif
}
