/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2008-2025 Stanford University and the Authors.      *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "openmm/Context.h"
#include "openmm/System.h"
#include "openmm/Platform.h"
#include "openmm/State.h"
#include "openmm/NonbondedForce.h"
#include "openmm/HarmonicBondForce.h"
#include "openmm/MonteCarloBarostat.h"
#include "openmm/LangevinMiddleIntegrator.h"
#include "openmm/VerletIntegrator.h"
#include "openmm/internal/AssertionUtilities.h"
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
using namespace OpenMM;
using namespace std;

class DerivedMiddle : public LangevinMiddleIntegrator {
public:
    DerivedMiddle() : LangevinMiddleIntegrator(300, 1, 0.001) {
    }
};

enum IntegratorKind {StandardMiddle, SubclassMiddle, Verlet};

vector<double> simulate(IntegratorKind kind, bool enabled, bool rigid, int groups, double pressure, int& changedBoxes) {
#ifdef _WIN32
    _putenv_s("OPENMM_EXPERIMENT_BAROSTAT_POTENTIAL_ONLY", enabled ? "1" : "0");
#else
    setenv("OPENMM_EXPERIMENT_BAROSTAT_POTENTIAL_ONLY", enabled ? "1" : "0", 1);
#endif
    System system;
    NonbondedForce* nonbonded = new NonbondedForce();
    nonbonded->setNonbondedMethod(NonbondedForce::CutoffPeriodic);
    nonbonded->setCutoffDistance(1.0);
    nonbonded->setForceGroup(1);
    HarmonicBondForce* bonds = new HarmonicBondForce();
    bonds->setForceGroup(2);
    vector<Vec3> positions, velocities;
    for (int i = 0; i < 12; i++) {
        system.addParticle(18.0);
        nonbonded->addParticle(0, 0.3, 0.05);
        positions.push_back(Vec3(0.7*(i%3), 0.7*((i/3)%2), 0.7*(i/6)));
        velocities.push_back(Vec3(0.001*i, -0.002*i, 0.003*i));
        if (i%3 == 1)
            bonds->addBond(i-1, i, 0.7, 10.0);
    }
    system.addForce(nonbonded);
    system.addForce(bonds);
    system.setDefaultPeriodicBoxVectors(Vec3(3.5, 0, 0), Vec3(0, 3.5, 0), Vec3(0, 0, 3.5));
    MonteCarloBarostat* barostat = new MonteCarloBarostat(pressure, 300, 1);
    barostat->setRandomNumberSeed(44221);
    barostat->setScaleMoleculesAsRigid(rigid);
    system.addForce(barostat);
    unique_ptr<Integrator> integrator;
    if (kind == StandardMiddle)
        integrator.reset(new LangevinMiddleIntegrator(300, 1, 0.001));
    else if (kind == SubclassMiddle)
        integrator.reset(new DerivedMiddle());
    else
        integrator.reset(new VerletIntegrator(0.001));
    if (kind != Verlet)
        dynamic_cast<LangevinMiddleIntegrator&>(*integrator).setRandomNumberSeed(99231);
    integrator->setIntegrationForceGroups(groups);
    Context context(system, *integrator, Platform::getPlatformByName("Reference"));
    context.setPositions(positions);
    context.setVelocities(velocities);
    vector<double> record;
    double previousBox = 3.5;
    for (int step = 0; step < 40; step++) {
        integrator->step(1);
        State state = context.getState(State::Positions|State::Velocities|State::Energy, false, groups);
        for (const Vec3& p : state.getPositions())
            for (int j = 0; j < 3; j++)
                record.push_back(p[j]);
        for (const Vec3& v : state.getVelocities())
            for (int j = 0; j < 3; j++)
                record.push_back(v[j]);
        Vec3 a, b, c;
        state.getPeriodicBoxVectors(a, b, c);
        for (const Vec3& v : {a, b, c})
            for (int j = 0; j < 3; j++)
                record.push_back(v[j]);
        record.push_back(state.getPotentialEnergy());
        record.push_back(state.getKineticEnergy());
        if (a[0] != previousBox)
            changedBoxes++;
        previousBox = a[0];
    }
    return record;
}

void testPotentialEnergyQuery() {
    int changedBoxes = 0;
    for (IntegratorKind kind : {StandardMiddle, SubclassMiddle, Verlet})
        for (bool rigid : {false, true})
            for (int groups : {2, 6})
                for (double pressure : {2.0, 10000.0}) {
                    vector<double> off = simulate(kind, false, rigid, groups, pressure, changedBoxes);
                    vector<double> on = simulate(kind, true, rigid, groups, pressure, changedBoxes);
                    ASSERT_EQUAL(off.size(), on.size());
                    ASSERT_EQUAL(0, memcmp(off.data(), on.data(), off.size()*sizeof(double)));
                }
    ASSERT(changedBoxes > 0);
}

int main() {
    try {
        testPotentialEnergyQuery();
    }
    catch (const exception& error) {
        cout << "exception: " << error.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
