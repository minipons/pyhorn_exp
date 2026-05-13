FeatureScript 2945;
import(path : "onshape/std/common.fs", version : "2945.0");
import(path : "c5f898a314df4e00b700db23", version : "36d3b6bfb2dcdf18c0b6b4a9");

// ── pyhorn_core ────────────────────────────────────────────────────────────────
// import(path : "<pyhorn-core-doc-id>", version : "<version-hash>");
//
// Helpers inlined from pyhorn_core (must match pyhorn_core.fs):
//   _vectorMagnitude, _point3dDist, _pathLengthFromSamples,
//   _hyperbolicAreaAtFraction, _solveFlareArgumentFromAreasAndT,
//   _arcLengthStation, _deriveSketchPlaneFromSamples,
//   _buildTraces, _orderEdgeChain

const TOL_METERS = 1e-6 * meter;

// ── Inlined pyhorn_core helpers ────────────────────────────────────────────────

export function _vectorMagnitude(v is Vector) returns ValueWithUnits
{
    return sqrt(v[0] * v[0] + v[1] * v[1]);
}

export function _point3dDist(a is Vector, b is Vector) returns ValueWithUnits
{
    var dx = a[0] - b[0];
    var dy = a[1] - b[1];
    var dz = a[2] - b[2];
    return sqrt(dx * dx + dy * dy + dz * dz);
}

export function _hyperbolicAreaAtFraction(
    throatArea is ValueWithUnits,
    uL        is number,
    t         is number,
    fraction  is number
) returns ValueWithUnits
{
    var ratio = cosh(uL * fraction) + t * sinh(uL * fraction);
    return throatArea * ratio * ratio;
}

export function _solveFlareArgumentFromAreasAndT(
    throatArea is ValueWithUnits,
    mouthArea  is ValueWithUnits,
    t          is number
) returns number
{
    var areaRatioSqrt = sqrt(mouthArea / throatArea);

    if (abs(t + 1.0) <= 1e-12)
    {
        if (areaRatioSqrt >= 1.0)
        {
            throw regenError("T = -1 cannot produce an expanding horn (tractrix needs S2 < S1).");
        }
        return -log(areaRatioSqrt);
    }

    var discriminant = areaRatioSqrt * areaRatioSqrt + t * t - 1.0;
    if (discriminant < 0.0)
    {
        throw regenError("S1, S2, and T do not yield a real hyperbolic horn solution.");
    }

    var root = sqrt(discriminant);
    var denominator = 1.0 + t;
    var candidateA = (areaRatioSqrt + root) / denominator;
    var candidateB = (areaRatioSqrt - root) / denominator;

    if (candidateA > 1.0) { return log(candidateA); }
    if (candidateB > 1.0) { return log(candidateB); }

    throw regenError("S1, S2, and T imply a non-expanding hyperbolic horn.");
}

// ── Trace builder (from pyhorn_core) ─────────────────────────────────────────

function _findNext(lastPt is array, prevPt is array, edgeData is array, nEdges is number, TOL is ValueWithUnits) returns map
{
    var bestAngle = PI * radian;
    var nextIdx   = -1;
    var forward   = true;

    for (var j = 0; j < nEdges; j += 1)
    {
        if (edgeData[j]["used"]) { continue; }
        var q0 = edgeData[j]["p0"];
        var q1 = edgeData[j]["p1"];

        var d0x = lastPt[0] - q0[0]; var d0y = lastPt[1] - q0[1]; var d0z = lastPt[2] - q0[2];
        var d1x = lastPt[0] - q1[0]; var d1y = lastPt[1] - q1[1]; var d1z = lastPt[2] - q1[2];
        var dist0 = sqrt(d0x*d0x + d0y*d0y + d0z*d0z);
        var dist1 = sqrt(d1x*d1x + d1y*d1y + d1z*d1z);
        if (dist0 >= TOL && dist1 >= TOL) { continue; }

        var cX = (dist0 < TOL) ? (q1[0] - q0[0]) : (q0[0] - q1[0]);
        var cY = (dist0 < TOL) ? (q1[1] - q0[1]) : (q0[1] - q1[1]);
        var cZ = (dist0 < TOL) ? (q1[2] - q0[2]) : (q0[2] - q1[2]);
        var lC = sqrt(cX*cX + cY*cY + cZ*cZ);
        if (lC < 1e-9 * meter) { continue; }

        var pX = lastPt[0] - prevPt[0]; var pY = lastPt[1] - prevPt[1]; var pZ = lastPt[2] - prevPt[2];
        var lP = sqrt(pX*pX + pY*pY + pZ*pZ);
        if (lP < 1e-9 * meter) { continue; }

        var cosA = clamp((pX*cX + pY*cY + pZ*cZ) / (lP * lC), -1.0, 1.0);
        var angle = acos(cosA);
        if (angle < bestAngle) { bestAngle = angle; nextIdx = j; forward = dist0 < TOL; }
    }
    return { "nextIdx": nextIdx, "forward": forward };
}

export function _buildTraces(context is Context, rawEdges is array, samplesPerEdge is number) returns array
{
    var edgeData = [];
    for (var i = 0; i < size(rawEdges); i += 1)
    {
        var ep = evEdgeTangentLines(context, { "edge" : rawEdges[i], "parameters" : [0.0, 1.0] });
        edgeData = append(edgeData, { "query": rawEdges[i], "p0": ep[0].origin, "p1": ep[1].origin, "used": false });
    }

    var traces = [];
    var nEdges = size(rawEdges);

    // Forward walk
    for (var start = 0; start < nEdges; start += 1)
    {
        if (edgeData[start]["used"]) { continue; }
        var trace = []; var cur = start;
        var prevPt = [0 * meter, 0 * meter, 0 * meter];

        while (cur >= 0 && !edgeData[cur]["used"])
        {
            edgeData[cur]["used"] = true;
            var p0 = edgeData[cur]["p0"];
            var p1 = edgeData[cur]["p1"];
            var dir = edgeData[cur]["forward"];  // not stored; default forward
            trace = append(trace, p0);
            if (cur == start) { trace = append(trace, p1); }
            var lastPt = trace[size(trace) - 1];
            var cont = _findNext(lastPt, prevPt, edgeData, nEdges, TOL_METERS);
            if (cont["nextIdx"] < 0) { break; }
            cur = cont["nextIdx"]; prevPt = lastPt;
        }
        if (size(trace) > 0) { traces = append(traces, trace); }
    }

    return traces;
}

// ── Sketch-plane from centreline samples ───────────────────────────────────────

export function _deriveSketchPlaneFromSamples(sampledLines is array) returns Plane
{
    if (size(sampledLines) < 2)
    {
        throw regenError("Cannot derive sketch plane from fewer than 2 centreline samples.");
    }
    var pA = sampledLines[0].origin;
    var pB = sampledLines[floor(size(sampledLines) / 2)].origin;
    var pC = sampledLines[size(sampledLines) - 1].origin;
    var v1 = pB - pA;
    var v2 = pC - pA;
    var normal = cross(v1, v2);
    var nMag = sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);

    if (nMag < 1e-15 * meter * meter)
    {
        throw regenError("Cannot determine centreline plane — path appears degenerate.");
    }
    var normalUnit = normal / nMag;
    var tHat = v1 / sqrt(v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2]);
    var dotTN = tHat[0]*normalUnit[0] + tHat[1]*normalUnit[1] + tHat[2]*normalUnit[2];
    var xDirX = tHat[0] - dotTN * normalUnit[0];
    var xDirY = tHat[1] - dotTN * normalUnit[1];
    var xDirZ = tHat[2] - dotTN * normalUnit[2];
    var xMag = sqrt(xDirX*xDirX + xDirY*xDirY + xDirZ*xDirZ);
    if (xMag < 1e-10)
    {
        var seed = (abs(normalUnit[0]) < 0.9) ? vector(1,0,0) : vector(0,1,0);
        var cx = normalUnit[1]*seed[2]-normalUnit[2]*seed[1];
        var cy = normalUnit[2]*seed[0]-normalUnit[0]*seed[2];
        var cz = normalUnit[0]*seed[1]-normalUnit[1]*seed[0];
        var cMag = sqrt(cx*cx + cy*cy + cz*cz);
        xDirX = cx/cMag; xDirY = cy/cMag; xDirZ = cz/cMag;
    }
    else
    {
        xDirX /= xMag; xDirY /= xMag; xDirZ /= xMag;
    }
    return plane(coordSystem(pA, vector(xDirX,xDirY,xDirZ), normalUnit));
}

// ── Centreline arc-length sampling ─────────────────────────────────────────────

export function _arcLengthStation(
    sampledLines is array, curveLength is ValueWithUnits, fraction is number
) returns map
{
    var targetDistMM = (curveLength / millimeter) * fraction;
    var cumDistMM = 0.0;

    for (var s = 0; s < size(sampledLines) - 1; s += 1)
    {
        var segDxMM = (sampledLines[s + 1].origin[0] - sampledLines[s].origin[0]) / millimeter;
        var segDyMM = (sampledLines[s + 1].origin[1] - sampledLines[s].origin[1]) / millimeter;
        var segDzMM = (sampledLines[s + 1].origin[2] - sampledLines[s].origin[2]) / millimeter;
        var segLenMM = sqrt(segDxMM*segDxMM + segDyMM*segDyMM + segDzMM*segDzMM);

        if (cumDistMM + segLenMM >= targetDistMM || s == size(sampledLines) - 2)
        {
            var t = (s == size(sampledLines) - 2) ? 1.0
                : (targetDistMM - cumDistMM) / segLenMM;
            var samplePos = vector(
                sampledLines[s].origin[0] + t * (sampledLines[s + 1].origin[0] - sampledLines[s].origin[0]),
                sampledLines[s].origin[1] + t * (sampledLines[s + 1].origin[1] - sampledLines[s].origin[1]),
                sampledLines[s].origin[2] + t * (sampledLines[s + 1].origin[2] - sampledLines[s].origin[2])
            );
            return { "pos": samplePos, "dir": sampledLines[s].direction };
        }
        cumDistMM += segLenMM;
    }
    // Fallback
    return { "pos": sampledLines[size(sampledLines)-1].origin, "dir": sampledLines[size(sampledLines)-1].direction };
}

// ── Path length from samples ────────────────────────────────────────────────────

export function _pathLengthFromSamples(sampledLines is array) returns ValueWithUnits
{
    var totalMM = 0.0;
    for (var k = 0; k < size(sampledLines) - 1; k += 1)
    {
        var dx = (sampledLines[k + 1].origin[0] - sampledLines[k].origin[0]) / millimeter;
        var dy = (sampledLines[k + 1].origin[1] - sampledLines[k].origin[1]) / millimeter;
        var dz = (sampledLines[k + 1].origin[2] - sampledLines[k].origin[2]) / millimeter;
        totalMM += sqrt(dx * dx + dy * dy + dz * dz);
    }
    return totalMM * millimeter;
}

// ═══════════════════════════════════════════════════════════════════════════════════
// pyhorn_horn_sections — multi-section folded horn
//
// Key idea: split the horn into independent swept sections at each bend.
// Each section is swept from its local throat to its local mouth,
// then all sections are boolean-unioned. This avoids the wall-crossing
// failure that occurs when a single centreline sweep must traverse a sharp bend.
// ═══════════════════════════════════════════════════════════════════════════════════

// ── Per-section sweep helpers ─────────────────────────────────────────────────

/**
 * Sweep one horn section (one leg between two bends) as a lofted body.
 *
 * Build profile sketches at throat (frac=0) and mouth (frac=1) of the section,
 * loft between them, then extrude the result perpendicular to the sketch plane
 * by `width` to form the rectangular cross-section.
 *
 * The section centreline is provided as a pre-sampled array of tangent lines.
 * Profile is built using the consistent unitSide approach (locked at throat).
 */
function _sweepSection(
    context           is Context,
    sectionId         is Id,
    sectionLines      is array,         // centreline tangent lines for this section only
    sectionLength     is ValueWithUnits, // pre-computed path length of this section
    throatArea        is ValueWithUnits, // throat cross-section area (m²)
    mouthArea         is ValueWithUnits, // mouth cross-section area (m²)
    width             is ValueWithUnits, // horn width perpendicular to sweep plane
    uL                is number,        // hyperbolic flare constant for THIS section
    t                 is number,         // T parameter
    sketchPlane       is Plane,
    numStations       is number          // profile sketches at each station
) returns Query
{
    var leftPts = [];
    var rightPts = [];

    // ── Compute profile points at each station ────────────────────────────────
    // unitSide is locked at the FIRST station (throat) — holds through entire sweep
    var unitSide = undefined;

    for (var idx = 0; idx <= numStations; idx += 1)
    {
        var fraction = idx / numStations;

        var station = _arcLengthStation(sectionLines, sectionLength, fraction);
        var samplePos = station["pos"];
        var sampleDir = station["dir"];

        var areaAtStation = _hyperbolicAreaAtFraction(throatArea, uL, t, fraction);
        var halfHeight = areaAtStation / (2.0 * width);

        var origin = worldToPlane(sketchPlane, samplePos);
        var tangentPoint = worldToPlane(sketchPlane, samplePos + sampleDir * millimeter);
        var tangent = tangentPoint - origin;
        var tangentMag = _vectorMagnitude(tangent);

        var unitPerp;
        if (tangentMag > 1e-9 * meter)
        {
            unitPerp = vector(-tangent[1], tangent[0]) / tangentMag;
            if (unitSide == undefined)
            {
                unitSide = unitPerp;  // lock at throat — consistent for entire sweep
            }
        }
        else
        {
            unitPerp = unitSide;  // at degenerate/corner stations, hold last valid direction
        }

        leftPts  = append(leftPts,  origin + unitSide * halfHeight);
        rightPts = append(rightPts, origin - unitSide * halfHeight);
    }

    if (size(leftPts) < 2)
    {
        throw regenError("Section has too few profile stations — check centreline sampling.");
    }

    // ── Build lofted body between profiles ───────────────────────────────────
    // Create profile sketches at first and last station, loft between them
    var throatSketchId = sectionId + "throatProfile";
    var throatSketch = newSketchOnPlane(context, throatSketchId, { "sketchPlane" : sketchPlane });
    skLineSegment(throatSketch, "tw", { "start" : leftPts[0], "end" : rightPts[0] });
    skSolve(throatSketch);

    var mouthSketchId = sectionId + "mouthProfile";
    var mouthSketch = newSketchOnPlane(context, mouthSketchId, { "sketchPlane" : sketchPlane });
    skLineSegment(mouthSketch, "mw", { "start" : leftPts[size(leftPts)-1], "end" : rightPts[size(rightPts)-1] });
    skSolve(mouthSketch);

    // Loft between throat and mouth profiles
    var loft = loftProfiles(context, throatSketchId, [mouthSketchId]);

    return loft;
}

/**
 * Find the index into `centrelineSamples` that is closest to a given 3D point.
 */
function _closestSampleIdx(targetPt is array, centrelineSamples is array) returns number
{
    var bestIdx = 0;
    var bestDist = 1e30 * meter;
    for (var i = 0; i < size(centrelineSamples); i += 1)
    {
        var d = _point3dDist(targetPt, centrelineSamples[i].origin);
        if (d < bestDist) { bestDist = d; bestIdx = i; }
    }
    return bestIdx;
}

// ── Feature ───────────────────────────────────────────────────────────────────

annotation { "Feature Type Name" : "Horn Profile (Sections)",
             "Feature Type Description" : "Multi-section folded horn — each leg swept independently, boolean-unioned at bends" }
export const pyhornHornSections = defineFeature(function(context is Context, id is Id, definition is map)
    precondition
    {
        annotation { "Name" : "Centreline", "Filter" : EntityType.EDGE || (EntityType.BODY && BodyType.WIRE), "MaxNumberOfPicks" : 1000 }
        definition.edge is Query;

        annotation { "Name" : "Internal Width (mm)", "UIHint" : UIHint.REMEMBER_PREVIOUS_VALUE, "Default" : 100.0, "Min" : 1.0, "Max" : 2000.0 }
        isLength(definition.width, LENGTH_BOUNDS);

        annotation { "Name" : "S1 Throat (cm^2)", "Default" : 40.0, "Min" : 0.01, "Max" : 10000.0 }
        isReal(definition.s1Cm2, { (unitless) : [0.01, 40.0, 10000.0] } as RealBoundSpec);

        annotation { "Name" : "S2 Mouth (cm^2)", "Default" : 600.0, "Min" : 0.01, "Max" : 10000.0 }
        isReal(definition.s2Cm2, { (unitless) : [0.01, 600.0, 10000.0] } as RealBoundSpec);

        annotation { "Name" : "T", "Default" : 0.7, "Min" : -0.99, "Max" : 10.0 }
        isReal(definition.t, { (unitless) : [-0.99, 0.7, 10.0] } as RealBoundSpec);

        annotation { "Name" : "Samples", "Default" : 300, "Min" : 8, "Max" : 2000 }
        isInteger(definition.samples, { (unitless) : [8, 300, 2000] } as IntegerBoundSpec);

        annotation { "Name" : "Sections profile stations", "Default" : 6, "Min" : 2, "Max" : 50 }
        isInteger(definition.sectionStations, { (unitless) : [2, 6, 50] } as IntegerBoundSpec);

        annotation { "Name" : "Output JSON", "Default" : true }
        definition.outputJson is boolean;
    }
    {
        // ── Resolve edges ──────────────────────────────────────────────────
        var edges = evaluateQuery(context, qEntityFilter(definition.edge, EntityType.EDGE));
        if (size(edges) == 0)
        {
            edges = evaluateQuery(context, qOwnedByBody(definition.edge, EntityType.EDGE));
        }
        if (size(edges) == 0)
        {
            throw regenError("No edges found. Select edges or a composite curve.");
        }

        // ── Sample full centreline ────────────────────────────────────────
        var samplesPerEdge = max(2, round(definition.samples / size(edges)));
        var allCentrelinePts = [];
        var allTangentLines = [];

        for (var e = 0; e < size(edges); e += 1)
        {
            var params = [];
            for (var i = 0; i <= samplesPerEdge; i += 1) { params = append(params, i / samplesPerEdge); }
            var tangentLines = evEdgeTangentLines(context, { "edge" : edges[e], "parameters" : params });
            var startIdx = (e == 0) ? 0 : 1;
            for (var s = startIdx; s < size(tangentLines); s += 1)
            {
                allTangentLines = append(allTangentLines, tangentLines[s]);
            }
        }

        if (size(allTangentLines) < 2)
        {
            throw regenError("Selected centreline is too short or degenerate.");
        }

        var curveLength = _pathLengthFromSamples(allTangentLines);

        // ── Sketch plane from full centreline ───────────────────────────────
        var sketchPlane = _deriveSketchPlaneFromSamples(allTangentLines);

        // ── Extract wall traces from 2D air-volume face ──────────────────
        // User must also select the 2D air-volume face to get the wall geometry
        // Fallback: use a simple perpendicular-offset of the centreline
        // (This requires the user to have created the wall geometry separately)
        //
        // For now: build sections directly from the centreline + width + area profile.
        // The section decomposition is driven by the global horn T parameter
        // and the user-specified section count.  Bends are detected from the
        // centreline tangent changes; each significant direction change (> 20°)
        // is treated as a section boundary.
        //
        // Full wall-trace extraction from a selected face would require an additional
        // face selection — deferred to a future version.

        var throatArea = definition.s1Cm2 * centimeter * centimeter;
        var mouthArea  = definition.s2Cm2  * centimeter * centimeter;
        var widthM     = definition.width;

        // ── Detect section boundaries from bend angles ──────────────────────
        var bendAngles = [];
        var nSamples = size(allTangentLines);
        for (var i = 1; i < nSamples - 1; i += 1)
        {
            var tCurr = allTangentLines[i].direction;
            var tNext = allTangentLines[i + 1].direction;
            var cosDelta = clamp(tCurr[0]*tNext[0] + tCurr[1]*tNext[1] + tCurr[2]*tNext[2], -1.0, 1.0);
            var angleDeg = acos(cosDelta) * 180.0 / PI;
            bendAngles = append(bendAngles, angleDeg);
        }

        // Section boundaries: indices where bend angle exceeds threshold (e.g. 20°)
        var BEND_THRESHOLD_DEG = 20.0;
        var sectionBoundaries = [0];  // always start at 0
        for (var i = 0; i < size(bendAngles); i += 1)
        {
            if (bendAngles[i] > BEND_THRESHOLD_DEG)
            {
                sectionBoundaries = append(sectionBoundaries, i + 1);
            }
        }
        sectionBoundaries = append(sectionBoundaries, nSamples - 1);  // always end at last

        var numSections = size(sectionBoundaries) - 1;
        if (numSections < 1)
        {
            numSections = 1;  // fallback: single section
        }

        println("Detected " ~ numSections ~ " section(s) from bend analysis.");

        // ── Compute global horn parameters ─────────────────────────────────
        var globalUL = _solveFlareArgumentFromAreasAndT(throatArea, mouthArea, definition.t);
        var globalF12 = (globalUL / (curveLength / meter)) * 343.0 / (2.0 * PI);

        // ── Sweep each section independently ───────────────────────────────
        var sectionBodies = [];

        for (var s = 0; s < numSections; s += 1)
        {
            var startIdx = sectionBoundaries[s];
            var endIdx   = sectionBoundaries[s + 1];

            // Extract sub-centreline for this section
            var sectionLines = [];
            for (var k = startIdx; k <= endIdx; k += 1)
            {
                sectionLines = append(sectionLines, allTangentLines[k]);
            }

            var sectionLength = _pathLengthFromSamples(sectionLines);

            // Each section spans from its own start fraction to its own end fraction
            // of the global hyperbolic profile.  The global profile is evaluated at those
            // fractions to get the section's local throat and mouth areas.
            //
            // sectionBoundaries[s] and sectionBoundaries[s+1] are indices into the
            // centreline sample array.  Converting to a path fraction:
            //   frac = boundary_index / (nSamples - 1)
            // (nSamples - 1 is the total number of intervals in the path)
            var pathFracStart = sectionBoundaries[s]         / (nSamples - 1.0);
            var pathFracEnd   = sectionBoundaries[s + 1]   / (nSamples - 1.0);

            var sectionThroatArea = throatArea;
            var sectionMouthArea  = mouthArea;

            if (s == 0)
            {
                // First section: throat = global throat, mouth = global profile at end frac
                sectionThroatArea = throatArea;
                sectionMouthArea  = _hyperbolicAreaAtFraction(throatArea, globalUL, definition.t, pathFracEnd);
            }
            else if (s == numSections - 1)
            {
                // Last section: throat = global profile at start frac, mouth = global mouth
                sectionThroatArea = _hyperbolicAreaAtFraction(throatArea, globalUL, definition.t, pathFracStart);
                sectionMouthArea  = mouthArea;
            }
            else
            {
                // Interior section: both throat and mouth from global profile
                sectionThroatArea = _hyperbolicAreaAtFraction(throatArea, globalUL, definition.t, pathFracStart);
                sectionMouthArea  = _hyperbolicAreaAtFraction(throatArea, globalUL, definition.t, pathFracEnd);
            }

            // Solve local flare constant for this section
            var sectionUL = _solveFlareArgumentFromAreasAndT(sectionThroatArea, sectionMouthArea, definition.t);

            // Sweep this section
            var sectionId = id + "section" ~ s;
            try
            {
                var sectionBody = _sweepSection(
                    context, sectionId,
                    sectionLines, sectionLength,
                    sectionThroatArea, sectionMouthArea,
                    widthM, sectionUL, definition.t,
                    sketchPlane,
                    definition.sectionStations
                );
                sectionBodies = append(sectionBodies, sectionBody);
            }
            catch
            {
                // If a section fails to sweep (e.g. degenerate), skip it gracefully
                println("Warning: section " ~ s ~ " failed to generate — skipping.");
            }
        }

        if (size(sectionBodies) == 0)
        {
            throw regenError("No horn sections could be generated. Check centreline selection.");
        }

        // ── Boolean union all sections ────────────────────────────────────
        if (size(sectionBodies) == 1)
        {
            // Single section — no boolean needed
            println("Single-section horn generated. F12 = " ~ round(globalF12 * 100.0) / 100.0 ~ " Hz");
        }
        else
        {
            // Multiple sections — boolean union
            var result = sectionBodies[0];
            for (var b = 1; b < size(sectionBodies); b += 1)
            {
                result = booleanUnion(context, result, sectionBodies[b]);
            }
            println(size(sectionBodies) ~ "-section horn boolean-unioned. F12 = " ~ round(globalF12 * 100.0) / 100.0 ~ " Hz");
        }

        // ── Build JSON summary ────────────────────────────────────────────
        if (definition.outputJson)
        {
            var sectionData = [];
            for (var s = 0; s < numSections; s += 1)
            {
                var startIdx = sectionBoundaries[s];
                var endIdx   = sectionBoundaries[s + 1];
                var sectionThroatArea = throatArea;
                var sectionMouthArea  = mouthArea;
                if (s == 0) { sectionThroatArea = throatArea; sectionMouthArea = mouthArea; }
                else if (s == numSections - 1) { sectionThroatArea = throatArea; sectionMouthArea = mouthArea; }
                else { sectionThroatArea = throatArea; sectionMouthArea = _hyperbolicAreaAtFraction(throatArea, globalUL, definition.t, 1.0); }

                var sectionUL = _solveFlareArgumentFromAreasAndT(sectionThroatArea, sectionMouthArea, definition.t);
                sectionData = append(sectionData, {
                    "section": s,
                    "start_idx": startIdx,
                    "end_idx": endIdx,
                    "throat_area_cm2": (sectionThroatArea / (centimeter * centimeter)),
                    "mouth_area_cm2":  (sectionMouthArea  / (centimeter * centimeter)),
                    "uL": sectionUL
                });
            }

            var json = "{\n";
            json ~= " \"enclosure_type\": \"BLH\",\n";
            json ~= " \"n_sections\": " ~ numSections ~ ",\n";
            json ~= " \"width\": " ~ (widthM / meter) ~ ",\n";
            json ~= " \"path_length\": " ~ (curveLength / meter) ~ ",\n";
            json ~= " \"sections\": [\n";
            for (var s = 0; s < size(sectionData); s += 1)
            {
                var sd = sectionData[s];
                json ~= "  { \"i\": " ~ sd["section"] ~ ", ";
                json ~= "\"throat_cm2\": " ~ sd["throat_area_cm2"] ~ ", ";
                json ~= "\"mouth_cm2\": " ~ sd["mouth_area_cm2"] ~ ", ";
                json ~= "\"uL\": " ~ sd["uL"] ~ " }";
                if (s < size(sectionData) - 1) json ~= ",";
                json ~= "\n";
            }
            json ~= " ],\n";
            json ~= " \"metadata\": {\n";
            json ~= "  \"t\": " ~ definition.t ~ ",\n";
            json ~= "  \"s1_cm2\": " ~ definition.s1Cm2 ~ ",\n";
            json ~= "  \"s2_cm2\": " ~ definition.s2Cm2 ~ ",\n";
            json ~= "  \"uL_global\": " ~ globalUL ~ ",\n";
            json ~= "  \"f12_hz\": " ~ round(globalF12 * 100.0) / 100.0 ~ "\n";
            json ~= " }\n";
            json ~= "}\n";

            setFeatureComputedParameter(context, id, { "name" : "pyhorn_sections_json", "value" : json });
        }

        reportFeatureInfo(context, id,
            numSections ~ " section(s), F12=" ~ round(globalF12 * 100.0) / 100.0 ~ " Hz, "
            ~ (widthM / meter) ~ " m wide, path=" ~ round((curveLength / meter) * 100.0) / 100.0 ~ " m");
    }
);
