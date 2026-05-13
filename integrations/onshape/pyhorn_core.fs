FeatureScript 2945;
import(path : "onshape/std/common.fs", version : "2931.0");


/**
 * pyhorn_core — shared helpers for Pyhorn Onshape integrations.
 *
 * Contains horn math, trace-building, and geometry utilities.
 * stdlib helpers (sinh, cosh, PI, etc.) are used directly — not duplicated here.
 *
 * NO feature annotations — this is a pure library module.
 * Import into a Feature Studio tab, publish, then import by doc ID in consumers.
 */

// ── Constants ───────────────────────────────────────────────────────────────────
// Note: PI is available as a built-in constant from stdlib. Use PI directly.

// ── Math helpers ───────────────────────────────────────────────────────────────
// Note: sinh/cosh are provided by the stdlib via common.fs → mathUtils → math.
// Use them directly. Only convenience wrappers below.

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

// ── Path length: chord-length sum (mm for numeric stability) ───────────────────

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

// ── Profile-area calculators ───────────────────────────────────────────────────
// All areas are ValueWithUnits (cm^2).

// Hyperbolic: area_ratio = cosh(uL*x) + T*sinh(uL*x)
export function _hyperbolicAreaAtFraction(
    throatArea    is ValueWithUnits,
    uL            is number,
    t             is number,
    fraction      is number
) returns ValueWithUnits
{
    var ratio = cosh(uL * fraction) + t * sinh(uL * fraction);
    return throatArea * ratio * ratio;
}

// Exponential: area_ratio = e^(mL*x)  where mL = ln(S2/S1)
export function _exponentialAreaAtFraction(
    throatArea    is ValueWithUnits,
    mL            is number,
    fraction      is number
) returns ValueWithUnits
{
    return throatArea * exp(mL * fraction);
}

// Conical: linear area expansion
export function _conicalAreaAtFraction(
    throatArea    is ValueWithUnits,
    mouthArea     is ValueWithUnits,
    fraction      is number
) returns ValueWithUnits
{
    return throatArea + (mouthArea - throatArea) * fraction;
}

// ── Solve flare constant from S1, S2, T ─────────────────────────────────────
// Solves for uL in:  cosh(uL) + T*sinh(uL) = S2/S1
export function _solveFlareArgumentFromAreasAndT(
    throatArea    is ValueWithUnits,
    mouthArea     is ValueWithUnits,
    t             is number
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

    var root      = sqrt(discriminant);
    var denominator = 1.0 + t;
    var candidateA = (areaRatioSqrt + root) / denominator;
    var candidateB = (areaRatioSqrt - root) / denominator;

    if (candidateA > 1.0) { return log(candidateA); }
    if (candidateB > 1.0) { return log(candidateB); }

    throw regenError("S1, S2, and T imply a non-expanding hyperbolic horn.");
}

// ── Sketch-plane derivation ─────────────────────────────────────────────────────

// Derive a sketch plane from three non-collinear 3D points (first three of a trace).
export function _deriveSketchPlane(traces is array) returns Plane
{
    if (size(traces) == 0 || size(traces[0]) < 3)
    {
        return plane(coordSystem(vector(0, 0, 0) * meter, vector(1, 0, 0), vector(0, 0, 1)));
    }

    var pA = traces[0][0];
    var pB = traces[0][1];
    var pC = traces[0][2];

    var v1x = pB[0] - pA[0];
    var v1y = pB[1] - pA[1];
    var v1z = pB[2] - pA[2];
    var v2x = pC[0] - pA[0];
    var v2y = pC[1] - pA[1];
    var v2z = pC[2] - pA[2];

    // Normal = v1 × v2
    var nx = v1y * v2z - v1z * v2y;
    var ny = v1z * v2x - v1x * v2z;
    var nz = v1x * v2y - v1y * v2x;
    var nMag = sqrt(nx * nx + ny * ny + nz * nz);

    if (nMag < 1e-12 * meter * meter)
    {
        return plane(coordSystem(pA, vector(1, 0, 0), vector(0, 0, 1)));
    }

    var nHat = vector(nx / nMag, ny / nMag, nz / nMag);

    // x-axis: project first tangent onto plane perpendicular to normal
    var tLen = sqrt(v1x * v1x + v1y * v1y + v1z * v1z);
    var tHat = vector(v1x / tLen, v1y / tLen, v1z / tLen);

    // Gram-Schmidt: xDir = t - (t·n)n
    var dotTN = tHat[0] * nHat[0] + tHat[1] * nHat[1] + tHat[2] * nHat[2];
    var xDirX = tHat[0] - dotTN * nHat[0];
    var xDirY = tHat[1] - dotTN * nHat[1];
    var xDirZ = tHat[2] - dotTN * nHat[2];
    var xMag = sqrt(xDirX * xDirX + xDirY * xDirY + xDirZ * xDirZ);

    if (xMag < 1e-10)
    {
        var seed = (abs(nHat[0]) < 0.9) ? vector(1, 0, 0) : vector(0, 1, 0);
        var cx = nHat[1] * seed[2] - nHat[2] * seed[1];
        var cy = nHat[2] * seed[0] - nHat[0] * seed[2];
        var cz = nHat[0] * seed[1] - nHat[1] * seed[0];
        var cMag = sqrt(cx * cx + cy * cy + cz * cz);
        xDirX = cx / cMag;
        xDirY = cy / cMag;
        xDirZ = cz / cMag;
    }
    else
    {
        xDirX /= xMag;
        xDirY /= xMag;
        xDirZ /= xMag;
    }

    return plane(coordSystem(pA, vector(xDirX, xDirY, xDirZ), nHat));
}

// ── Trace-builder helpers ───────────────────────────────────────────────────────

// Best-continuation lookup for greedy edge-walking.
function _findNext(lastPt is array, prevPt is array, edgeData is array, nEdges is number, TOL_METERS is ValueWithUnits) returns map
{
    var bestAngle = PI * radian;
    var nextIdx   = -1;
    var forward   = true;

    for (var j = 0; j < nEdges; j += 1)
    {
        if (edgeData[j]["used"]) { continue; }

        var q0 = edgeData[j]["p0"];
        var q1 = edgeData[j]["p1"];

        var d0x = lastPt[0] - q0[0];
        var d0y = lastPt[1] - q0[1];
        var d0z = lastPt[2] - q0[2];
        var d1x = lastPt[0] - q1[0];
        var d1y = lastPt[1] - q1[1];
        var d1z = lastPt[2] - q1[2];

        var dist0 = sqrt(d0x * d0x + d0y * d0y + d0z * d0z);
        var dist1 = sqrt(d1x * d1x + d1y * d1y + d1z * d1z);

        if (dist0 >= TOL_METERS && dist1 >= TOL_METERS) { continue; }

        var cX = (dist0 < TOL_METERS) ? (q1[0] - q0[0]) : (q0[0] - q1[0]);
        var cY = (dist0 < TOL_METERS) ? (q1[1] - q0[1]) : (q0[1] - q1[1]);
        var cZ = (dist0 < TOL_METERS) ? (q1[2] - q0[2]) : (q0[2] - q1[2]);
        var lC = sqrt(cX * cX + cY * cY + cZ * cZ);
        if (lC < 1e-9 * meter) { continue; }

        var pX = lastPt[0] - prevPt[0];
        var pY = lastPt[1] - prevPt[1];
        var pZ = lastPt[2] - prevPt[2];
        var lP = sqrt(pX * pX + pY * pY + pZ * pZ);
        if (lP < 1e-9 * meter) { continue; }

        var cosA = clamp((pX * cX + pY * cY + pZ * cZ) / (lP * lC), -1.0, 1.0);
        var angle = acos(cosA);

        if (angle < bestAngle)
        {
            bestAngle = angle;
            nextIdx   = j;
            forward   = dist0 < TOL_METERS;
        }
    }
    return { "nextIdx": nextIdx, "forward": forward };
}

// Sample an edge with reversal — thin wrapper around _sampleEdge.
// Reversal is needed when building traces backwards from a trace end.
function _sampleEdgePoint(context is Context, edgeData is array, idx is number, numPts is number, reversed is boolean) returns array
{
    var pts = _sampleEdge(context, edgeData[idx]["query"], numPts);
    if (reversed)
    {
        var rev = [];
        for (var k = numPts; k >= 0; k -= 1) { rev = append(rev, pts[k]); }
        return rev;
    }
    return pts;
}

// Sample an edge into (numPts + 1) evenly-spaced 3D waypoints.
export function _sampleEdge(context is Context, edgeQuery is Query, numPts is number) returns array
{
    var params = [];
    for (var j = 0; j <= numPts; j += 1) { params = append(params, j / numPts); }
    var ev = evEdgeTangentLines(context, { "edge" : edgeQuery, "parameters" : params });
    var pts = [];
    for (var j = 0; j <= numPts; j += 1)
    {
        var p = ev[j].origin;
        pts = append(pts, [p[0], p[1], p[2]]);
    }
    return pts;
}

// ── Trace builder ─────────────────────────────────────────────────────────────
// Greedy edge-walk that joins adjacent collinear edge fragments into continuous
// wall traces.
//
// Returns an array of traces; each trace is an array of [x, y, z] waypoints.
export function _buildTraces(context is Context, rawEdges is array, samplesPerEdge is number, TOL_METERS is ValueWithUnits) returns array
{
    // 1. Cache every edge's endpoints once
    var edgeData = [];
    for (var i = 0; i < size(rawEdges); i += 1)
    {
        var ep = evEdgeTangentLines(context, {
                "edge" : rawEdges[i],
                "parameters" : [0.0, 1.0]
            });
        edgeData = append(edgeData, {
            "query": rawEdges[i],
            "p0": ep[0].origin,
            "p1": ep[1].origin,
            "used": false
        });
    }

    var traces = [];
    var numPts = samplesPerEdge;
    var nEdges = size(rawEdges);

    // 2. Pass 1 — forward walk from each unused edge
    for (var start = 0; start < nEdges; start += 1)
    {
        if (edgeData[start]["used"]) { continue; }

        var trace  = [];
        var cur    = start;
        var prevPt = [0 * meter, 0 * meter, 0 * meter];

        while (cur >= 0 && !edgeData[cur]["used"])
        {
            edgeData[cur]["used"] = true;
            var pts = _sampleEdgePoint(context, edgeData, cur, numPts, false);

            if (size(trace) == 0)
            {
                for (var k = 0; k <= numPts; k += 1) { trace = append(trace, pts[k]); }
                prevPt = pts[numPts - 1];
            }
            else
            {
                for (var k = 1; k <= numPts; k += 1) { trace = append(trace, pts[k]); }
                prevPt = trace[size(trace) - numPts - 1];
            }

            var lastPt = trace[size(trace) - 1];
            var cont   = _findNext(lastPt, prevPt, edgeData, nEdges, TOL_METERS);
            if (cont["nextIdx"] < 0) { break; }
            cur = cont["nextIdx"];
        }

        if (size(trace) > 0) { traces = append(traces, trace); }
    }

    // 3. Pass 2 — backward walk from open trace starts
    for (var t = 0; t < size(traces); t += 1)
    {
        var trace = traces[t];
        if (size(trace) < 2) { continue; }

        var lastPt = trace[0];
        var prevPt = trace[1];

        var cont = _findNext(lastPt, prevPt, edgeData, nEdges, TOL_METERS);
        if (cont["nextIdx"] < 0) { continue; }

        var current  = cont["nextIdx"];
        var extended = false;

        while (current >= 0 && !edgeData[current]["used"])
        {
            edgeData[current]["used"] = true;
            var pts = _sampleEdgePoint(context, edgeData, current, numPts, cont["forward"]);

            var prefix = [];
            for (var k = numPts; k >= 1; k -= 1)
            {
                prefix = append(prefix, pts[k]);
            }
            trace = concatenateArrays([prefix, trace]);

            lastPt = trace[0];
            prevPt = trace[1];

            cont = _findNext(lastPt, prevPt, edgeData, nEdges, TOL_METERS);
            if (cont["nextIdx"] < 0) { break; }
            current = cont["nextIdx"];
            extended = true;
        }

        if (extended) { traces[t] = trace; }
    }

    return traces;
}

// ── Sketch helpers ─────────────────────────────────────────────────────────────

// Draw the centreline path as a polyline sketch entity.
export function _sketchCenterline(context is Context, id is Id, sampledLines is array, sketchPlane is Plane)
{
    var sketchId = id + "centerline_PATH";
    var sketch = newSketchOnPlane(context, sketchId, {
            "sketchPlane" : sketchPlane
        });

    var segIdGen = getUnstableIncrementingId(sketchId);
    for (var i = 0; i < size(sampledLines) - 1; i += 1)
    {
        var pA = worldToPlane(sketchPlane, sampledLines[i].origin);
        var pB = worldToPlane(sketchPlane, sampledLines[i + 1].origin);
        skLineSegment(sketch, segIdGen(), {
                "start" : pA,
                "end"   : pB
            });
    }

    skSolve(sketch);
}

// Draw the longest trace as a closed polygon sketch.
export function _sketchPerimeter(context is Context, id is Id, traces is array, sketchPlane is Plane)
{
    // Find the longest trace — assume that is the outer perimeter
    var longestIdx = 0;
    var longestLen = 0.0 * meter;

    for (var i = 0; i < size(traces); i += 1)
    {
        var L = 0.0 * meter;
        for (var k = 0; k < size(traces[i]) - 1; k += 1)
        {
            var dx = traces[i][k + 1][0] - traces[i][k][0];
            var dy = traces[i][k + 1][1] - traces[i][k][1];
            var dz = traces[i][k + 1][2] - traces[i][k][2];
            L += sqrt(dx * dx + dy * dy + dz * dz);
        }
        if (L > longestLen)
        {
            longestLen = L;
            longestIdx = i;
        }
    }

    var sketchId = id + "perimeter";
    var sketch = newSketchOnPlane(context, sketchId, {
            "sketchPlane" : sketchPlane
        });

    var segIdGen = getUnstableIncrementingId(sketchId);
    var trace = traces[longestIdx];
    for (var k = 0; k < size(trace) - 1; k += 1)
    {
        skLineSegment(sketch, segIdGen(), {
                "start" : trace[k],
                "end"   : trace[k + 1]
            });
    }
    // Close the polygon
    skLineSegment(sketch, segIdGen(), {
            "start" : trace[size(trace) - 1],
            "end"   : trace[0]
        });

    skSolve(sketch);
}

// ── Arc-length lookup ─────────────────────────────────────────────────────────
// Given sampledLines and a target arc-length fraction, return the 3D position
// and tangent direction at that arc-length station.
export function _arcLengthStation(
    sampledLines   is array,
    curveLength    is ValueWithUnits,
    fraction       is number
) returns map
{
    var targetDistMM = (curveLength / millimeter) * fraction;
    var cumDistMM = 0.0;
    var samplePos = sampledLines[0].origin;
    var sampleDir = sampledLines[0].direction;

    for (var s = 0; s < size(sampledLines) - 1; s += 1)
    {
        var segDxMM = (sampledLines[s + 1].origin[0] - sampledLines[s].origin[0]) / millimeter;
        var segDyMM = (sampledLines[s + 1].origin[1] - sampledLines[s].origin[1]) / millimeter;
        var segDzMM = (sampledLines[s + 1].origin[2] - sampledLines[s].origin[2]) / millimeter;
        var segLenMM = sqrt(segDxMM * segDxMM + segDyMM * segDyMM + segDzMM * segDzMM);

        if (cumDistMM + segLenMM >= targetDistMM || s == size(sampledLines) - 2)
        {
            var t = (s == size(sampledLines) - 2) ? 1.0
                : (targetDistMM - cumDistMM) / segLenMM;
            samplePos = vector(
                sampledLines[s].origin[0] + t * (sampledLines[s + 1].origin[0] - sampledLines[s].origin[0]),
                sampledLines[s].origin[1] + t * (sampledLines[s + 1].origin[1] - sampledLines[s].origin[1]),
                sampledLines[s].origin[2] + t * (sampledLines[s + 1].origin[2] - sampledLines[s].origin[2])
            );
            sampleDir = sampledLines[s].direction;
            break;
        }
        cumDistMM += segLenMM;
    }

    return { "pos": samplePos, "dir": sampleDir };
}

// ── Edge-chain ordering ────────────────────────────────────────────────────────
// Given a list of edges, order them into a connected chain using greedy
// nearest-endpoint join.  Returns { order: [edgeIdx,...], flip: [bool,...] }.
export function _orderEdgeChain(edges is array, edgeEndpoints is array) returns map
{
    var n = size(edges);
    var order = [0];
    var flip  = [false];
    var used  = makeArray(n, false);
    used[0] = true;

    for (var step = 1; step < n; step += 1)
    {
        var prevIdx  = order[size(order) - 1];
        var prevFlip = flip[size(flip) - 1];
        var chainEnd = prevFlip
            ? edgeEndpoints[prevIdx]["start"]
            : edgeEndpoints[prevIdx]["end"];

        var bestIdx  = -1;
        var bestDist = 1e30 * meter;
        var bestFlip = false;

        for (var j = 0; j < n; j += 1)
        {
            if (!used[j])
            {
                var dStart = _point3dDist(chainEnd, edgeEndpoints[j]["start"]);
                var dEnd   = _point3dDist(chainEnd, edgeEndpoints[j]["end"]);

                if (dStart < bestDist)
                {
                    bestDist = dStart;
                    bestIdx  = j;
                    bestFlip = false;
                }
                if (dEnd < bestDist)
                {
                    bestDist = dEnd;
                    bestIdx  = j;
                    bestFlip = true;
                }
            }
        }

        order = append(order, bestIdx);
        flip  = append(flip,  bestFlip);
        used[bestIdx] = true;
    }

    return { "order": order, "flip": flip };
}

// ── Bend-angle computation ─────────────────────────────────────────────────────
// Compute centreline bend angles (degrees) at each station.
// angle at station i = angle between (tCurr → tNext) tangent vectors.
export function _computeBendAngles(sampledLines is array, sketchPlane is Plane) returns array
{
    var bendAngles = [];
    var n = size(sampledLines);

    for (var i = 0; i < n; i += 1)
    {
        if (i == 0 || i == n - 1)
        {
            bendAngles = append(bendAngles, 0.0);
            continue;
        }

        var tCurr = worldToPlane(sketchPlane, sampledLines[i].origin + sampledLines[i].direction * millimeter)
                  - worldToPlane(sketchPlane, sampledLines[i].origin);
        var tNext = worldToPlane(sketchPlane, sampledLines[i + 1].origin + sampledLines[i + 1].direction * millimeter)
                  - worldToPlane(sketchPlane, sampledLines[i + 1].origin);

        var tCurrMag = _vectorMagnitude(tCurr);
        var tNextMag = _vectorMagnitude(tNext);

        if (tCurrMag < 1e-9 * meter || tNextMag < 1e-9 * meter)
        {
            bendAngles = append(bendAngles, 0.0);
            continue;
        }

        var tCurrU = tCurr / tCurrMag;
        var tNextU = tNext / tNextMag;

        var cosDelta = clamp(tCurrU[0] * tNextU[0] + tCurrU[1] * tNextU[1], -1.0, 1.0);
        var deltaDeg = acos(cosDelta) * 180.0 / PI;

        bendAngles = append(bendAngles, deltaDeg);
    }

    return bendAngles;
}

// ── Sketch-plane from centreline samples ─────────────────────────────────────
// Derive a sketch plane from an ordered array of sampled tangent lines.
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
    var tangent = sampledLines[0].direction;
    var xDir = tangent - dot(tangent, normalUnit) * normalUnit;
    var xMag = sqrt(xDir[0] * xDir[0] + xDir[1] * xDir[1] + xDir[2] * xDir[2]);

    if (xMag < 1e-10)
    {
        var seed = (abs(normalUnit[0]) < 0.9) ? vector(1, 0, 0) : vector(0, 1, 0);
        xDir = cross(normalUnit, seed);
        xMag = sqrt(xDir[0] * xDir[0] + xDir[1] * xDir[1] + xDir[2] * xDir[2]);
    }

    xDir = xDir / xMag;
    return plane(coordSystem(pA, xDir, normalUnit));
}

// ── JSON helpers ───────────────────────────────────────────────────────────────
// _roundCoord and _containsEntity are inlined in pyhorn_auto_segment.fs
// since they are only used there.
