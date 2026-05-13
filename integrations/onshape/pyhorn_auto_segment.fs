FeatureScript 2931;
import(path : "onshape/std/common.fs", version : "2931.0");
import(path : "onshape/std/geometry.fs", version : "2931.0");

// ── pyhorn_core ────────────────────────────────────────────────────────────────
// After publishing pyhorn_core.fs to a Feature Studio tab, replace the inline
// helpers below with this import:
//   import(path : "<pyhorn-core-doc-id>", version : "<version-hash>");
//
// The following helpers are currently inlined (mirrors of pyhorn_core):
//   _sampleEdge, _buildTraces, _deriveSketchPlane, _sketchPerimeter

const TOL_METERS = 1e-6 * meter;

// ── Helpers (from pyhorn_core — remove once importing pyhorn_core) ─────────────────

function _containsEntity(context is Context, parentQuery is Query, target is Query) returns boolean
{
    return !isQueryEmpty(context, qIntersection([parentQuery, target]));
}

function _sampleEdge(context is Context, edgeQuery is Query, numPts is number) returns array
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

function _sampleEdgePts(context is Context, edgeData is array, idx is number, numPts is number, reversed is boolean) returns array
{
    var params = [];
    for (var k = 0; k <= numPts; k += 1) { params = append(params, k / numPts); }
    var ev = evEdgeTangentLines(context, { "edge" : edgeData[idx]["query"], "parameters" : params });
    if (reversed)
    {
        var rev = [];
        for (var k = numPts; k >= 0; k -= 1) { var pt = ev[k].origin; rev = append(rev, [pt[0], pt[1], pt[2]]); }
        return rev;
    }
    var pts = [];
    for (var k = 0; k <= numPts; k += 1) { var pt = ev[k].origin; pts = append(pts, [pt[0], pt[1], pt[2]]); }
    return pts;
}

function _buildTraces(context is Context, rawEdges is array, samplesPerEdge is number) returns array
{
    var edgeData = [];
    for (var i = 0; i < size(rawEdges); i += 1)
    {
        var ep = evEdgeTangentLines(context, { "edge" : rawEdges[i], "parameters" : [0.0, 1.0] });
        edgeData = append(edgeData, { "query": rawEdges[i], "p0": ep[0].origin, "p1": ep[1].origin, "used": false });
    }

    var traces = [];
    var numPts = samplesPerEdge;
    var nEdges = size(rawEdges);

    // Pass 1 — forward walk
    for (var start = 0; start < nEdges; start += 1)
    {
        if (edgeData[start]["used"]) { continue; }
        var trace = []; var cur = start;
        var prevPt = [0 * meter, 0 * meter, 0 * meter];

        while (cur >= 0 && !edgeData[cur]["used"])
        {
            edgeData[cur]["used"] = true;
            var pts = _sampleEdgePts(context, edgeData, cur, numPts, false);
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
            var cont = _findNext(lastPt, prevPt, edgeData, nEdges, TOL_METERS);
            if (cont["nextIdx"] < 0) { break; }
            cur = cont["nextIdx"];
        }
        if (size(trace) > 0) { traces = append(traces, trace); }
    }

    // Pass 2 — backward walk from open starts
    for (var t = 0; t < size(traces); t += 1)
    {
        var trace = traces[t];
        if (size(trace) < 2) { continue; }
        var lastPt = trace[0]; var prevPt = trace[1];
        var cont = _findNext(lastPt, prevPt, edgeData, nEdges, TOL_METERS);
        if (cont["nextIdx"] < 0) { continue; }
        var current = cont["nextIdx"]; var extended = false;

        while (current >= 0 && !edgeData[current]["used"])
        {
            edgeData[current]["used"] = true;
            var pts = _sampleEdgePts(context, edgeData, current, numPts, cont["forward"]);
            var prefix = [];
            for (var k = numPts; k >= 1; k -= 1) { prefix = append(prefix, pts[k]); }
            trace = concatenateArrays([prefix, trace]);
            lastPt = trace[0]; prevPt = trace[1];
            cont = _findNext(lastPt, prevPt, edgeData, nEdges, TOL_METERS);
            if (cont["nextIdx"] < 0) { break; }
            current = cont["nextIdx"]; extended = true;
        }
        if (extended) { traces[t] = trace; }
    }
    return traces;
}

function _deriveSketchPlane(traces is array) returns Plane
{
    if (size(traces) == 0 || size(traces[0]) < 3) { return plane(coordSystem(vector(0,0,0)*meter, vector(1,0,0), vector(0,0,1))); }
    var pA = traces[0][0]; var pB = traces[0][1]; var pC = traces[0][2];
    var v1x = pB[0]-pA[0]; var v1y = pB[1]-pA[1]; var v1z = pB[2]-pA[2];
    var v2x = pC[0]-pA[0]; var v2y = pC[1]-pA[1]; var v2z = pC[2]-pA[2];
    var nx = v1y*v2z - v1z*v2y; var ny = v1z*v2x - v1x*v2z; var nz = v1x*v2y - v1y*v2x;
    var nMag = sqrt(nx*nx + ny*ny + nz*nz);
    if (nMag < 1e-12 * meter * meter) { return plane(coordSystem(pA, vector(1,0,0), vector(0,0,1))); }
    var nHat = vector(nx/nMag, ny/nMag, nz/nMag);
    var tLen = sqrt(v1x*v1x + v1y*v1y + v1z*v1z);
    var tHat = vector(v1x/tLen, v1y/tLen, v1z/tLen);
    var dotTN = tHat[0]*nHat[0] + tHat[1]*nHat[1] + tHat[2]*nHat[2];
    var xDirX = tHat[0]-dotTN*nHat[0]; var xDirY = tHat[1]-dotTN*nHat[1]; var xDirZ = tHat[2]-dotTN*nHat[2];
    var xMag = sqrt(xDirX*xDirX + xDirY*xDirY + xDirZ*xDirZ);
    if (xMag < 1e-10)
    {
        var seed = (abs(nHat[0]) < 0.9) ? vector(1,0,0) : vector(0,1,0);
        var cx = nHat[1]*seed[2]-nHat[2]*seed[1]; var cy = nHat[2]*seed[0]-nHat[0]*seed[2]; var cz = nHat[0]*seed[1]-nHat[1]*seed[0];
        var cMag = sqrt(cx*cx + cy*cy + cz*cz);
        xDirX = cx/cMag; xDirY = cy/cMag; xDirZ = cz/cMag;
    }
    else { xDirX /= xMag; xDirY /= xMag; xDirZ /= xMag; }
    return plane(coordSystem(pA, vector(xDirX, xDirY, xDirZ), nHat));
}

function _sketchPerimeter(context is Context, id is Id, traces is array, sketchPlane is Plane)
{
    var longestIdx = 0; var longestLen = 0.0 * meter;
    for (var i = 0; i < size(traces); i += 1)
    {
        var L = 0.0 * meter;
        for (var k = 0; k < size(traces[i]) - 1; k += 1)
        {
            var dx = traces[i][k+1][0]-traces[i][k][0]; var dy = traces[i][k+1][1]-traces[i][k][1]; var dz = traces[i][k+1][2]-traces[i][k][2];
            L += sqrt(dx*dx + dy*dy + dz*dz);
        }
        if (L > longestLen) { longestLen = L; longestIdx = i; }
    }
    var sketchId = id + "perimeter";
    var sketch = newSketchOnPlane(context, sketchId, { "sketchPlane" : sketchPlane });
    var trace = traces[longestIdx];
    for (var k = 0; k < size(trace) - 1; k += 1)
    {
        skLineSegment(sketch, "_e" ~ k, { "start" : trace[k], "end" : trace[k+1] });
    }
    skLineSegment(sketch, "_e" ~ size(trace), { "start" : trace[size(trace)-1], "end" : trace[0] });
    skSolve(sketch);
}

// ── Feature ────────────────────────────────────────────────────────────────────

annotation { "Feature Type Name" : "Pyhorn Auto-Segment",
             "Feature Type Description" : "Export 2D air volume for Pyhorn medial-axis - sketch-polygon + JSON via computed parameter" }
export const pyhornAutoSegment = defineFeature(function(context is Context, id is Id, definition is map)
    precondition
    {
        annotation { "Name" : "Internal Width", "UIHint" : UIHint.REMEMBER_PREVIOUS_VALUE }
        isLength(definition.width, LENGTH_BOUNDS);

        annotation { "Name" : "Samples Per Edge", "Default" : 32, "Min" : 4, "Max" : 500 }
        isInteger(definition.samplesPerEdge, { (unitless) : [4, 32, 500] } as IntegerBoundSpec);

        annotation { "Name" : "2D Air Volume Face", "Filter" : EntityType.FACE, "MaxNumberOfPicks" : 1 }
        definition.airFace is Query;

        annotation { "Name" : "Throat Edge", "Filter" : EntityType.EDGE, "MaxNumberOfPicks" : 1 }
        definition.throatEdge is Query;

        annotation { "Name" : "Mouth Edge", "Filter" : EntityType.EDGE, "MaxNumberOfPicks" : 1 }
        definition.mouthEdge is Query;

        annotation { "Name" : "Draw sketch polygon", "Default" : true }
        definition.drawPolygon is boolean;
    }
    {
        var faceEdges = qAdjacent(definition.airFace, AdjacencyType.EDGE, EntityType.EDGE);
        var allEdges  = evaluateQuery(context, faceEdges);

        if (size(allEdges) == 0) { throw regenError("Selected face has no edges."); }
        if (definition.samplesPerEdge < 4 || definition.samplesPerEdge > 500) { throw regenError("Samples Per Edge must be between 4 and 500."); }
        if (!_containsEntity(context, faceEdges, definition.throatEdge)) { throw regenError("Throat Edge must be an edge of the selected 2D Air Volume Face."); }
        if (!_containsEntity(context, faceEdges, definition.mouthEdge)) { throw regenError("Mouth Edge must be an edge of the selected 2D Air Volume Face."); }

        var throatPick = evaluateQuery(context, definition.throatEdge);
        var mouthPick  = evaluateQuery(context, definition.mouthEdge);
        if (size(throatPick) != 1 || size(mouthPick) != 1) { throw regenError("Select exactly one throat edge and one mouth edge."); }
        if (throatPick[0] == mouthPick[0]) { throw regenError("Throat Edge and Mouth Edge must be different edges."); }

        var numPts = definition.samplesPerEdge;

        // Raw flat edge list
        var rawEdgeList = [];
        for (var i = 0; i < size(allEdges); i += 1)
        {
            rawEdgeList = append(rawEdgeList, _sampleEdge(context, allEdges[i], numPts));
        }

        // Pre-merged continuous wall traces
        var traces = _buildTraces(context, allEdges, numPts);

        // Map throat / mouth edge to raw edge index
        var throatIdx = -1; var mouthIdx = -1;
        for (var i = 0; i < size(allEdges); i += 1)
        {
            if (allEdges[i] == throatPick[0]) { throatIdx = i; }
            if (allEdges[i] == mouthPick[0])  { mouthIdx = i; }
        }

        // Throat and mouth endpoints
        var tPts = evEdgeTangentLines(context, { "edge" : definition.throatEdge, "parameters" : [0.0, 1.0] });
        var mPts = evEdgeTangentLines(context, { "edge" : definition.mouthEdge,  "parameters" : [0.0, 1.0] });
        var widthMM = round((definition.width / meter) * 1000.0) / 1000.0;

        // Build JSON
        var json = "{\n";
        json ~= "  \"version\": 3,\n";
        json ~= "  \"width\": " ~ widthMM ~ ",\n";
        json ~= "  \"throat\": [["
             ~ round((tPts[0].origin[0]/meter)*1e6)/1e6 ~ ", " ~ round((tPts[0].origin[1]/meter)*1e6)/1e6 ~ ", " ~ round((tPts[0].origin[2]/meter)*1e6)/1e6 ~ "], ["
             ~ round((tPts[1].origin[0]/meter)*1e6)/1e6 ~ ", " ~ round((tPts[1].origin[1]/meter)*1e6)/1e6 ~ ", " ~ round((tPts[1].origin[2]/meter)*1e6)/1e6 ~ "]],\n";
        json ~= "  \"mouth\": [["
             ~ round((mPts[0].origin[0]/meter)*1e6)/1e6 ~ ", " ~ round((mPts[0].origin[1]/meter)*1e6)/1e6 ~ ", " ~ round((mPts[0].origin[2]/meter)*1e6)/1e6 ~ "], ["
             ~ round((mPts[1].origin[0]/meter)*1e6)/1e6 ~ ", " ~ round((mPts[1].origin[1]/meter)*1e6)/1e6 ~ ", " ~ round((mPts[1].origin[2]/meter)*1e6)/1e6 ~ "]],\n";
        json ~= "  \"boundary_edges\": [\n";
        for (var i = 0; i < size(rawEdgeList); i += 1)
        {
            json ~= "    [\n";
            for (var j = 0; j <= numPts; j += 1)
            {
                var p = rawEdgeList[i][j];
                json ~= "      [" ~ round((p[0]/meter)*1e6)/1e6 ~ ", " ~ round((p[1]/meter)*1e6)/1e6 ~ ", " ~ round((p[2]/meter)*1e6)/1e6 ~ "]";
                if (j < numPts) json ~= ","; json ~= "\n";
            }
            json ~= "    ]"; if (i < size(rawEdgeList) - 1) json ~= ","; json ~= "\n";
        }
        json ~= "  ],\n";
        json ~= "  \"pre_merged\": [\n";
        for (var i = 0; i < size(traces); i += 1)
        {
            json ~= "    [\n";
            for (var j = 0; j < size(traces[i]); j += 1)
            {
                var p = traces[i][j];
                json ~= "      [" ~ round((p[0]/meter)*1e6)/1e6 ~ ", " ~ round((p[1]/meter)*1e6)/1e6 ~ ", " ~ round((p[2]/meter)*1e6)/1e6 ~ "]";
                if (j < size(traces[i]) - 1) json ~= ","; json ~= "\n";
            }
            json ~= "    ]"; if (i < size(traces) - 1) json ~= ","; json ~= "\n";
        }
        json ~= "  ],\n";
        json ~= "  \"roles\": {\"throat\": " ~ throatIdx ~ ", \"mouth\": " ~ mouthIdx ~ "},\n";
        json ~= "  \"metadata\": {\n";
        json ~= "    \"samples_per_edge\": " ~ numPts ~ ",\n";
        json ~= "    \"n_raw_edges\": " ~ size(allEdges) ~ ",\n";
        json ~= "    \"n_traces\": " ~ size(traces) ~ "\n";
        json ~= "  }\n";
        json ~= "}\n";

        // Draw sketch polygon
        if (definition.drawPolygon)
        {
            var sketchPlane = _deriveSketchPlane(traces);
            _sketchPerimeter(context, id, traces, sketchPlane);
        }

        // Store JSON as computed parameter — readable via Onshape API FeatureInfo
        setFeatureComputedParameter(context, id, {
                "name"  : "pyhorn_json",
                "value" : json
            });

        // Console preview (truncated)
        var previewLen = min(500, size(json));
        var suffix = size(json) > previewLen ? "\n... [truncated, full JSON stored as feature computed parameter]" : "";

        println("===============================");
        println("PYHORN AUTO-SEGMENT JSON:");
        println("===============================");
        println(substring(json, 0, previewLen) ~ suffix);
        println("===============================");

        reportFeatureInfo(context, id,
            size(traces) ~ " wall trace(s), " ~ size(allEdges) ~ " raw edge(s), "
            ~ size(json) ~ " chars JSON (stored as feature parameter — read via API).");
    }
);
