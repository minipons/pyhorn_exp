FeatureScript 2945;
import(path : "onshape/std/common.fs", version : "2945.0");
import(path : "onshape/std/common.fs", version : "2945.0");
import(path : "c5f898a314df4e00b700db23", version : "36d3b6bfb2dcdf18c0b6b4a9");



// ── pyhorn_core ────────────────────────────────────────────────────────────────
// After publishing pyhorn_core.fs to a Feature Studio tab, replace the inline
// helpers below with this import:
// import(path : "<pyhorn-core-doc-id>", version : "<version-hash>");
//
// The following helpers are currently inlined (mirrors of pyhorn_core):
// _vectorMagnitude, _point3dDist, _pathLengthFromSamples,
// _hyperbolicAreaAtFraction, _exponentialAreaAtFraction, _conicalAreaAtFraction,
// _solveFlareArgumentFromAreasAndT, _orderEdgeChain, _arcLengthStation,
// _deriveSketchPlaneFromSamples, _computeBendAngles, _sketchCenterline,
// sinh, cosh (from stdlib via common.fs)

// ═══════════════════════════════════════════════════════════════════════════════
// _buildPyhornJson — the ONLY function unique to this file (not in pyhorn_core)
// ═══════════════════════════════════════════════════════════════════════════════


function _buildPyhornJson(
 throatArea is ValueWithUnits, mouthArea is ValueWithUnits,
 widthM is number, uL is number, mL is number,
 profileType is string, derivedF12 is number, curveLength is ValueWithUnits,
 leftOffset is array, rightOffset is array,
 sampledLines is array, sketchPlane is Plane,
 throatAreaCm2 is number, mouthAreaCm2 is number, tParam is number
) returns string
{
 var coords = [];
 var segHeights = [];
 var xMin = 1e30 * meter;
 var yMin = 1e30 * meter;
 var xMax = -1e30 * meter;
 var yMax = -1e30 * meter;
 for (var i = 0; i < size(leftOffset); i += 1)
 {
 var cx = (leftOffset[i][0] + rightOffset[i][0]) / 2.0;
 var cy = (leftOffset[i][1] + rightOffset[i][1]) / 2.0;
 var halfH = (leftOffset[i][1] - rightOffset[i][1]) / 2.0;
 if (cx < xMin)
 xMin = cx;
 if (cy < yMin)
 yMin = cy;
 if (cx > xMax)
 xMax = cx;
 if (cy > yMax)
 yMax = cy;
 coords = append(coords, [cx / meter, cy / meter]);
 segHeights = append(segHeights, halfH / meter);
 }
 var bendAngles = _computeBendAngles(sampledLines, sketchPlane);
 var throatX = coords[0][0];
 var throatY = coords[0][1];
 var throatH = segHeights[0];
 var mouthX = coords[size(coords) - 1][0];
 var mouthY = coords[size(coords) - 1][1];
 var mouthH = segHeights[size(segHeights) - 1];

 var json = "{\n";
 json ~= " \"enclosure_type\": \"BLH\",\n";
 json ~= " \"width\": " ~ widthM ~ ",\n";
 json ~= " \"dimensions\": [" ~ (xMin / meter) ~ ", " ~ (yMin / meter) ~ ", " ~ (xMax / meter) ~ ", " ~ (yMax / meter) ~ "],\n";
 json ~= " \"throat_segment\": [" ~ throatX ~ ", " ~ throatY ~ ", " ~ throatH ~ "],\n";
 json ~= " \"mouth_segment\": [" ~ mouthX ~ ", " ~ mouthY ~ ", " ~ mouthH ~ "],\n";
 json ~= " \"coordinates\": [\n";
 for (var i = 0; i < size(coords); i += 1)
 {
 json ~= " [" ~ coords[i][0] ~ ", " ~ coords[i][1] ~ "]";
 if (i < size(coords) - 1)
 json ~= ",";
 json ~= "\n";
 }
 json ~= " ],\n";
 json ~= " \"rectangular_segments\": [\n";
 for (var i = 0; i < size(segHeights); i += 1)
 {
 json ~= " " ~ segHeights[i];
 if (i < size(segHeights) - 1)
 json ~= ",";
 json ~= "\n";
 }
 json ~= " ],\n";
 json ~= " \"bend_angles\": [\n";
 for (var i = 0; i < size(bendAngles); i += 1)
 {
 json ~= " " ~ bendAngles[i];
 if (i < size(bendAngles) - 1)
 json ~= ",";
 json ~= "\n";
 }
 json ~= " ],\n";
 json ~= " \"path_length\": " ~ (curveLength / meter) ~ ",\n";
 json ~= " \"metadata\": {\n";
 json ~= " \"profile\": \"" ~ profileType ~ "\",\n";
 json ~= " \"s1_cm2\": " ~ throatAreaCm2 ~ ",\n";
 json ~= " \"s2_cm2\": " ~ mouthAreaCm2 ~ ",\n";
 json ~= " \"t\": " ~ tParam ~ ",\n";
 json ~= " \"uL\": " ~ uL ~ ",\n";
 json ~= " \"f12_hz\": " ~ round(derivedF12 * 100.0) / 100.0 ~ ",\n";
 json ~= " \"n_samples\": " ~ size(coords) ~ "\n";
 json ~= " }\n";
 json ~= "}\n";
 return json;
}

// ── Feature ────────────────────────────────────────────────────────────────────

annotation { "Feature Type Name" : "Horn Profile",
 "Feature Type Description" : "Sweep hyperbolic/exponential/conical horn along a centreline - machine-readable JSON + centreline sketch for API extraction" }
export const pyhornHorn = defineFeature(function(context is Context, id is Id, definition is map)
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

 annotation { "Name" : "Profile", "Default" : "hyperbolic" }
 definition.profile is string;

 annotation { "Name" : "Samples", "Default" : 500, "Min" : 8, "Max" : 2000 }
 isInteger(definition.samples, { (unitless) : [8, 500, 2000] } as IntegerBoundSpec);

 annotation { "Name" : "Output JSON", "Default" : true }
 definition.outputJson is boolean;

 annotation { "Name" : "Smooth Walls", "Default" : false }
 definition.smoothWalls is boolean;
 }
 {
 if (definition.width <= 0.0 * meter)
 {
 throw regenError("Internal Width must be greater than zero.");
 }

 // Resolve edges from selection
 var edges = evaluateQuery(context, qEntityFilter(definition.edge, EntityType.EDGE));
 if (size(edges) == 0)
 {
 edges = evaluateQuery(context, qOwnedByBody(definition.edge, EntityType.EDGE));
 }
 if (size(edges) == 0)
 {
 throw regenError("No edges found. Select edges or a composite curve.");
 }

 // Pre-compute endpoints for each edge
 var edgeEndpoints = [];
 for (var e = 0; e < size(edges); e += 1)
 {
 var ends = evEdgeTangentLines(context, { "edge" : edges[e], "parameters" : [0.0, 1.0] });
 edgeEndpoints = append(edgeEndpoints, { "start" : ends[0].origin, "end" : ends[1].origin });
 }

 // Order edges into a connected chain
 var chain = _orderEdgeChain(edges, edgeEndpoints);
 var order = chain["order"];
 var flip = chain["flip"];

 // Sample and concatenate edges
 var sampledLines = [];
 var samplesPerEdge = max(2, round(definition.samples / size(edges)));
 for (var c = 0; c < size(order); c += 1)
 {
 var eIdx = order[c];
 var isFlipped = flip[c];
 var params = [];
 for (var i = 0; i <= samplesPerEdge; i += 1)
 {
 params = append(params, i / samplesPerEdge);
 }
 var edgeSamples = evEdgeTangentLines(context, { "edge" : edges[eIdx], "parameters" : params });
 if (isFlipped)
 {
 var reversed = [];
 for (var s = size(edgeSamples) - 1; s >= 0; s -= 1)
 {
 reversed = append(reversed, line(edgeSamples[s].origin, -edgeSamples[s].direction));
 }
 edgeSamples = reversed;
 }
 var startIdx = (c == 0) ? 0 : 1;
 for (var s = startIdx; s < size(edgeSamples); s += 1)
 {
 sampledLines = append(sampledLines, edgeSamples[s]);
 }
 }

 if (size(sampledLines) < 2)
 {
 throw regenError("Selected centreline is too short or degenerate.");
 }
 var curveLength = _pathLengthFromSamples(sampledLines);
 if (curveLength <= 0.0 * meter)
 {
 throw regenError("Selected centreline has zero length.");
 }

 // Derive sketch plane
 var sketchPlane;
 try
 {
 sketchPlane = evOwnerSketchPlane(context, { "entity" : edges[0] });
 }
 catch
 {
 sketchPlane = _deriveSketchPlaneFromSamples(sampledLines);
 }

 // Areas and flare constant
 var throatArea = definition.s1Cm2 * centimeter * centimeter;
 var mouthArea = definition.s2Cm2 * centimeter * centimeter;
 var profileType = definition.profile;
 var uL = 0.0;
 var mL = 0.0;
 var derivedF12 = 0.0;

 if (profileType == "hyperbolic")
 {
 uL = _solveFlareArgumentFromAreasAndT(throatArea, mouthArea, definition.t);
 derivedF12 = (uL / (curveLength / meter)) * 343.0 / (2.0 * PI);
 }
 else if (profileType == "exponential")
 {
 mL = log(mouthArea / throatArea);
 uL = mL;
 derivedF12 = (mL / (curveLength / meter)) * 343.0 / (4.0 * PI);
 }

 // Arc-length parameterised sweep
 var totalSamples = definition.samples;
 var leftOffset = [];
 var rightOffset = [];
 var lastValidUnitPerp = undefined;

 for (var idx = 0; idx <= totalSamples; idx += 1)
 {
 var fraction = idx / totalSamples;
 var station = _arcLengthStation(sampledLines, curveLength, fraction);
 var samplePos = station["pos"];
 var sampleDir = station["dir"];

 var areaAtStation = throatArea;
 if (profileType == "hyperbolic")
 {
 areaAtStation = _hyperbolicAreaAtFraction(throatArea, uL, definition.t, fraction);
 }
 else if (profileType == "exponential")
 {
 areaAtStation = _exponentialAreaAtFraction(throatArea, mL, fraction);
 }
 else
 {
 areaAtStation = _conicalAreaAtFraction(throatArea, mouthArea, fraction);
 }

 var halfHeight = areaAtStation / (2.0 * definition.width);

 var origin = worldToPlane(sketchPlane, samplePos);
 var tangentPoint = worldToPlane(sketchPlane, samplePos + sampleDir * millimeter);
 var tangent = tangentPoint - origin;
 var tangentMag = _vectorMagnitude(tangent);

 var unitPerp;
 if (tangentMag > 1e-9 * meter)
 {
 unitPerp = vector(-tangent[1], tangent[0]) / tangentMag;
 lastValidUnitPerp = unitPerp;
 }
 else if (lastValidUnitPerp != undefined)
 {
 unitPerp = lastValidUnitPerp;
 }
 else
 {
 var planeX = sketchPlane.xAxis;
 var planeY = sketchPlane.yAxis;
 var xLen = _vectorMagnitude(planeX);
 var yLen = _vectorMagnitude(planeY);
 unitPerp = xLen >= yLen ? (planeX / xLen) : (planeY / yLen);
 }

 leftOffset = append(leftOffset, origin + unitPerp * halfHeight);
 rightOffset = append(rightOffset, origin - unitPerp * halfHeight);
 }

 if (size(leftOffset) < 2)
 {
 throw regenError("Not enough valid points — check centreline selection.");
 }

 // Build profile sketch — walls as polyline or smooth spline
 var sketchId = id + "profile";
 var sketch = newSketchOnPlane(context, sketchId, { "sketchPlane" : sketchPlane });

 if (definition.smoothWalls)
 {
 // Smooth walls: fit a spline through all wall points (no corner aliasing)
 skFitSpline(sketch, "leftWall", { "points" : leftOffset });
 skFitSpline(sketch, "rightWall", { "points" : rightOffset });
 }
 else
 {
 // Sharp walls: line segments for exact geometry at centreline corners
 for (var s = 0; s < size(leftOffset) - 1; s += 1)
 {
 skLineSegment(sketch, "_l" ~ s, { "start" : leftOffset[s], "end" : leftOffset[s + 1] });
 skLineSegment(sketch, "_r" ~ s, { "start" : rightOffset[s], "end" : rightOffset[s + 1] });
 }
 }
 // Throat and mouth edges are always straight — they close the profile
 skLineSegment(sketch, "throatEdge", { "start" : leftOffset[0], "end" : rightOffset[0] });
 skLineSegment(sketch, "mouthEdge", { "start" : leftOffset[size(leftOffset) - 1], "end" : rightOffset[size(rightOffset) - 1] });
 skSolve(sketch);



 // Build and store JSON
 var jsonStr = "";
 if (definition.outputJson)
 {
 jsonStr = _buildPyhornJson(
 throatArea, mouthArea, (definition.width / meter),
 uL, mL, profileType, derivedF12, curveLength,
 leftOffset, rightOffset, sampledLines, sketchPlane,
 definition.s1Cm2, definition.s2Cm2, definition.t
 );
 setFeatureComputedParameter(context, id, { "name" : "pyhorn_profile_json", "value" : jsonStr });
 }

 var hypCm = curveLength / centimeter;
 var infoMsg = "path=" ~ round(hypCm * 100.0) / 100.0 ~ "cm "
 ~ "S1=" ~ definition.s1Cm2 ~ "cm^2 "
 ~ "S2=" ~ definition.s2Cm2 ~ "cm^2 "
 ~ "T=" ~ definition.t ~ " "
 ~ "F12=" ~ round(derivedF12 * 100.0) / 100.0 ~ "Hz "
 ~ "profile=" ~ profileType;
 if (definition.outputJson)
 // {
 // infoMsg ~= " [" ~ size(jsonStr.len()) ~ " chars JSON stored as feature parameter]";
 // }
 reportFeatureInfo(context, id, infoMsg);

 }
 );
