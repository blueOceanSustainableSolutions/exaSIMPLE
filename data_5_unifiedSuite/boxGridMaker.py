# -*- coding: utf-8 -*-
"""
Created on Fri Jun 03 09:00:49 2024

@author: ALidtke
"""
import pandas
import numpy as np
import os

try:
    from pointwise import GlyphClient
    from pointwise.glyphapi import *
except ImportError:
    print("Ooops, no Pointwise client!")

# ===
# Settings
gridPath = "./grids"

baseCellCountPerDim = 10
nGrids = 12
gridDims = ["3D"]#"2D", 
# NOTE: TirangleQuad and AdvancingFrontOrtho yield the same grid as Structured for a perfect box domain.
cellTypes = ["Structured", "Triangle", "TriangleQuad"]
unstrAlgorithms = {"2D": ["Delaunay", "AdvancingFront", "AdvancingFrontOrtho"], "3D": ["Delaunay"]}
boxDims = [(1., 1., 1.), (2., 1., 1.), (2., 2., 1.)]

# ===
# Open Glyph client connection.
glf = GlyphClient()
pw = glf.get_glyphapi()

# Loop over all the stuff.
gridStats = []
for gridDim in gridDims:
    for boxDim in boxDims:
        # Box dimensions.
        Lx, Ly, Lz = boxDim

        # Empirical refinement for "nice values".
        if gridDim == "3D":
            fac = 1.26
        else:
            fac = 1.41
        Ncs = [int(np.round(baseCellCountPerDim*fac**i)) for i in range(nGrids)]
        
        # Loop over each cell type.
        for cellType in cellTypes:
            # Loop over all unstructured algorithms - use a dummy for structured.
            if cellType == "Structured":
                algos = [""]
            else:
                algos = unstrAlgorithms[gridDim]
                
            for unstrAlgorithm in algos:
        
                # Loop over each refinement level.
                for iGrid, Nc in enumerate(Ncs):
                    gridType = cellType + "_" + unstrAlgorithm
                
                    gridName = "boxGrid_{:.1f}x{:.1f}x{:.1f}_{:s}_{:s}_Nc_{:d}".format(
                            Lx, Ly, Lz, gridDim, gridType, Nc).replace("__", "_")
                    
                    if os.path.isfile(os.path.join(gridPath, gridName+".cgns")):
                        continue
            
                    print(gridName)

                    # Remove previously generated entities.
                    pw.Application.reset()

                    # Create connectors.
                    pts = np.array([
                        [0., 0., 0.],
                        [1., 0., 0.],
                        [1., 1., 0.],
                        [0., 1., 0.],
                        [0., 0., 1.],
                        [1., 0., 1.],
                        [1., 1., 1.],
                        [0., 1., 1.],
                    ])

                    npPerEdge = [int(np.round(Nc*L)+1) for L in [
                        Lx, Ly, Lx, Ly, Lx, Ly, Lx, Ly, Lz, Lz, Lz, Lz]]

                    edges = [
                        # Front
                        [0, 1], # 0
                        [1, 2],
                        [2, 3],
                        [3, 0],
                        # Back
                        [4, 5], # 4
                        [5, 6],
                        [6, 7],
                        [7, 4],
                        # Spanwise
                        [0, 4], # 8
                        [1, 5],
                        [2, 6],
                        [3, 7],
                    ]

                    connectors = []
                    with pw.Application.begin("Create") as creator:
                        for i in range(len(edges)):
                            pw.Connector.setDefault("Dimension", npPerEdge[i])
                            if (gridDim == "2D") and (i == 4):
                                break
                            p0 = tuple(pts[edges[i][0]] * [Lx, Ly, Lz])
                            p1 = tuple(pts[edges[i][1]] * [Lx, Ly, Lz])
                            seg = pw.SegmentSpline()
                            seg.addPoint(p0)
                            seg.addPoint(p1)
                            con = pw.Connector()
                            con.addSegment(seg)
                            con.calculateDimension()
                            connectors.append(con)

                    # Create domains.
                    if cellType != "Structured":
                        pw.DomainUnstructured.setDefault("BoundaryDecay", 1.0)
                        pw.DomainUnstructured.setDefault("Algorithm", unstrAlgorithm)
                        pw.DomainUnstructured.setDefault("IsoCellType", cellType)

                    domains = []
                    with pw.Application.begin("Create") as creator:
                        if gridDim == "2D":
                            connectorLoops = [connectors]
                        else:
                            connectorLoops = [
                                [connectors[0], connectors[1], connectors[2], connectors[3]],
                                [connectors[0], connectors[8], connectors[4], connectors[9]],
                                [connectors[1], connectors[9], connectors[5], connectors[10]],
                                [connectors[2], connectors[10], connectors[6], connectors[11]],
                                [connectors[3], connectors[11], connectors[7], connectors[8]],
                                [connectors[4], connectors[5], connectors[6], connectors[7]],
                            ]

                        for cons in connectorLoops:
                            if cellType == "Structured":
                                dom = pw.DomainStructured()
                                for con in cons:
                                    edge = pw.Edge.createFromConnectors(con)
                                    dom.addEdge(edge)
                            else:
                                dom = pw.DomainUnstructured()
                                edge = pw.Edge.createFromConnectors(cons)
                                dom.addEdge(edge)

                            domains.append(dom)

                    # Extrude the domain for a 2D grid.
                    if gridDim == "2D":
                        dom = pw.GridEntity.getByName("dom-{:d}".format(1))
                        if cellType == "Structured":
                            block = pw.BlockStructured()
                            face = pw.FaceStructured.createFromDomains(dom, single=True)
                        else:
                            block = pw.BlockExtruded()
                            face = pw.FaceUnstructured.createFromDomains(dom, single=True)

                        block.addFace(face)
                        with pw.Application.begin("ExtrusionSolver", block) as extruder:
                            block.setExtrusionSolverAttribute("Mode", "Translate")
                            block.setExtrusionSolverAttribute("TranslateDirection", (0, 0, 1))
                            block.setExtrusionSolverAttribute("TranslateDistance", Lz)
                            extruder.run(1)

                    # Create from domains for a 3D grid.
                    else:
                        if cellType == "Structured":
                            block = pw.BlockStructured.createFromDomains(domains)[0]
                        else:
                            pw.BlockUnstructured.setDefault("BoundaryDecay", 1.0)
                            block = pw.BlockUnstructured.createFromDomains(domains)[0]
                            with pw.Application.begin("UnstructuredSolver", block) as solver:
                                solver.run("Initialize")

                    # Set BCs.
                    for i, name in enumerate(["z_neg", "y_neg", "x_pos", "y_pos", "x_neg", "z_pos"]):
                        bc = pw.BoundaryCondition()
                        bc.setName(name)
                        bc.setPhysicalType("General")
                        dom = pw.GridEntity.getByName("dom-{:d}".format(i+1))
                        bc.apply([[block, dom]])

                    # Save.
                    pw.Application.export(block, os.path.join(gridPath, gridName+".cgns"), precision="Double")

                    # Print cell count.
                    nCells = block.getCellCount()
                    glf.eval("puts {{  Finall cell count: {:,d}}}".format(nCells))
                    
                    # Store the stats.
                    gridStats.append({
                        "grid": gridName,
                        "nCells": nCells,
                        "baseCellCountPerUnitDim": Nc,
                        "cellType": cellType,
                        "unstructuredAlgorithm": unstrAlgorithm,
                        "Lx": Lx,
                        "Ly": Ly,
                        "Lz": Lz,
                    })

# Clean up.
glf.close()

# Save stats.
gridStats = pandas.DataFrame(gridStats)
gridStats.to_csv("gridStats.csv", index=False)

