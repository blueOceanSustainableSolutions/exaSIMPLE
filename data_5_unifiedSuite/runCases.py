# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 13:46:39 2023

@author: ALidtke
"""

import numpy as np
import re
import sys
import os
import shutil
import pandas
import subprocess
import pathlib
from xml.etree import ElementTree

def setElem(elem, txt, attrs=None):
    elem.text = txt
    if attrs is not None:
        for attr_key, attr_value in attrs.items():
            elem.set(attr_key, attr_value)
    return elem

def update_nested_xml(root, tag, updates):
    """
    Update a parent XML element and its nested elements based on a dictionary of values.
    Optionally rename the parent element.
    
    :param root: The root element of the XML.
    :param tag: The current tag name of the parent element to update.
    :param updates: Dictionary with tag names as keys and (text, attributes) tuples as values.
                    The parent tag updates should be included under the key 'self'.
    """
    parent = root.find(f".//{tag}")
    
    if parent is not None:
        parent.tag = updates["bcType"]

        # Update the nested elements
        for subtag in updates:
            if subtag == 'bcType':
                continue
            text, attrs = updates[subtag]
            element = parent.find(subtag)
            if element is None:
                element = ElementTree.SubElement(parent, subtag)
            setElem(element, text, attrs)

# ===
runDir = "calcs"
nCoresPerNode = 20
maxNodesPerJob = 5

# Global for all cases - tolerance needs to be small enough for the required no. outerloops to be executed.
maxNoOuterLoops = 50
LinfTol = 1e-18

grids = pandas.read_csv("gridStats.csv")

caseDefinitions = {
    "LDCF": {
        "BCs": {
            "x_neg": {"bcType": "BCWall"},
            "x_pos": {"bcType": "BCWall"},
            "y_neg": {"bcType": "BCWall"},
            "y_pos": {"bcType": "BCWall", "velocity": ("1 0 0", {"userCode": "false"})},
       },
       "equations/equation/EQPressure/pressureReferencePosition": "0.5 0.99999 0.5",
    },
    "convDiff": {
        "BCs": {
            "x_neg": {"bcType": "BCInflow", "velocity": ("0.705 0 0", {"userCode": "false"})},
            "x_pos": {"bcType": "BCPressure"},
            "y_neg": {"bcType": "BCInflow", "velocity": ("0 0.705 0", {"userCode": "false"})},
            "y_pos": {"bcType": "BCPressure"},
       },
    },
    "channel": {
        "BCs": {
            "x_neg": {"bcType": "BCInflow", "velocity": ("1 0 0", {"userCode": "false"})},
            "x_pos": {"bcType": "BCPressure"},
            "y_neg": {"bcType": "BCWall"},
            "y_pos": {"bcType": "BCWall"},
       },
       "materials/material/fluid/viscosityMolecular": "2e-2",
    },
    "plate": {
        "BCs": {
            "x_neg": {"bcType": "BCInflow", "velocity": ("1 0 0", {"userCode": "false"})},
            "x_pos": {"bcType": "BCOutflow"},
            "y_neg": {"bcType": "BCWall"},
            "y_pos": {"bcType": "BCPressure"},
       },
       "materials/material/fluid/viscosityMolecular": "1e-2",
    },
    "Poiseuille": {
        "gridNameRegex": "2.0x1.0x1.0",
        "BCs": {
            "x_neg": {"bcType": "BCInflow", "velocity": ("1 0 0", {"userCode": "true"}), "extrapolationOrder": ("1", {})},
            "x_pos": {"bcType": "BCPressure", "pressure": ("1.0", {})},
            "y_neg": {"bcType": "BCWall"},
            "y_pos": {"bcType": "BCWall"},
       },
       "equations/equation/EQMomentum/initialization/USER_DEFINED/initialVelocity": ("1 0 0", {"userCode": "true"}),
       "equations/equation/EQPressure/initialPressure": ("1.0", {"userCode": "true"}),
    },
    "TaylorVortex": {
        "gridNameRegex": "2.0x2.0x1.0",
        "BCs": {
            "x_neg": {"bcType": "BCInflow", "velocity": ("0 0 0", {"userCode": "true"}), "checkFlux": ("false", {})},
            "x_pos": {"bcType": "BCInflow", "velocity": ("0 0 0", {"userCode": "true"}), "checkFlux": ("false", {})},
            "y_neg": {"bcType": "BCInflow", "velocity": ("0 0 0", {"userCode": "true"}), "checkFlux": ("false", {})},
            "y_pos": {"bcType": "BCInflow", "velocity": ("0 0 0", {"userCode": "true"}), "checkFlux": ("false", {})},
       },
       "bodyForces/apply": "true",
       "equations/equation/EQMomentum/initialization/USER_DEFINED/initialVelocity": ("0 0 0", {"userCode": "true"}),
       "equations/equation/EQPressure/initialPressure": ("0.0", {"userCode": "true"}),
    },
}

specialGridSettings = {
    "3D_Triangle_Delaunay": {
        "equations/equation/EQMomentum/applyEccentricityCorrection": "false",
        "equations/equation/EQPressure/applyEccentricityCorrection": "false",    
    }
}

subprocess.run("rm -f jobNames.txt && touch jobNames.txt", shell=True)

for caseName in caseDefinitions:

    if os.path.isdir("./src/src_"+caseName):
        print("Compiling user code for:", caseName)
        subprocess.run("./compile.sh > log.compile", cwd="./src/src_"+caseName, shell=True)

    caseDefinition = caseDefinitions[caseName].copy()

    # Loop over all the grids.
    for iGrid in grids.index:
    
        # TODO temp
        #if (grids.loc[iGrid, "nCells"] > 200e3) or ("3D" in grids.loc[iGrid, "grid"]):
        if ("3D" in grids.loc[iGrid, "grid"]):
            continue

        if "gridNameRegex" in caseDefinition:
            if len(re.findall(caseDefinition["gridNameRegex"], grids.loc[iGrid, "grid"])) == 0:
                continue

        currentCase = os.path.join(runDir, "case_{:s}_grid_{:d}".format(caseName, iGrid))
        print("Preparing:", currentCase, grids.loc[iGrid, "grid"])

        try:
            # Prepare the case directory.
            os.makedirs(currentCase, exist_ok=False)

            # Modify the controls.
            controls = ElementTree.parse("controls_template.xml")
            root = controls.getroot()
            root.findall("./grids/grid/gridFileName")[0].text = grids.loc[iGrid, "grid"]+"_refresco"
            root.findall("./grids/grid/gridFilePath")[0].text = "../../grids_refresco"
            root.findall("./outerLoop/maxIteration")[0].text = "{:d}".format(maxNoOuterLoops)
            root.findall("./outerLoop/convergenceToleranceLinf")[0].text = "{:.6e}".format(LinfTol)
            
            if "3D" in grids.loc[iGrid, "grid"]:
                root.findall("./equations/equation/EQMomentum/solve_z")[0].text = "true"
            
            # Special settings for some "special" grids
            for g in specialGridSettings:
                if g in grids.loc[iGrid, "grid"]:
                    caseDefinition.update(specialGridSettings[g])

            for bcName in caseDefinition["BCs"]:
                # Find the specific <family> element of type BCSymmetryPlane by name attribute
                update_nested_xml(root.find(".//family[@name='{:s}']".format(bcName)),
                    "BCSymmetryPlane", caseDefinition["BCs"][bcName])

            # Replace other tags
            for tag in caseDefinition:
                # Skip special parts.
                if tag in ["BCs", "gridNameRegex"]:
                    continue
                element = root.find(tag)
                # New element - create and add
                if element is None:
                    element = root.find("/".join(tag.split("/")[:-1]))
                    newElement = ElementTree.Element(tag.split("/")[-1])
                    txt = caseDefinition[tag]
                    setElem(newElement, txt)
                    element.append(newElement)
                # Existing element - replace value
                else:
                    if type(caseDefinition[tag]) == str:
                        element.text = caseDefinition[tag]
                    elif type(caseDefinition[tag]) == tuple:
                        txt, attrs = caseDefinition[tag]
                        setElem(element, txt, attrs=attrs)
                    else:
                        raise ValueError

            controls.write(os.path.join(currentCase, "controls.xml"))

            # Copy the user code.
            if os.path.isdir("./src/src_"+caseName):
                shutil.copytree("./src/src_"+caseName+"/usercode", currentCase+"/usercode")

            # Prepare the jobfile.
            shutil.copy("job.sh", currentCase)
            nCores = int(np.ceil(grids.loc[iGrid, "nCells"] / 10e3))
            nNodes = min(max(int(nCores / nCoresPerNode), 1), maxNodesPerJob)
            with open(os.path.join(currentCase, "job.sh"), "r") as infile:
                s = infile.read()
            s = re.sub("#SBATCH --nodes=.*", "#SBATCH --nodes={:d}".format(nNodes), s)
            s = re.sub("\$SLURM_NTASKS", "{:d}".format(nCores), s)
            with open(os.path.join(currentCase, "job.sh"), "w") as outfile:
                outfile.write(s)

            # Submit the case.
            subprocess.run("echo >> jobNames.txt", shell=True)
            subprocess.run("echo {:s} {:s} : {:s} >> jobNames.txt".format(caseName, grids.loc[iGrid, "grid"], currentCase), shell=True)
            subprocess.run("sbatch job.sh >> ../../jobNames.txt", cwd=currentCase, shell=True)

        except FileExistsError:
            print("  Case dir exists! Remove it before running the case again.")
        
        #break

