# exaSIMPLE
Repository to be used for the exaSIMPLE project

## Meeting - 19/03/2024
`People present:` Artur Lidtke (MARIN); João Muralha (blueOASIS)
`Topic:` Initial talks about SIMPLE prototype

`Brainstorming:`
- Possible use of FiPY (JM) (https://www.ctcms.nist.gov/fipy/index.html)
- Adaptation of Artur OpenFOAM tutorials (https://github.com/UnnamedMoose/BasicOpenFOAMProgrammingTutorials/blob/master/OFtutorial14_SIMPLE_algorithm/OFtutorial14.C) to python:
    1 - Start it simple only equally cartesian grids (simplifies interpolations)
    2 - Increase complexity of interpolations and adaptations to handle more complex grids - explore whether or not it's possible to export interpolation weights from ReFRESCO and read them in the same way we plan to handle matrices. This would allow us to handle grids of arbitrary complexity without the need to make the prototype complex.
- Possible start Python script that only receives the pressure equation matrices from OpenFOAM or ReFRESCO and solves them. The necessary part on ReFRESCO and Python side have already been implemented, we could start on this right away.
    1 - explore the concept explored by [Weymouth](https://www.sciencedirect.com/science/article/pii/S0045793022002213)? Artur has got his code running (in Julia) so we can trace the steps back exactly and even reuse the same data. We don't have a GAMG solver in Python/C/Fortran though so would need to adapt the implementation from [here](https://github.com/weymouth/GeometricMultigrid.jl)
    2 - start thinking about the use of graph NNs for this?

Extrating matrices from ReFRESCO to python: Artur can easily do that and has it set-up

Artur also run some ML examples that can be used by Mariana as examples and possible test cases.

TODO:
- Create github repository (JM)
- Add already available ML data (AL) and ReFRESCO matrices to start ML work


