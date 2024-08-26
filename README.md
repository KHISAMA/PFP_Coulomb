# Codes for "Molecular dynamics of liquid–electrode interface by integrating Coulomb interaction into universal neural network potential"

Kaoru Hisama

## Citation

Kaoru Hisama, Gerardo Valadez Huerta, and Michihisa Koyama, J. Comput. Chem. (2024).
https://doi.org/10.1002/jcc.27487

## requirement

- ASE, pfp_api_client (version 1.3.0 or later) 
- confirmed with ASE 3.22.1, python 3.8 environment in Matlantis

This code only work in Matlantis (https://matlantis.com/), the PFP hosting service.

Note that your "calculator" in your simulation should include "get_charge()" method which calculate the charge in each step, such as the PFP calculator.

## Usage

Please follow jupyter notebook files in the examples.

For NVT simulations, NVT_charge.ipynb is available.
For NPT simulations, NPT_charge.ipynb is available.

## Examples

- water_graphite: water-graphene interface

- water_GO: water-graphene oxide interface
