import os
from argparse import ArgumentParser
from cgparam import CGParam

from .mod_topol import simplify

def run_cg_param(p):
    os.makedirs(p.path_tmp,exist_ok=True)
    if p.smiles is not None:
        os.system(f'cg-param-m3 -n {p.name} -s "{p.smiles}" --path_out {p.path_tmp}') # switch to non-api mode?

def run_calvatinize():

    parser = ArgumentParser()
    parser.add_argument('--name',required=True,type=str) # nargs='?',
    parser.add_argument('--smiles',type=str,required=True)
    # parser.add_argument('--path_in',type=str,default='.')
    parser.add_argument('--path_out',type=str,default='output')
    # parser.add_argument('--path_cgparam',type=str,
                        # default='/home/sobuelow/software/fork_cg_param_m3/cg_param_m3')
    parser.add_argument('--path_tmp',type=str,default='tmp_cgparam')
    parser.add_argument('--qtype',type=str,default='Qx')
    args = parser.parse_args()

    if (args.smiles is None):# and (args.sdf is None):
        raise

    run_cg_param(args)
    simplify(args.name,args.path_tmp,args.path_out,args.qtype)

if __name__ == '__main__':
    run_calvatinize()
