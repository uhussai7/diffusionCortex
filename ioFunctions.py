import dataWhere
import nibabel
import dipy
from pathlib import Path
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table

#from meshes import annot

def loadSurf(subject=None, hemi=None, surf=None,**kwargs):
    # subject is subject id
    # type is pial, white matter, etc
    # hemi is left or right hemisphere
    # arguments will be keywords that freesurfer uses
    if subject is None:
        raise ValueError("Please provide subject, subject=...")
    if hemi is None:
        raise ValueError("Please provide hemisphere (lh or rh), hemi=...")
    if surf is None:
        raise ValueError("Please provide surface (pial, etc), surf=...")

    subject_path = dataWhere.freesurfer_path / subject
    filename=hemi+"."+surf
    surface_path = subject_path / "surf" / filename
    print("loading: " + str(surface_path.resolve()))

    return nibabel.freesurfer.io.read_geometry(surface_path, read_metadata=True)


def loadAparc(subject=None, hemi=None,**kwargs):
    # subject is subject id
    # type is pial, white matter, etc
    # hemi is left or right hemisphere
    # arguments will be keywords that freesurfer uses
    if subject is None:
        raise ValueError("Please provide subject, subject=...")
    if hemi is None:
        raise ValueError("Please provide hemisphere (lh or rh), hemi=...")

    subject_path = dataWhere.freesurfer_path / subject
    filename=hemi+".aparc.annot"
    annot_path = subject_path / "label" / filename
    print("loading: " + str(annot_path.resolve()))

    return nibabel.freesurfer.io.read_annot(annot_path)

def loadMorph(subject=None, hemi=None,**kwargs):
    # subject is subject id
    # type is pial, white matter, etc
    # hemi is left or right hemisphere
    # arguments will be keywords that freesurfer uses
    if subject is None:
        raise ValueError("Please provide subject, subject=...")
    if hemi is None:
        raise ValueError("Please provide hemisphere (lh or rh), hemi=...")

    subject_path = dataWhere.freesurfer_path / subject
    filename=hemi+".curv"
    curv_path = subject_path / "surf" / filename
    print("loading: " + str(curv_path.resolve()))

    return nibabel.freesurfer.io.read_morph_data(curv_path)

def loadVol(filename=None, **kwargs):
    if filename is None:
        raise ValueError("Please provide filename, filename=...")
    print("loading: " + filename)
    return nibabel.load(filename)

def loadgetVol(filename=None, **kwargs):
    if filename is None:
        raise ValueError("Please provide filename, filename=...")
    print("loading: " + filename)
    temp=nibabel.load(filename)
    return temp.get_data()

def loadDiffVol(folder=None, **kwargs):
    if folder is None:
        raise ValueError("Please provide diffusion folder path, folder=...")

    folder=Path(folder)
    diffile =folder / "data.nii.gz"
    bvecsf = folder / "bvecs"
    bvalsf = folder / "bvals"
    print("loading " + str(diffile.resolve()))
    diff=nibabel.load(str(diffile.resolve()))
    print("loading " + str(bvecsf.resolve()))
    print("loading " + str(bvalsf.resolve()))
    bvals, bvecs = read_bvals_bvecs(str(bvalsf.resolve() ),str(bvecsf.resolve() ))
    gtab=gradient_table(bvals,bvecs)
    return (diff, gtab)

def loadMgz(subject=None, **kwards):
    #This will load T1 mgz file
    if subject is None:
        raise ValueError("Please provide subject, subject=...")

    subject_path = dataWhere.freesurfer_path / subject
    filename = "T1.mgz"
    mgz_path = subject_path / "mri" / filename
    print("loading: " + str(mgz_path.resolve()))
    return nibabel.load(str(mgz_path.resolve()))