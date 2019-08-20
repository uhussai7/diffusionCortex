import dataWhere
import nibabel


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

    return(nibabel.freesurfer.io.read_geometry(surface_path))