function tform_cum_ordered = MultiStackReg_Fiji(vol, fdir, Nz, mode)
% MULTISTACKREG_FIJI  Per-slice optotune registration via Fiji's MultiStackReg.
%
%   tforms = MultiStackReg_Fiji(vol, fdir, Nz, mode)
%
%   Drives the (headless) ImageJ MultiStackReg plugin to register each z-slice
%   of MEAN reference volume VOL to the brightest slice, then reconstructs the
%   cumulative per-slice affine transforms in MATLAB. Writes the plugin's
%   transform file into FDIR.
%
%   mode = 'affine' (optotune lens) | 'rigid' (rigid body).
%
%   This replaces the three near-identical Shipley-2020 wrappers
%   (MultiStackReg_Fiji_affine, _affine_2, _rigid). It is always headless
%   (Miji(false)); classpath/jars are set up by Miji.m (see FIJI_HOME there),
%   so there is no per-file javaaddpath. Only reached when opttype is 'affine'
%   or 'rigid' (i.e. optotune acquisitions) — dead on the piezo/'none' path.

    if nargin < 4, mode = 'affine'; end

    switch lower(mode)
        case 'rigid'
            txtname        = 'TransformationMatricesRigid.txt';
            transformation = '[Rigid Body]';
            geomodel       = 'nonreflectivesimilarity';
        case 'affine'
            txtname        = 'TransformationMatricesAffine.txt';
            transformation = 'Affine';
            geomodel       = 'affine';
        otherwise
            error('MultiStackReg_Fiji:mode', ...
                  'mode must be ''affine'' or ''rigid'' (got ''%s'').', mode);
    end

    txtpath = fullfile(fdir, txtname);

    % Anchor on the brightest slice (max summed intensity).
    B = sum(reshape(vol, [], Nz), 1);
    [~, I] = max(B);

    % --- ImageJ round trip: align stack, save the transform matrices to disk ---
    Miji(false);
    MIJ.createImage(vol);
    MIJ.setSlice(I);
    MIJ.run('MultiStackReg', ...
        ['stack_1=[Import from Matlab] action_1=Align file_1=', txtpath, ...
         ' stack_2=None action_2=Ignore file_2=[] transformation=', transformation, ' save']);
    MIJ.run('Close');
    MIJ.exit;

    % --- Reconstruct per-slice transforms from the plugin's output file ---
    transforms = LoadTransforms(txtpath);

    M = repmat([true; true; true; false; false; false], Nz - 1, 2);
    movingPoints = reshape(transforms(M), [], 2);

    % step-wise (NOT cumulative) transform per z-step
    idx = 1:3:size(movingPoints, 1);
    fixedPoints = transforms(4:6, :);
    tform(1) = fitgeotrans(fixedPoints, fixedPoints, geomodel);
    for i = 1:length(idx)
        A = movingPoints(idx(i):idx(i) + 2, :);
        tform(i + 1) = fitgeotrans(A, fixedPoints, geomodel);
    end

    tform_ordered = [fliplr(tform(1:I)), tform(I + 1:end)];

    % invert the transforms for slices BEFORE the reference slice
    for i = 1:I
        tform_ordered(i) = invert(tform_ordered(i));
    end

    % accumulate into cumulative affine transforms
    M_cum = eye(3);
    for i = 1:length(tform_ordered)
        M_cum = M_cum * tform_ordered(i).T;
        tform_cum_ordered(i) = affine2d(M_cum);
    end
end
