% config_example.m
% Copy this, edit the paths, and run.  Two scenarios shown.
%
%   >> cfg = config_example();      % or paste the block below into the console
%   >> zproj = run_registration(cfg);

function cfg = config_example()

    % ---- Scenario A: re-run from an already-converted .sbx (the common case) ----
    cfg = struct();
    cfg.sbx_path   = '/path/to/data/0309-0721-009_200813_002.sbx';  % sidecar .mat sits next to it
    cfg.refchannel = 1;        % 1 = red vessels (IV dye), 2 = green
    cfg.opttype    = 'none';   % 'none' for piezo; 'affine'/'rigid' for optotune lens
    cfg.scale      = 4;
    cfg.chunksize  = 20;       % <= 20
    cfg.proj_type  = 'mean';   % 'mean' | 'max' | 'median'

    % Where to find Fiji.app. Omit to use the bundled clean/fiji/Fiji.app,
    % or set FIJI_HOME in the environment instead.
    % cfg.fiji_home = '/Applications/Fiji.app';

    % Outputs land next to the .sbx by default:
    %   <out_base>.dftshifts
    %   <out_base>_mean_zproj.tif
    % Override the stem with cfg.out_base if you want them elsewhere.
    % cfg.out_base = '/path/to/results/run002';

    % ---- Scenario B: first-time conversion from a FluoView TIFF export ----------
    % Leave sbx_path empty and point at the <fbase>.tif.frames folder's parent.
    %
    % cfg = struct();
    % cfg.sbx_path   = '';                       % triggers ConvertOIR_SBX
    % cfg.fdir       = '/path/to/data';          % holds <fbase>.tif.frames/
    % cfg.fbase      = '0309-0721-009_08132020_post_IP_LiCl';
    % cfg.mouse      = '0309-0721-009';
    % cfg.date       = '200813';                 % YYMMDD
    % cfg.run        = 2;
    % cfg.refchannel = 1;
    % cfg.opttype    = 'none';
end
