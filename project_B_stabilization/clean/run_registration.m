function zproj_mean = run_registration(cfg)
% RUN_REGISTRATION  Config-driven entry point for the Shipley-2020 z-stack
% stabilization pipeline (cleaned, self-contained port).
%
%   zproj_mean = run_registration(cfg)
%
% This replaces the original hardcoded RegistrationMasterPipeline.m script.
% Instead of resolving data locations from a machine-specific
% hostname -> Windows-drive table (pipe.lab.pathbase/datedir/datapath/rundir),
% you pass explicit paths in CFG. The numerical stages (optotune warp, DFT 3D
% registration, shift application, z-projection) are byte-for-byte the original
% functions and are NOT modified here.
%
% REQUIRED cfg fields
%   cfg.sbx_path   full path to the raw .sbx (its sidecar <name>.mat must sit
%                  next to it). Leave '' to convert from a FluoView TIFF export
%                  (then cfg.tif_frames_dir / fbase / mouse / date / run apply).
%
% OPTIONAL cfg fields (defaults in brackets)
%   cfg.fdir         directory holding the data         [fileparts(sbx_path)]
%   cfg.fbase        base name of the <fbase>.tif.frames folder        ['']
%   cfg.out_base     output path stem (no extension)
%                    -> <out_base>.dftshifts and <out_base>_<proj>_zproj.tif
%                    [sbx_path with .sbx stripped]
%   cfg.refchannel   1 = red vessels, 2 = green          [1]
%   cfg.opttype      'none' (piezo) | 'affine' | 'rigid' [ 'none' ]
%   cfg.scale        spatial downsample factor for shift estimation [4]
%   cfg.chunksize    volumes per registration chunk (<= 20)          [20]
%   cfg.proj_type    'mean' | 'max' | 'median'           ['mean']
%   cfg.fiji_home    path to Fiji.app (sets the Miji preference)     [bundle]
%   cfg.mouse/date/run   only needed when converting from TIFFs (naming)
%
% Returns the registered z-projection (also written to <out_base>_<proj>_zproj.tif).

    %% ---- defaults -------------------------------------------------------
    if nargin < 1, error('run_registration:cfg','Pass a config struct. See config_example.m'); end
    cfg = with_default(cfg, 'refchannel', 1);
    cfg = with_default(cfg, 'opttype',    'none');
    cfg = with_default(cfg, 'scale',      4);
    cfg = with_default(cfg, 'chunksize',  20);
    cfg = with_default(cfg, 'proj_type',  'mean');
    cfg = with_default(cfg, 'sbx_path',   '');
    cfg = with_default(cfg, 'fbase',      '');

    %% ---- put this pipeline on the MATLAB path ---------------------------
    here = fileparts(mfilename('fullpath'));        % the clean/ folder
    addpath(here, ...                               % parent of +pipe package
            fullfile(here, 'registration'), ...
            fullfile(here, 'data'));

    %% ---- Fiji location --------------------------------------------------
    % Only actually loaded when ImageJ is touched (final TIFF write; optotune).
    if isfield(cfg, 'fiji_home') && ~isempty(cfg.fiji_home)
        setpref('shipley_clean', 'fiji_home', cfg.fiji_home);
    end

    %% ---- resolve paths (replaces the pipe.lab.* chain) ------------------
    if isempty(cfg.sbx_path) && ~isfield(cfg, 'fdir')
        error('run_registration:paths', 'Provide cfg.sbx_path (or cfg.fdir for a fresh conversion).');
    end
    if ~isfield(cfg, 'fdir') || isempty(cfg.fdir)
        cfg.fdir = fileparts(cfg.sbx_path);
    end
    fdir = cfg.fdir;
    path = cfg.sbx_path;                            % '' triggers conversion below

    %% ---- dimensions -----------------------------------------------------
    [Nchan, Nx, Ny, Nz, Nt] = GetDimensions(path, fdir, cfg.fbase); %#ok<ASGLU>

    Nchunks = round(Nt / cfg.chunksize);

    %% ---- (optional) convert a FluoView TIFF export to .sbx --------------
    if isempty(path)
        require_fields(cfg, {'mouse','date','run'}, 'TIFF->SBX conversion');
        ConvertOIR_SBX(cfg.mouse, cfg.date, cfg.run, fdir, cfg.fbase, ...
                       Nx, Ny, Nz, Nt, Nchan, 'lineshift', true);
        path = fullfile(fdir, sprintf('%s_%s_%03d.sbx', cfg.mouse, cfg.date, cfg.run));
    end

    %% ---- output stems ---------------------------------------------------
    if ~isfield(cfg, 'out_base') || isempty(cfg.out_base)
        [b, n] = fileparts(path);                   % strip .sbx
        cfg.out_base = fullfile(b, n);
    end
    shiftpath = [cfg.out_base '.dftshifts'];
    savepath  = [cfg.out_base '_' cfg.proj_type '_zproj.tif'];

    %% ---- pipeline (numerics unchanged from Shipley 2020) ----------------
    timerval = tic;

    tforms_optotune = CalculateOptotuneWarp(path, cfg.refchannel, cfg.scale, ...
                                            'regtype', cfg.opttype, 'save', 'true');

    DFT_warp_3D_2(path, shiftpath, cfg.refchannel, cfg.scale, Nchunks, ...
                  tforms_optotune, 'reftype', 'mean');

    zproj_mean = MakeSBXall(path, shiftpath, 'refchannel', cfg.refchannel);

    write2chanTiff(uint16(zproj_mean), savepath);

    fprintf('run_registration: wrote %s  (%.1f min)\n', savepath, toc(timerval)/60);
end

% ----------------------------------------------------------------------------
function s = with_default(s, field, val)
    if ~isfield(s, field) || isempty(s.(field)), s.(field) = val; end
end

function require_fields(s, fields, why)
    for i = 1:numel(fields)
        if ~isfield(s, fields{i}) || isempty(s.(fields{i}))
            error('run_registration:missing', 'cfg.%s is required for %s.', fields{i}, why);
        end
    end
end
