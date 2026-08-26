function result = run_benchmark(cfg)
% RUN_BENCHMARK  Timed one-shot run of the ORIGINAL MATLAB pipeline.
%
%   result = run_benchmark(cfg)
%
% NOTE ON LANGUAGE: this file is deliberately pure ASCII. MATLAB releases
% before R2020a read .m files in the platform's default encoding (GBK or
% windows-1252 on Windows), so UTF-8 comments and, worse, UTF-8 text inside
% fprintf/error strings come out as mojibake on the lab machine. The Chinese
% instructions live in README.md, which is read by an editor, not by MATLAB.
%
% What this does -- three things, none of them numerical:
%   1. puts clean/ on the MATLAB path (run_registration.m then adds its own
%      +pipe / registration / data subfolders);
%   2. checks the environment BEFORE starting -- MATLAB release, Image
%      Processing Toolbox, Parallel Computing Toolbox, Java heap, Fiji/MIJ --
%      so the traps that would waste hours surface in two seconds
%      (see WHY PRECHECK below);
%   3. wraps run_registration in tic/toc, prints the elapsed time and the
%      output paths, and keeps the whole console session in a diary log.
%
% The numerics are untouched: run_registration and every registration /
% projection function under clean/ are exactly as shipped.
%
% cfg fields (all optional; default in brackets)
%   cfg.sbx_path      raw .sbx; its <name>.mat sidecar must sit next to it
%                     [the single .sbx found in <this folder>/data]
%   cfg.refchannel    1 = red vessel channel, 2 = green            [1]
%   cfg.scale         spatial downsample factor for shift estimation [4]
%   cfg.chunksize     volumes per registration chunk (<= 20)       [20]
%   cfg.proj_type     'mean' | 'max' | 'median'                    ['mean']
%   cfg.opttype       'none' (piezo) | 'affine' | 'rigid'          ['none']
%   cfg.out_base      output stem, no extension        [sbx_path minus .sbx]
%   cfg.fiji_home     path to Fiji.app              [clean/fiji/Fiji.app]
%   cfg.require_fiji  abort when the precheck fails rather than run  [true]
%   cfg.label         one free-text line recorded in the log         ['']
%
% Returns a struct: timings, output paths and sizes, environment.
%
% WHY PRECHECK
% ------------
% The last step of run_registration is write2chanTiff, which goes through
% Fiji/MIJ: MIJ.createImage pushes the WHOLE projection (Nchan*Nx*Ny*Nt
% uint16) into the JVM heap. At the production Nt=1500 that is ~1.5 GiB,
% while MATLAB's default Java heap is a few hundred MB. That step is at the
% very END of the pipeline, and zproj_mean exists only as the function's
% return value -- so an OutOfMemoryError there throws away every hour of
% computation that preceded it. Two seconds of checking is cheap.
%
% Example:
%   cfg = struct();
%   cfg.sbx_path = 'D:\shipley\FAD-F_1_raw.sbx';
%   cfg.label    = 'Dell T2 / Ultra 9 285 / 24 cores';
%   r = run_benchmark(cfg);

    if nargin < 1 || isempty(cfg), cfg = struct(); end

    here = fileparts(mfilename('fullpath'));        % .../pyport/matlab_bench
    clean_dir = fullfile(here, '..', '..', 'clean');
    if exist(fullfile(clean_dir, 'run_registration.m'), 'file') ~= 2
        error('run_benchmark:clean', ...
              ['Cannot find %s/run_registration.m.\n' ...
               'This script assumes the layout ' ...
               'project_B_stabilization/{clean, pyport/matlab_bench}.\n' ...
               'If you copied only matlab_bench, copy clean/ too and keep ' ...
               'the same relative position.'], clean_dir);
    end
    % run_registration adds these itself, but this file calls GetDimensions
    % (which lives in clean/data) BEFORE run_registration for the precheck --
    % so add the same three entries here. clean_dir itself is the parent of
    % the +pipe package and must be on the path for pipe.io.* to resolve.
    addpath(clean_dir, ...
            fullfile(clean_dir, 'registration'), ...
            fullfile(clean_dir, 'data'));

    cfg = with_default(cfg, 'refchannel',   1);
    cfg = with_default(cfg, 'scale',        4);
    cfg = with_default(cfg, 'chunksize',    20);
    cfg = with_default(cfg, 'proj_type',    'mean');
    cfg = with_default(cfg, 'opttype',      'none');
    cfg = with_default(cfg, 'require_fiji', true);
    cfg = with_default(cfg, 'label',        '');
    if ~isfield(cfg, 'sbx_path') || isempty(cfg.sbx_path)
        cfg.sbx_path = find_only_sbx(fullfile(here, 'data'));
    end
    if exist(cfg.sbx_path, 'file') ~= 2
        error('run_benchmark:sbx', 'No such .sbx: %s', cfg.sbx_path);
    end
    [sbx_dir, sbx_name] = fileparts(cfg.sbx_path);
    mat_path = fullfile(sbx_dir, [sbx_name '.mat']);
    if exist(mat_path, 'file') ~= 2
        error('run_benchmark:sidecar', ...
              ['Missing sidecar %s.\nThe .sbx and its same-named .mat must ' ...
               'live in the SAME folder -- pipe.io.sbxInfo finds the ' ...
               'geometry that way, and without it Nx/Ny/Nz/Nt cannot be ' ...
               'read at all. make_sbx.py writes both; do not copy just ' ...
               'one of them.'], mat_path);
    end
    if ~isfield(cfg, 'out_base') || isempty(cfg.out_base)
        cfg.out_base = fullfile(sbx_dir, sbx_name);
    end
    if isfield(cfg, 'fiji_home') && ~isempty(cfg.fiji_home)
        fiji_home = cfg.fiji_home;
    else
        fiji_home = fullfile(clean_dir, 'fiji', 'Fiji.app');
    end

    logfile = [cfg.out_base '_matlab_bench_' ...
               datestr(now, 'yyyymmdd_HHMMSS') '.log'];
    diary(logfile); diary on;
    cleanup = onCleanup(@() diary('off'));  %#ok<NASGU> % survive Ctrl-C

    fprintf('\n===========================================================\n');
    fprintf('MATLAB benchmark: Shipley stabilization, original pipeline\n');
    fprintf('  when      : %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
    if ~isempty(cfg.label), fprintf('  label     : %s\n', cfg.label); end
    fprintf('  input     : %s\n', cfg.sbx_path);
    fprintf('  sidecar   : %s\n', mat_path);
    fprintf('  out stem  : %s\n', cfg.out_base);
    fprintf('  log       : %s\n', logfile);
    fprintf(['  params    : refchannel=%d scale=%d chunksize=%d ' ...
             'proj=%s opttype=%s\n'], cfg.refchannel, cfg.scale, ...
            cfg.chunksize, cfg.proj_type, cfg.opttype);
    fprintf('===========================================================\n');

    env = report_environment(fiji_home);

    % ---- dimensions + precheck ------------------------------------------
    [Nchan, Nx, Ny, Nz, Nt] = GetDimensions(cfg.sbx_path, sbx_dir, '');
    fprintf('\nDimensions: Nchan=%d Nx(rows)=%d Ny(cols)=%d Nz=%d Nt=%d\n', ...
            Nchan, Nx, Ny, Nz, Nt);
    d = dir(cfg.sbx_path);
    fprintf('  .sbx size : %.1f GiB, %d records (Nz*Nt = %d)\n', ...
            d(1).bytes / 2^30, ...
            floor(d(1).bytes / (Nx * Ny * 2 * Nchan)), Nz * Nt);
    fprintf('  Nchunks   = round(Nt/chunksize) = %d\n', ...
            round(Nt / cfg.chunksize));

    zproj_bytes = Nchan * Nx * Ny * Nt * 2;   % what MIJ.createImage must hold
    fprintf(['  projection: %d x %d x %d x %d uint16 = %.2f GiB ' ...
             '(goes into the JVM heap)\n'], Nchan, Nx, Ny, Nt, ...
            zproj_bytes / 2^30);
    % MakeSBXall.m L137-193 writes a SECOND full-size copy of the raw data --
    % Nchan*Nx*Ny*Nz*Nt uint16, 60.1 GiB at production Nt -- and RegWriter.m
    % L65/L86 throw away fwrite's return count (no ferror anywhere in the
    % repo), so a full disk truncates it silently and the run only dies hours
    % later at write2chanTiff, with zproj_mean living nowhere but a return
    % value. That is the same "hours wasted" failure this precheck exists to
    % catch, and it costs the same two seconds as the heap check.
    sbxall_bytes = Nchan * Nx * Ny * Nz * Nt * 2;
    fprintf(['  .sbxall   : %d x %d x %d x %d x %d uint16 = %.2f GiB ' ...
             '(written NEXT TO the .sbx, not to out_base)\n'], ...
            Nchan, Nx, Ny, Nz, Nt, sbxall_bytes / 2^30);
    ok = precheck(env, zproj_bytes, fiji_home, sbxall_bytes, sbx_dir);
    if ~ok
        if cfg.require_fiji
            error('run_benchmark:precheck', ...
                  ['Precheck failed (see the [FAIL] lines above). Fix it, ' ...
                   'or set cfg.require_fiji=false to run anyway.\nBeware: ' ...
                   'a failed TIFF write at the end discards the whole run ' ...
                   '-- zproj lives only in the return value.']);
        end
        fprintf('\n[WARN] precheck failed but require_fiji=false; going on.\n');
    end

    % ---- timed section ---------------------------------------------------
    rcfg = struct();
    rcfg.sbx_path   = cfg.sbx_path;
    rcfg.fdir       = sbx_dir;
    rcfg.out_base   = cfg.out_base;
    rcfg.refchannel = cfg.refchannel;
    rcfg.opttype    = cfg.opttype;
    rcfg.scale      = cfg.scale;
    rcfg.chunksize  = cfg.chunksize;
    rcfg.proj_type  = cfg.proj_type;
    if exist(fiji_home, 'dir') == 7, rcfg.fiji_home = fiji_home; end

    fprintf(['\nStarting run_registration ... on the full stack this is ' ...
             'hours; leave the window open.\n']);
    fprintf('-----------------------------------------------------------\n');
    t0 = tic;
    zproj_mean = run_registration(rcfg);   %#ok<NASGU>
    elapsed = toc(t0);
    fprintf('-----------------------------------------------------------\n');

    % ---- outputs ---------------------------------------------------------
    shiftpath = [cfg.out_base '.dftshifts'];
    savepath  = [cfg.out_base '_' cfg.proj_type '_zproj.tif'];
    % NOT [cfg.out_base '.sbxall']. MakeSBXall.m L137 hands the raw .sbx path
    % to pipe.io.RegWriter, and RegWriter.m L20-21 does
    % fileparts(path) + [name extension] -- so the .sbxall lands next to the
    % .sbx and cfg.out_base is never consulted. With a custom out_base the old
    % expression reported "(missing)" for a 60 GiB file that had in fact been
    % written to the input's drive, i.e. exactly the drive out_base was set to
    % avoid. (The Python port already carries this fix: cpstab/config.py
    % registered_stack_path(), PORTING NOTE #6 / review fix F3.)
    sbxall    = fullfile(sbx_dir, [sbx_name '.sbxall']);

    fprintf('\n===========================================================\n');
    fprintf('DONE. total %.1f s = %.2f min = %.3f h\n', ...
            elapsed, elapsed / 60, elapsed / 3600);
    fprintf('  per volume %.3f s  (Nt=%d)\n', elapsed / max(Nt, 1), Nt);
    fprintf('Outputs:\n');
    outs = {savepath, shiftpath, sbxall};
    for i = 1:numel(outs)
        report_file(outs{i});
    end
    fprintf('Log: %s\n', logfile);
    fprintf('===========================================================\n');

    result = struct();
    result.elapsed_s    = elapsed;
    result.elapsed_min  = elapsed / 60;
    result.per_volume_s = elapsed / max(Nt, 1);
    result.dims         = struct('Nchan', Nchan, 'Nx', Nx, 'Ny', Ny, ...
                                 'Nz', Nz, 'Nt', Nt);
    result.sbx_path     = cfg.sbx_path;
    result.zproj_path   = savepath;
    result.shift_path   = shiftpath;
    result.sbxall_path  = sbxall;
    result.logfile      = logfile;
    result.env          = env;
    result.label        = cfg.label;

    resfile = [cfg.out_base '_matlab_bench_result.mat'];
    save(resfile, 'result');
    fprintf('Result struct saved: %s\n', resfile);
    diary off;
end

% ---------------------------------------------------------------------------
function env = report_environment(fiji_home)
    env = struct();
    env.matlab   = version;
    env.release  = version('-release');
    env.computer = computer;
    env.host     = '';
    try
        env.host = char(java.net.InetAddress.getLocalHost.getHostName);
    catch
        env.host = getenv('COMPUTERNAME');
    end
    env.ncores      = feature('numcores');
    env.java_max_gb = double(java.lang.Runtime.getRuntime.maxMemory) / 2^30;
    env.has_ipt = license('test', 'image_toolbox') && ~isempty(ver('images'));
    env.has_pct = license('test', 'distrib_computing_toolbox') && ...
                  ~isempty(ver('parallel'));
    env.fiji_home = fiji_home;
    env.has_fiji  = exist(fiji_home, 'dir') == 7;
    env.has_mij   = exist(fullfile(fiji_home, 'jars', 'mij.jar'), 'file') == 2;

    fprintf('\nEnvironment:\n');
    fprintf('  MATLAB    : %s (R%s) on %s\n', ...
            env.matlab, env.release, env.computer);
    fprintf('  host/cores: %s / %d\n', env.host, env.ncores);
    fprintf(['  Java heap : %.2f GiB ' ...
             '(Preferences > General > Java Heap Memory)\n'], env.java_max_gb);
    fprintf('  Image Processing Toolbox  : %s\n', tf(env.has_ipt));
    fprintf('  Parallel Computing Toolbox: %s\n', tf(env.has_pct));
    fprintf('  Fiji.app  : %s  (%s)\n', tf(env.has_fiji), fiji_home);
    fprintf('  mij.jar   : %s\n', tf(env.has_mij));
end

% ---------------------------------------------------------------------------
function ok = precheck(env, zproj_bytes, fiji_home, sbxall_bytes, sbx_dir)
    ok = true;
    fprintf('\nPrecheck:\n');
    if ~env.has_ipt
        fprintf(['  [FAIL] no Image Processing Toolbox -- imresize / ' ...
                 'imtranslate / imwarp /\n         imgaussfilt / ' ...
                 'fitgeotrans all live there; the pipeline cannot take ' ...
                 'one step.\n']);
        ok = false;
    else
        fprintf('  [ OK ] Image Processing Toolbox\n');
    end
    if ~env.has_pct
        fprintf(['  [WARN] no Parallel Computing Toolbox: the parfor in ' ...
                 'DFT_warp_3D_2 degrades\n         to serial. It still ' ...
                 'finishes, but the registration timing is then not ' ...
                 'comparable to a parallel run.\n']);
    else
        fprintf('  [ OK ] Parallel Computing Toolbox (parfor available)\n');
    end
    if ~env.has_fiji || ~env.has_mij
        fprintf('  [FAIL] Fiji/MIJ incomplete (%s).\n', fiji_home);
        fprintf(['         write2chanTiff needs it for the final TIFF, so ' ...
                 'a missing jar blows up\n         AFTER the computation ' ...
                 '-- hours wasted.\n']);
        fprintf('         Fix: run clean/fiji/setup_fiji.sh, then put mij.jar in\n');
        fprintf('         %s\n', fullfile(fiji_home, 'jars'));
        ok = false;
    else
        fprintf('  [ OK ] Fiji + mij.jar\n');
    end
    need_gb = zproj_bytes / 2^30 * 2.5;   % createImage copies; leave headroom
    if env.java_max_gb < need_gb
        fprintf('  [FAIL] Java heap %.2f GiB < recommended %.2f GiB.\n', ...
                env.java_max_gb, need_gb);
        fprintf(['         MIJ.createImage must hold the whole %.2f GiB ' ...
                 'projection (plus a copy).\n'], zproj_bytes / 2^30);
        fprintf(['         Fix: Preferences > General > Java Heap Memory, ' ...
                 'then restart MATLAB.\n']);
        fprintf(['         If it cannot go that high, shrink Nt ' ...
                 '(make_sbx.py --limit-t) and run in parts.\n']);
        ok = false;
    else
        fprintf('  [ OK ] Java heap %.2f GiB >= recommended %.2f GiB\n', ...
                env.java_max_gb, need_gb);
    end
    % ---- free disk for the .sbxall (see the call site) --------------------
    if isempty(sbx_dir), sbx_dir = pwd; end
    free_b = NaN;
    try
        % double() is NOT decoration: getUsableSpace returns a Java long, i.e.
        % an int64 in MATLAB, and int64 poisons both tests below --
        % isnan(int64) errors outright, and int64/2^30 rounds to whole GiB.
        free_b = double(java.io.File(sbx_dir).getUsableSpace());
    catch
        % no JVM (matlab -nojvm): say so rather than pretend it passed
    end
    need_b = sbxall_bytes * 1.02;   % +2%: the .dftshifts and the log too
    if isnan(free_b) || free_b <= 0
        fprintf(['  [WARN] could not read free space on %s -- check by ' ...
                 'hand that it holds\n         %.1f GiB for the .sbxall ' ...
                 'MakeSBXall writes next to the .sbx.\n'], ...
                sbx_dir, sbxall_bytes / 2^30);
    elseif free_b < need_b
        fprintf('  [FAIL] free disk on %s: %.1f GiB < needed %.1f GiB.\n', ...
                sbx_dir, free_b / 2^30, need_b / 2^30);
        fprintf(['         MakeSBXall writes a full second copy of the raw ' ...
                 'data (.sbxall) into\n         THAT folder -- cfg.out_base ' ...
                 'does not move it. RegWriter ignores fwrite''s\n' ...
                 '         return value, so a full disk truncates it ' ...
                 'silently and the run only fails\n         hours later at ' ...
                 'the TIFF write, losing zproj_mean entirely.\n']);
        fprintf(['         Fix: free the space, move the .sbx+.mat pair to ' ...
                 'a roomier drive, or\n         shrink Nt ' ...
                 '(make_sbx.py --limit-t) and run in parts.\n']);
        ok = false;
    else
        fprintf(['  [ OK ] free disk on %s: %.1f GiB >= %.1f GiB needed ' ...
                 'for the .sbxall\n'], sbx_dir, free_b / 2^30, ...
                need_b / 2^30);
    end
end

% ---------------------------------------------------------------------------
function report_file(p)
    d = dir(p);
    if isempty(d)
        fprintf('  (missing) %s\n', p);
    else
        fprintf('  %8.2f GiB  %s\n', d(1).bytes / 2^30, p);
    end
end

function p = find_only_sbx(dirpath)
    d = dir(fullfile(dirpath, '*.sbx'));
    if numel(d) ~= 1
        error('run_benchmark:autofind', ...
              ['Found %d .sbx files in %s, cannot pick one. ' ...
               'Pass cfg.sbx_path explicitly.'], numel(d), dirpath);
    end
    p = fullfile(d(1).folder, d(1).name);
end

function s = with_default(s, field, val)
    if ~isfield(s, field) || isempty(s.(field))
        s.(field) = val;
    end
end

function s = tf(b)
    if b
        s = 'yes';
    else
        s = 'NO';
    end
end

% ===========================================================================
% DESIGN NOTES
% ===========================================================================
% 1. Why the precheck runs BEFORE tic, not after. The pipeline's only output
%    gate is write2chanTiff, on the last line. A missing Fiji or a too-small
%    Java heap therefore fails "after all 1500 volumes have been computed",
%    and zproj_mean exists only as run_registration's return value -- the
%    function never returns, so that memory goes with the stack. Hours of
%    compute for an OutOfMemoryError is the most expensive failure available
%    on this machine, so the check is not optional.
% 2. Why nothing under clean/ is modified to dodge note 1. The thing being
%    benchmarked has to BE the original pipeline; patch it and the timing
%    stops being comparable to the published one. The precheck is a fence
%    around it, not a change to it.
% 3. Why the diary spans everything, with onCleanup. The registration stage
%    reports progress with fprintf only -- nothing is written to a file. Once
%    the run is over, that diary is the only record able to answer "how much
%    of the wall clock was registration and how much was apply". onCleanup
%    guarantees the diary is closed on Ctrl-C or on an error, otherwise the
%    file stays locked and its tail may never reach disk.
% 4. Why a missing sidecar is raised here instead of letting GetDimensions
%    handle it. GetDimensions.m L10's catch branch falls through to looking
%    for a <fbase>.tif.frames directory and then fails with an error that
%    has nothing to do with the real cause. Checking here makes the message
%    point at the actual problem.
% 5. The --limit-t suggestion is serious. If the Java heap cannot reach ~4
%    GiB, do not gamble on the full stack: build a 300-volume .sbx with
%    make_sbx.py --limit-t 300, measure seconds per volume, and extrapolate.
%    The apply stage is per-volume independent and the registration stage is
%    per-chunk, so linear extrapolation is sound for both.
