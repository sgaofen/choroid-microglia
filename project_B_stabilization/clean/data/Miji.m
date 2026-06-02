function [mij] = Miji(open_imagej)
    %% Set up the classpath to Fiji and optionally start MIJ.
    % Author: Jacques Pecreaux, Johannes Schindelin, Jean-Yves Tinevez
    % GNU Octave compatibility added by Eric Barnhill, Jul 2016
    %
    % Self-contained rework (Project B): the Fiji location is resolved from a
    % configurable FIJI_HOME instead of being assumed two folders up from this
    % file, and the per-version mij.jar hardcode has been removed. mij.jar is
    % expected to live inside <FIJI_HOME>/jars (it is picked up by the jars/
    % scan below), so no MATLAB-version-specific javaaddpath is needed.
    %
    % FIJI_HOME resolution order:
    %   1. environment variable  FIJI_HOME
    %   2. preference            getpref('shipley_clean','fiji_home')
    %   3. bundle default        <clean>/fiji/Fiji.app
    % run_registration.m sets the preference from cfg.fiji_home for you.

    if nargin < 1
        open_imagej = true;
    end

    %% Resolve the Fiji directory (configurable / bundle-relative)
    fiji_directory = resolve_fiji_home();

    %% Get the Java classpath
    classpath = javaclasspath('-all');

    %% Add all libraries in jars/ and plugins/ to the classpath
    warning_state = warning('off');                 % switch off warning
    add_to_classpath(classpath, fullfile(fiji_directory, 'jars'));
    add_to_classpath(classpath, fullfile(fiji_directory, 'plugins'));
    warning(warning_state)                          % restore warning state

    % Set the Fiji directory (and plugins.dir which is not Fiji.app/plugins/)
    javaMethod('setProperty', 'java.lang.System', 'ij.dir', fiji_directory);
    javaMethod('setProperty', 'java.lang.System', 'plugins.dir', fiji_directory);

    %% Maybe open the ImageJ window
    if open_imagej
        fprintf('\n\nUse MIJ.exit to end the session\n\n');
        mij = javaObject('MIJ');
        mij.start();
    else
        % initialize ImageJ with the NO_SHOW flag (== 2)
        ij.ImageJ([], 2);
    end
end

function home = resolve_fiji_home()
    home = getenv('FIJI_HOME');
    if isempty(home) && ispref('shipley_clean', 'fiji_home')
        home = getpref('shipley_clean', 'fiji_home');
    end
    if isempty(home)
        % default: <clean>/fiji/Fiji.app  (this file is at <clean>/data/Miji.m)
        clean_root = fileparts(fileparts(mfilename('fullpath')));
        home = fullfile(clean_root, 'fiji', 'Fiji.app');
    end
    if exist(home, 'dir') ~= 7
        error('Miji:noFiji', ...
            ['Fiji not found at "%s".\n' ...
             'Point FIJI_HOME at your Fiji.app (setenv(''FIJI_HOME'',path) or\n' ...
             'setpref(''shipley_clean'',''fiji_home'',path)), or run\n' ...
             'clean/fiji/setup_fiji.sh to bundle one under clean/fiji/.'], home);
    end
end

function add_to_classpath(classpath, directory)

    isoctave = exist('octave_config_info') > 0;

    % Get all .jar files in the directory
    dirData = dir(directory);
    dirIndex = [dirData.isdir];
    jarlist = dir(fullfile(directory,'*.jar'));
    path_= cell(0);
    for i = 1:length(jarlist)
      %disp(jarlist(i).name);
        if not_yet_in_classpath(classpath, jarlist(i).name)
            path_{length(path_) + 1} = fullfile(directory,jarlist(i).name);
        end
    end

    %% Add them to the classpath
    if ~isempty(path_)
      if isoctave
        for n = 1:numel(path_)
              err_code = javaMethod('addClassPath', 'org.octave.ClassHelper', path_{n});
              if err_code == 0
                  display(['Error importing ', path_{n}]);
              end
        end
      else
          javaaddpath(path_, '-end');
      end
     end

    %# Recurse over subdirectories
    subDirs = {dirData(dirIndex).name};
    validIndex = ~ismember(subDirs,{'.','..'});

    for iDir = find(validIndex)
      nextDir = fullfile(directory,subDirs{iDir});
      add_to_classpath(classpath, nextDir);
    end
end

function test = not_yet_in_classpath(classpath, filename)
%% Test whether the library was already imported
expression = strcat([filesep filename '$']);
test = isempty(cell2mat(regexp(classpath, expression)));
end
