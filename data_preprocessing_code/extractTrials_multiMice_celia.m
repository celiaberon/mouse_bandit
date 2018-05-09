clear
cd ~/'Dropbox (HMS)'/Celia_behaviorFiles/
mice = dir;
for mouse=6:11
    cd(mice(mouse).name)
    dates = dir;    
    for i=1:length(dates)
        if isempty(strfind(dates(i).name, '.')) && isempty(strfind(dates(i).name, 'err'))
            cd(dates(i).name)
            if length(dir)>4
                temp = ls('*.csv');
                 % only load in data if _trials.csv doesn't already exist:
                if isempty(findstr('parameters.csv', temp))
                    matFiles = dir('*.mat');
                    if size(matFiles,1) ~= 0 % check that .mat files exist
                        %loads the stats, pokeHistory, and parameters
                        for iFile = 1:size(matFiles,1)
                            load(matFiles(iFile).name);
                        end
                    end
                    try
                        trials = extractTrials(stats, pokeHistory, false);
                        trial_filename = [dates(i).name, '_', mice(mouse).name, '_trials.csv'];
                        csvwrite(trial_filename, trials);
                        p_filename = [dates(i).name,'_',mice(mouse).name, '_parameters','.csv'];
                        writetable(struct2table(p),p_filename);
                    catch ME
                        if ME.stack(1).line == 78
                            warning('no Trials completed for %s on %s', mice(mouse).name, dates(i).name);
                        else
                            warning('unidentified problem with trial extraction for %s on %s', mice(mouse).name, dates(i).name)
                        end
                    end
                end
            else warning('No files found for %s on %s', mice(mouse).name, dates(i).name)
            end
            cd ..
        end
        clear temp pokeHistory stats trial_filename p_filename trials
    end
    cd ..
end 