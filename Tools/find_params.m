function results = find_params(fname)
    PARAM_NAMES = ["noise","ndtct","agls","grdsz"];
    max_l = size(PARAM_NAMES,2);
    results = containers.Map
    for i = 1:max_l
        name = PARAM_NAMES(1,i);
        ind = strfind(fname,name);
        splits = split(fname(1:ind),'_');
        val = splits(end-1);
        disp(strjoin([name,"=",val]));
        results(name) = str2num(cell2mat(val));
        %find a way to create a struct based on strings%
    end
           
           
end